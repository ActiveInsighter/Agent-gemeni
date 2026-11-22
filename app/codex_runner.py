"""PocketBase-compatible runner for Codex App Server turns."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from agent_progress import MessageProgress
from app.codex_app_server import (
    CodexAppServerClient,
    map_thinking_to_codex_effort,
    resolve_codex_model,
)
from app.config import ServerSettings
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES, strip_image_markers


def _extract_previous_thread_id(job: dict[str, Any], message: dict[str, Any]) -> str:
    for container in (message, job):
        if not isinstance(container, dict):
            continue
        for key in ("previousProviderState", "providerState"):
            state = container.get(key)
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except ValueError:
                    state = None
            if isinstance(state, dict):
                tid = str(state.get("threadId") or "").strip()
                if tid:
                    return tid
    return ""


def _build_codex_prompt(job: dict[str, Any], message: dict[str, Any], *, include_history: bool) -> str:
    chunks: list[str] = []
    if include_history and isinstance(job.get("history"), list) and job["history"]:
        history_lines: list[str] = []
        for turn in job["history"]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "user").strip().upper()
            text = strip_image_markers(str(turn.get("text") or ""))
            if text:
                history_lines.append(f"[{role}]\n{text}")
        if history_lines:
            chunks.append("Previous conversation:\n" + "\n\n".join(history_lines))
    for value in (job.get("prompt", ""), message.get("inputText", "")):
        stripped = strip_image_markers(str(value or ""))
        if stripped:
            chunks.append(stripped)
    return "\n\n".join(chunks).strip()


def _extract_usage(turn: dict[str, Any], params: dict[str, Any]) -> dict[str, int]:
    raw = turn.get("usage") or params.get("usage") or {}
    if not isinstance(raw, dict):
        return {}
    mapping = {
        "input_tokens": "promptTokenCount",
        "prompt_tokens": "promptTokenCount",
        "promptTokenCount": "promptTokenCount",
        "output_tokens": "candidatesTokenCount",
        "completion_tokens": "candidatesTokenCount",
        "candidatesTokenCount": "candidatesTokenCount",
        "reasoning_tokens": "thoughtsTokenCount",
        "reasoning_output_tokens": "thoughtsTokenCount",
        "thoughtsTokenCount": "thoughtsTokenCount",
        "cached_tokens": "cachedContentTokenCount",
        "cached_input_tokens": "cachedContentTokenCount",
        "cachedContentTokenCount": "cachedContentTokenCount",
        "total_tokens": "totalTokenCount",
        "totalTokenCount": "totalTokenCount",
    }
    usage: dict[str, int] = {}
    for src, dst in mapping.items():
        val = raw.get(src)
        if isinstance(val, int) and val >= 0:
            usage[dst] = val
    if "totalTokenCount" not in usage and ("promptTokenCount" in usage or "candidatesTokenCount" in usage):
        usage["totalTokenCount"] = usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)
    return usage


def _extract_item_text(item: dict[str, Any]) -> str:
    if isinstance(item.get("text"), str):
        return item["text"]
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and isinstance(c.get("text"), str)
        ]
        return "".join(parts)
    return ""


async def generate_codex(
    job: dict[str, Any],
    message: dict[str, Any],
    *,
    checkpoint=None,
    progress: MessageProgress | None = None,
    settings: ServerSettings | None = None,
    client_factory=None,
) -> dict[str, Any]:
    cfg = settings or ServerSettings.from_env()
    factory = client_factory or CodexAppServerClient
    options = job.get("options") or {}
    model = resolve_codex_model(job, default_model=cfg.codex_default_model)
    effort = map_thinking_to_codex_effort(options.get("thinkingLevel"))
    system_prompt = str(job.get("systemPrompt") or "").strip()
    output_schema = options.get("responseJsonSchema") or options.get("responseSchema")
    if not isinstance(output_schema, dict):
        output_schema = None

    prev_thread_id = _extract_previous_thread_id(job, message)

    async with factory(
        codex_bin=cfg.codex_bin,
        cwd=cfg.codex_cwd,
        auto_approve=(cfg.codex_approvals_reviewer != "user"),
    ) as codex:
        thread_id = ""
        resumed = False
        if prev_thread_id:
            try:
                thread_id = await codex.thread_resume(
                    prev_thread_id,
                    model=model,
                    approvals_reviewer=cfg.codex_approvals_reviewer,
                )
                resumed = bool(thread_id)
            except Exception:
                thread_id = ""
                resumed = False

        if not thread_id:
            thread_id = await codex.thread_start(
                model=model,
                cwd=cfg.codex_cwd,
                sandbox=cfg.codex_sandbox,
                approvals_reviewer=cfg.codex_approvals_reviewer,
                base_instructions=system_prompt,
                effort=effort,
            )

        prompt = _build_codex_prompt(job, message, include_history=not resumed)
        if not prompt:
            raise RuntimeError("Codex prompt is empty")

        turn_id = await codex.turn_start(
            thread_id,
            prompt,
            model=model,
            effort=effort,
            output_schema=output_schema,
        )

        if checkpoint is not None:
            cp = await checkpoint({"provider": "codex", "threadId": thread_id, "turnId": turn_id})
            if isinstance(cp, dict) and cp.get("canceled"):
                await codex.turn_interrupt(thread_id, turn_id)
                raise AgentCanceled("Codex turn canceled")

        answer_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        step_seq = 0
        item_to_step: dict[str, int] = {}
        final_turn: dict[str, Any] = {}
        final_params: dict[str, Any] = {}
        turn_error = ""
        finish_reason = "COMPLETED"
        last_cancel_check = time.monotonic()
        deadline = time.monotonic() + cfg.codex_turn_timeout_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await codex.turn_interrupt(thread_id, turn_id)
                raise RuntimeError("Codex turn timed out")

            if checkpoint is not None and (time.monotonic() - last_cancel_check) >= 2.5:
                last_cancel_check = time.monotonic()
                cp = await checkpoint()
                if isinstance(cp, dict) and cp.get("canceled"):
                    await codex.turn_interrupt(thread_id, turn_id)
                    raise AgentCanceled("Codex turn canceled")

            try:
                method, params = await codex.next_notification(timeout=min(2.5, remaining))
            except asyncio.TimeoutError:
                continue

            if method == "item/agentMessage/delta":
                delta = str(params.get("delta") or "")
                if delta:
                    answer_chunks.append(delta)
                    current_answer = "".join(answer_chunks)
                    if len(current_answer.encode("utf-8")) > MAX_REPLY_BYTES:
                        await codex.turn_interrupt(thread_id, turn_id)
                        raise RuntimeError("Codex reply exceeds 2 MiB")
                    if progress is not None:
                        progress.put("answer", "text", current_answer, status="running")

            elif method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}:
                delta = str(params.get("delta") or "")
                if delta:
                    reasoning_chunks.append(delta)
                    if progress is not None:
                        progress.put("reasoning", "reasoning_summary", "".join(reasoning_chunks), status="running")

            elif method == "item/started":
                item = params.get("item") if isinstance(params.get("item"), dict) else {}
                item_type = str(item.get("type") or "")
                item_id = str(item.get("id") or "")
                if item_type in {"commandExecution", "fileChange", "mcpToolCall", "webSearch", "toolCall"}:
                    seq = step_seq
                    step_seq += 1
                    if item_id:
                        item_to_step[item_id] = seq
                    tool_name = str(item.get("tool") or item.get("name") or item_type)
                    payload_preview = json.dumps(
                        {k: v for k, v in item.items() if k not in {"id", "type"}},
                        ensure_ascii=False,
                    )
                    tool_calls.append({
                        "toolName": tool_name,
                        "kind": item_type,
                        "status": "running",
                        "callId": item_id,
                    })
                    if progress is not None:
                        progress.put(
                            f"step:{seq}",
                            "tool_call",
                            payload_preview,
                            status="running",
                            tool_name=tool_name,
                            call_id=item_id,
                            provider_type=item_type,
                            tool_kind="server",
                        )

            elif method == "item/completed":
                item = params.get("item") if isinstance(params.get("item"), dict) else {}
                item_type = str(item.get("type") or "")
                item_id = str(item.get("id") or "")
                if item_type == "agentMessage" and not answer_chunks:
                    fallback_text = _extract_item_text(item)
                    if fallback_text:
                        answer_chunks.append(fallback_text)
                elif item_type in {"commandExecution", "fileChange", "mcpToolCall", "webSearch", "toolCall"}:
                    seq = item_to_step.get(item_id)
                    if seq is None:
                        seq = step_seq
                        step_seq += 1
                    status = str(item.get("status") or "completed")
                    is_failed = status.lower() in {"failed", "error", "declined"}
                    tool_name = str(item.get("tool") or item.get("name") or item_type)
                    for tc in tool_calls:
                        if tc.get("callId") == item_id:
                            tc["status"] = "failed" if is_failed else "completed"
                    if progress is not None:
                        progress.put(
                            f"step:{seq}",
                            "tool_result",
                            json.dumps(
                                {k: v for k, v in item.items() if k not in {"id", "type"}},
                                ensure_ascii=False,
                            ),
                            status="failed" if is_failed else "completed",
                            tool_name=tool_name,
                            call_id=item_id,
                            provider_type=item_type,
                            tool_kind="server",
                        )

            elif method == "turn/completed":
                final_params = params
                final_turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
                status = str(final_turn.get("status") or "completed").lower()
                if status == "interrupted":
                    raise AgentCanceled("Codex turn interrupted")
                if status != "completed":
                    err_obj = final_turn.get("error")
                    if isinstance(err_obj, dict):
                        turn_error = str(err_obj.get("message") or err_obj)
                    else:
                        turn_error = str(err_obj or f"Codex turn ended with status: {status}")
                    finish_reason = status.upper()
                break

        reply = "".join(answer_chunks)
        if not reply and isinstance(final_turn.get("finalResponse"), str):
            reply = final_turn["finalResponse"]
        if not reply and not turn_error:
            turn_error = "Codex 返回空回复"
        reasoning = "".join(reasoning_chunks)
        usage = _extract_usage(final_turn, final_params)
        if step_seq:
            usage["agentStepCount"] = step_seq

        if progress is not None:
            if reply:
                progress.put("answer", "text", reply, status="completed")
            if reasoning:
                progress.put("reasoning", "reasoning_summary", reasoning, status="completed")
            progress.usage.update(usage)

        parts = (
            [
                {k: v for k, v in part.items() if k != "truncated"}
                for part in sorted(progress.parts.values(), key=lambda p: p["sequence"])
            ]
            if progress is not None
            else []
        )

        return {
            "reply": reply,
            "error": turn_error,
            "finishReason": finish_reason,
            "protocol": "codex-app-server",
            "thinkingSummary": reasoning,
            "usage": usage,
            "parts": parts,
            "toolCalls": tool_calls,
            "providerState": {
                "provider": "codex",
                "threadId": thread_id,
                "turnId": turn_id,
            },
            "response": {
                "protocol": "codex-app-server",
                "threadId": thread_id,
                "turnId": turn_id,
                "model": model,
                "metadata": {},
            },
        }
