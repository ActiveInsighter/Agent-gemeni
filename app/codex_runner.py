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
from app.codex_timeline import CodexTimeline
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES, strip_image_markers
from response_capture import snapshot


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
        "inputTokens": "promptTokenCount",
        "outputTokens": "candidatesTokenCount",
        "reasoningOutputTokens": "thoughtsTokenCount",
        "cachedInputTokens": "cachedContentTokenCount",
        "totalTokens": "totalTokenCount",
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
        if not model or effort in {"none", "minimal"}:
            catalog = (await codex.model_list()).get("data") or []
            if not model:
                model = next((entry["id"] for entry in catalog if entry.get("isDefault")), "")
            if not model:
                raise RuntimeError("Codex did not advertise a default model")
            if effort in {"none", "minimal"}:
                selected = next((entry for entry in catalog if entry.get("id") == model), {})
                supported = [entry["reasoningEffort"] for entry in selected.get("supportedReasoningEfforts", [])]
                if supported and effort not in supported:
                    effort = "low" if "low" in supported else supported[0]
        thread_id = ""
        resumed = False
        if prev_thread_id:
            try:
                thread_id = await codex.thread_resume(
                    prev_thread_id,
                    model=model,
                    approvals_reviewer=cfg.codex_approvals_reviewer,
                    sandbox=cfg.codex_sandbox,
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
            )

        prompt = _build_codex_prompt(job, message, include_history=not resumed)
        if not prompt:
            raise RuntimeError("Codex prompt is empty")

        if checkpoint is not None and (await checkpoint()).get("canceled"):
            raise AgentCanceled("Codex turn canceled")

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

        timeline = CodexTimeline(progress, MAX_REPLY_BYTES)
        usage: dict[str, int] = {}
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

            turn = params.get("turn") or {}
            if params.get("threadId") not in (None, thread_id) or (params.get("turnId") or turn.get("id")) not in (None, turn_id):
                continue
            if method == "thread/tokenUsage/updated":
                usage.update(_extract_usage({}, {"usage": (params.get("tokenUsage") or {}).get("last")}))
                if progress is not None:
                    progress.usage.update(usage)
            elif method == "turn/completed":
                final_params, final_turn = params, turn
                for item in turn.get("items") or []:
                    if isinstance(item, dict):
                        timeline.update(item, completed=True)
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

            else:
                try:
                    timeline.consume(method, params)
                except ValueError:
                    await codex.turn_interrupt(thread_id, turn_id)
                    raise

        reply = timeline.answer()
        if not reply and isinstance(final_turn.get("finalResponse"), str):
            reply = final_turn["finalResponse"]
        if not reply and not turn_error:
            turn_error = "Codex 返回空回复"
        if len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
            raise RuntimeError("Codex reply exceeds 2 MiB")
        parts = timeline.parts()
        if reply and not timeline.answer():
            parts.append({"type": "text", "text": reply})
        usage.update(_extract_usage(final_turn, final_params))
        steps = sum(part["type"] in ("tool", "server_tool") for part in parts)
        if steps:
            usage["agentStepCount"] = steps
        if progress is not None:
            progress.usage.update(usage)
        reasoning = "\n\n".join(part["reasoning"] for part in parts if part["type"] == "reasoning")
        return {
            "reply": reply, "error": turn_error, "finishReason": finish_reason,
            "protocol": "codex-app-server", "usage": usage, "parts": parts,
            "thinkingSummary": reasoning,
            "response": snapshot("codex-app-server", [], {"model": model}),
        }
