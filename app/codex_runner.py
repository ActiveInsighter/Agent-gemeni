"""PocketBase-compatible runner for Codex App Server turns."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from agent_progress import MessageProgress
from app.codex_app_server import (
    CodexAppServerClient,
    CodexRpcError,
    map_thinking_to_codex_effort,
    resolve_codex_model,
)
from app.config import ServerSettings
from app.codex_timeline import CodexTimeline
from app.native_sessions import native_prompt, native_session
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES
from response_capture import snapshot


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
    images=(),
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

    prev_thread_id, workspace_id, cwd = native_session(message, 'codex', cfg.codex_cwd, 'threadId')

    async with factory(
        codex_bin=cfg.codex_bin,
        cwd=cwd,
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
            except CodexRpcError as error:
                if not any(text in error.rpc_message.lower() for text in (
                    'no rollout found', 'thread not found', 'no thread found',
                )):
                    raise

        if not thread_id:
            thread_id = await codex.thread_start(
                model=model,
                cwd=cwd,
                sandbox=cfg.codex_sandbox,
                approvals_reviewer=cfg.codex_approvals_reviewer,
                base_instructions=system_prompt,
            )

        prompt = native_prompt(job, message, include_history=not resumed)
        if not prompt and not images:
            raise RuntimeError("Codex prompt is empty")

        if checkpoint is not None and (await checkpoint()).get("canceled"):
            raise AgentCanceled("Codex turn canceled")

        turn_id = await codex.turn_start(
            thread_id,
            prompt,
            model=model,
            effort=effort,
            output_schema=output_schema,
            **({"images": tuple(images)} if images else {}),
        )

        if checkpoint is not None:
            cp = await checkpoint({"provider": "codex", "threadId": thread_id, "turnId": turn_id, "workspaceId": workspace_id})
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
                    first_tool_output = timeline.consume(method, params)
                except ValueError:
                    await codex.turn_interrupt(thread_id, turn_id)
                    raise
                if first_tool_output and progress is not None:
                    # Some versions send the first delta immediately before
                    # item/completed. Upload it while the card is still live.
                    try:
                        await asyncio.wait_for(progress.flush(), timeout=1.5)
                    except Exception:
                        # Any outstanding batch can still be retried by the pump.
                        pass

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
