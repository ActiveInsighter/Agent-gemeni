"""Async JSON-RPC client for the OpenAI Codex App Server (`codex app-server`).

The Codex App Server uses newline-delimited JSON-RPC 2.0 over stdio with the
`"jsonrpc": "2.0"` field omitted on the wire. Authentication is handled on the
server host via `~/.codex/auth.json` (never via client credentials).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable


NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]


def is_codex(model: str | None) -> bool:
    value = (model or "").strip().lower().removeprefix("models/")
    if not value:
        return False
    return (
        value == "codex"
        or value.startswith(("codex:", "codex-", "codex/"))
        or "-codex" in value
    )


def resolve_codex_model(job: dict[str, Any], default_model: str = "") -> str:
    options = job.get("options") or {}
    agent_opts = options.get("agent") if isinstance(options, dict) else None
    if isinstance(agent_opts, dict):
        explicit = str(agent_opts.get("model") or "").strip()
        if explicit and explicit.lower() != "codex":
            return explicit
    top_agent_model = str(job.get("agentModel") or "").strip()
    if top_agent_model and top_agent_model.lower() != "codex":
        return top_agent_model
    raw_model = str(job.get("model") or "").strip().removeprefix("models/")
    lower = raw_model.lower()
    if lower.startswith("codex:") or lower.startswith("codex/"):
        sub = raw_model.split(":" if ":" in raw_model else "/", 1)[1].strip()
        if sub:
            return sub
    if lower != "codex" and raw_model:
        return raw_model
    return default_model.strip()


def map_thinking_to_codex_effort(thinking_level: str | None) -> str | None:
    value = str(thinking_level or "").strip().lower()
    if not value or value == "auto":
        return None
    aliases = {
        "off": "none",
        "none": "none",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "xhigh",
    }
    return aliases.get(value)


class CodexRpcError(RuntimeError):
    def __init__(self, method: str, code: int | str, message: str):
        self.method = method
        self.code = code
        self.rpc_message = message
        super().__init__(f"Codex RPC {method} failed ({code}): {message}")


class CodexAppServerClient:
    """Bidirectional async JSON-RPC client over `codex app-server` stdio."""

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        cwd: str | Path = ".",
        auto_approve: bool = True,
        on_notification: NotificationHandler | None = None,
        client_name: str = "anyworkflow_agent_server",
        client_version: str = "1.0.0",
    ):
        self.codex_bin = codex_bin
        self.cwd = str(cwd)
        self.auto_approve = auto_approve
        self.on_notification = on_notification
        self.client_name = client_name
        self.client_version = client_version

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._stderr_tail: list[str] = []
        self._closed = False

    async def __aenter__(self) -> "CodexAppServerClient":
        try:
            await self.start()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> dict[str, Any]:
        if self._proc is not None:
            return {}
        cwd_path = Path(self.cwd)
        cwd_path.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            self.codex_bin,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path),
            env={key: value for key, value in os.environ.items()
                 if key in {"PATH", "HOME", "CODEX_HOME", "LANG", "TERM", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR"}
                 or key.startswith("LC_")},
            limit=16 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())
        result = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.client_name,
                    "title": "AnyWorkflow Agent Server",
                    "version": self.client_version,
                },
                "capabilities": {
                    "experimentalApi": False,
                },
            },
            timeout=30,
        )
        await self.notify("initialized", {})
        return result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None:
            if proc.stdin is not None and not proc.stdin.is_closing():
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except Exception:
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(t for t in (self._reader_task, self._stderr_task) if t is not None),
            return_exceptions=True,
        )
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("Codex app-server closed"))
        self._pending.clear()

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Codex app-server process is not running")
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"method": method, "params": params or {}})

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        req_id = next(self._ids)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._send({"id": req_id, "method": method, "params": params or {}})
            msg = await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            if isinstance(err, dict):
                raise CodexRpcError(method, err.get("code", -1), str(err.get("message") or "Unknown error"))
            raise CodexRpcError(method, -1, str(err))
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    async def next_notification(self, timeout: float | None = None) -> tuple[str, dict[str, Any]]:
        if timeout is None:
            return await self._notifications.get()
        return await asyncio.wait_for(self._notifications.get(), timeout=timeout)

    async def _read_stdout_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg and "method" not in msg:
                    fut = self._pending.get(msg["id"])
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                elif "id" in msg and "method" in msg:
                    await self._handle_server_request(msg)
                elif "method" in msg:
                    method = str(msg["method"])
                    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
                    await self._notifications.put((method, params))
                    if self.on_notification is not None:
                        res = self.on_notification(method, params)
                        if asyncio.iscoroutine(res):
                            await res
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(exc)
        finally:
            stderr_msg = "\n".join(self._stderr_tail[-10:]).strip()
            detail = f"Codex app-server exited ({stderr_msg})" if stderr_msg else "Codex app-server stream ended"
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(RuntimeError(detail))

    async def _read_stderr_loop(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                raw_line = await self._proc.stderr.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
                    if len(self._stderr_tail) > 50:
                        self._stderr_tail.pop(0)
        except asyncio.CancelledError:
            raise

    async def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg["id"]
        method = str(msg.get("method") or "")
        if method in {
            "execCommandApproval",
            "applyPatchApproval",
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            decision = "accept" if self.auto_approve else "decline"
            await self._send({"id": req_id, "result": {"decision": decision}})
            return
        await self._send({
            "id": req_id,
            "error": {"code": -32601, "message": f"Unsupported server request: {method}"},
        })

    # ---- High-level Codex App Server operations ----

    async def account_read(self) -> dict[str, Any]:
        return await self.request("account/read", {}, timeout=15)

    async def model_list(self) -> dict[str, Any]:
        return await self.request("model/list", {}, timeout=15)

    async def thread_start(
        self,
        *,
        model: str = "",
        cwd: str = "",
        sandbox: str = "workspace-write",
        approvals_reviewer: str = "user",
        base_instructions: str = "",
    ) -> str:
        params: dict[str, Any] = {
            "cwd": cwd or self.cwd,
            "sandbox": sandbox,
            "approvalPolicy": "on-request" if self.auto_approve else "never",
            "approvalsReviewer": approvals_reviewer,
        }
        if model:
            params["model"] = model
        if base_instructions:
            params["baseInstructions"] = base_instructions
        result = await self.request("thread/start", params, timeout=30)
        thread = result.get("thread") if isinstance(result.get("thread"), dict) else {}
        thread_id = str(result.get("threadId") or thread.get("id") or "").strip()
        if not thread_id:
            raise RuntimeError("Codex thread/start did not return a threadId")
        return thread_id

    async def thread_resume(
        self,
        thread_id: str,
        *,
        model: str = "",
        approvals_reviewer: str = "",
        sandbox: str = "workspace-write",
    ) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "excludeTurns": True,
            "cwd": self.cwd,
            "sandbox": sandbox,
            "approvalPolicy": "on-request" if self.auto_approve else "never",
        }
        if model:
            params["model"] = model
        if approvals_reviewer:
            params["approvalsReviewer"] = approvals_reviewer
        result = await self.request("thread/resume", params, timeout=30)
        thread = result.get("thread") if isinstance(result.get("thread"), dict) else {}
        resumed = str(result.get("threadId") or thread.get("id") or "").strip()
        if resumed != thread_id:
            raise RuntimeError("Codex did not resume the requested thread")
        return resumed

    async def turn_start(
        self,
        thread_id: str,
        prompt: str,
        *,
        model: str = "",
        effort: str | None = None,
        output_schema: dict[str, Any] | None = None,
        images: tuple[tuple[str, str], ...] = (),
    ) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [
                *([{"type": "text", "text": prompt}] if prompt else []),
                *({"type": "image", "url": f"data:{mime};base64,{data}"} for mime, data in images),
            ],
        }
        if model:
            params["model"] = model
        if effort:
            params["effort"] = effort
        if output_schema is not None:
            params["outputSchema"] = output_schema
        result = await self.request("turn/start", params, timeout=30)
        turn = result.get("turn") if isinstance(result.get("turn"), dict) else {}
        turn_id = str(result.get("turnId") or turn.get("id") or "").strip()
        if not turn_id:
            raise RuntimeError("Codex turn/start did not return a turnId")
        return turn_id

    async def turn_interrupt(self, thread_id: str, turn_id: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {"threadId": thread_id}
        if turn_id:
            params["turnId"] = turn_id
        return await self.request("turn/interrupt", params, timeout=15)
