import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx
from fastapi.testclient import TestClient

from app.codex_app_server import CodexAppServerClient, is_codex, resolve_codex_model
from app.codex_runner import generate_codex
from app.config import ServerSettings
from app.executor import JobExecutor
from app.server import create_app
from gemini_text import run_job


FAKE_CODEX_SCRIPT = r"""
import json, sys

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"serverInfo": {"name": "fake-codex", "version": "0.1"}}}) + "\n")
        sys.stdout.flush()
    elif method == "initialized":
        pass
    elif method == "thread/start":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"thread": {"id": "thr_test_1"}}}) + "\n")
        sys.stdout.flush()
    elif method == "thread/resume":
        tid = msg.get("params", {}).get("threadId", "thr_test_1")
        sys.stdout.write(json.dumps({"id": req_id, "result": {"thread": {"id": tid}}}) + "\n")
        sys.stdout.flush()
    elif method == "turn/start":
        sys.stdout.write(json.dumps({"id": req_id, "result": {"turn": {"id": "turn_test_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"method": "turn/started", "params": {"turn": {"id": "turn_test_1"}}}) + "\n")
        sys.stdout.write(json.dumps({"id": 99, "method": "execCommandApproval", "params": {"command": ["ls"]}}) + "\n")
        sys.stdout.flush()
        # Wait for approval response
        approval_raw = sys.stdin.readline()
        approval = json.loads(approval_raw)
        if approval.get("result", {}).get("decision") == "accept":
            sys.stdout.write(json.dumps({"method": "item/reasoning/summaryTextDelta", "params": {"delta": "Thinking..."}}) + "\n")
            sys.stdout.write(json.dumps({"method": "item/started", "params": {"item": {"id": "cmd_1", "type": "commandExecution", "command": "ls"}}}) + "\n")
            sys.stdout.write(json.dumps({"method": "item/completed", "params": {"item": {"id": "cmd_1", "type": "commandExecution", "status": "completed", "output": "ok"}}}) + "\n")
            sys.stdout.write(json.dumps({"method": "item/agentMessage/delta", "params": {"delta": "Hello "}}) + "\n")
            sys.stdout.write(json.dumps({"method": "item/agentMessage/delta", "params": {"delta": "from Codex"}}) + "\n")
            sys.stdout.write(json.dumps({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn_test_1", "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
            }) + "\n")
        else:
            sys.stdout.write(json.dumps({
                "method": "turn/completed",
                "params": {"turn": {"id": "turn_test_1", "status": "failed", "error": {"message": "declined"}}},
            }) + "\n")
        sys.stdout.flush()
"""


class FakeScriptClient(CodexAppServerClient):
    def __init__(self, script_path: Path, **kwargs):
        kwargs.pop("codex_bin", None)
        super().__init__(codex_bin=sys.executable, **kwargs)
        self._script_path = script_path

    async def start(self):
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())
        result = await self.request("initialize", {"clientInfo": {"name": "test"}}, timeout=10)
        await self.notify("initialized", {})
        return result


class FakePB:
    def __init__(self, job):
        self.job = job
        self.calls = []
        self.token = "tok"

    async def authenticate(self, **kwargs):
        return None

    async def post(self, path: str, data: dict, *, retries=0):
        self.calls.append((path, data))
        if path.endswith("/claim"):
            return self.job
        if path.endswith("/start"):
            return {"send": True}
        if path.endswith("/progress"):
            return {"accepted": True}
        if path.endswith("/complete"):
            return {"ok": True}
        return {}

    async def checkpoint(self, path: str, identity: dict, state=None):
        self.calls.append((path, {"identity": identity, "state": state}))
        return {"canceled": False}


class CodexAndServerTests(unittest.TestCase):
    def test_is_codex_and_model_resolution(self):
        self.assertTrue(is_codex("Codex"))
        self.assertTrue(is_codex("codex:gpt-5.3-codex"))
        self.assertTrue(is_codex("gpt-5.3-codex"))
        self.assertFalse(is_codex("Antigravity"))
        self.assertFalse(is_codex("gemini-2.5-pro"))

        self.assertEqual(
            resolve_codex_model({"model": "Codex", "options": {"agent": {"model": "gpt-5.3-codex"}}}),
            "gpt-5.3-codex",
        )
        self.assertEqual(
            resolve_codex_model({"model": "codex:o3"}),
            "o3",
        )
        self.assertEqual(
            resolve_codex_model({"model": "Codex"}, default_model="gpt-5.4"),
            "gpt-5.4",
        )

    def test_generate_codex_stdio_protocol_and_approvals(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "fake_codex.py"
                script.write_text(FAKE_CODEX_SCRIPT, encoding="utf-8")
                checkpoints = []

                async def cp(state=None):
                    checkpoints.append(state)
                    return {"canceled": False}

                settings = ServerSettings(
                    pb_url="http://127.0.0.1:8090",
                    codex_cwd=tmp,
                    codex_approvals_reviewer="auto_review",
                    recovery_dir=Path(tmp) / "rec",
                )
                result = await generate_codex(
                    {
                        "id": "job123456789012",
                        "model": "Codex",
                        "prompt": "Summarize repo",
                        "options": {"agent": {"model": "gpt-5.3-codex"}, "thinkingLevel": "high"},
                    },
                    {"id": "msg123456789012", "inputText": ""},
                    checkpoint=cp,
                    settings=settings,
                    client_factory=lambda **kw: FakeScriptClient(script, **kw),
                )
                self.assertEqual(result["reply"], "Hello from Codex")
                self.assertEqual(result["error"], "")
                self.assertEqual(result["protocol"], "codex-app-server")
                self.assertEqual(result["thinkingSummary"], "Thinking...")
                self.assertEqual(result["usage"]["totalTokenCount"], 15)
                self.assertEqual(result["usage"]["agentStepCount"], 1)
                self.assertEqual(result["response"]["version"], 1)
                self.assertTrue(any(isinstance(c, dict) and c.get("threadId") == "thr_test_1" for c in checkpoints))

        asyncio.run(scenario())

    def test_run_job_routes_codex_without_gemini_keys(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "fake_codex.py"
                script.write_text(FAKE_CODEX_SCRIPT, encoding="utf-8")
                pb = FakePB({
                    "id": "aaaaaaaaaaaaaaa",
                    "model": "Codex",
                    "prompt": "Fix bug",
                    "options": {"concurrency": 1, "agent": {"model": "gpt-5.3-codex"}},
                    "messages": [{"id": "bbbbbbbbbbbbbbb", "index": 1, "status": "queued", "inputText": ""}],
                })
                settings = ServerSettings(
                    pb_url="http://127.0.0.1:8090",
                    codex_cwd=tmp,
                    codex_approvals_reviewer="auto_review",
                    recovery_dir=Path(tmp) / "rec",
                )

                # Patch CodexAppServerClient in app.codex_runner for this test
                import app.codex_runner as cr
                orig = cr.CodexAppServerClient
                cr.CodexAppServerClient = lambda **kw: FakeScriptClient(script, **kw)
                try:
                    async def fail_pool(_opts):
                        raise AssertionError("Gemini key pool should not be created for Codex jobs")

                    async with httpx.AsyncClient() as client:
                        code = await run_job(
                            pb,
                            client,
                            "aaaaaaaaaaaaaaa",
                            "1700000000:1",
                            fail_pool,
                            Path(tmp) / "rec",
                            server_settings=settings,
                            timing_basis="Self-hosted FastAPI server wall clock",
                        )
                    self.assertEqual(code, 0)
                    complete_calls = [c for c in pb.calls if c[0].endswith("/complete")]
                    self.assertEqual(len(complete_calls), 1)
                    self.assertEqual(complete_calls[0][1]["reply"], "Hello from Codex")
                    self.assertEqual(complete_calls[0][1]["execution"]["timingBasis"], "Self-hosted FastAPI server wall clock")
                finally:
                    cr.CodexAppServerClient = orig

        asyncio.run(scenario())

    def test_fastapi_server_endpoints_and_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = ServerSettings(
                pb_url="http://127.0.0.1:8090",
                pb_token="pb-tok",
                server_auth_token="secret-123",
                recovery_dir=Path(tmp) / "rec",
            )
            ran = []

            async def fake_run_job(pb, client, job_id, execution_id, pool_factory, work_dir, **kwargs):
                ran.append((job_id, execution_id))
                await asyncio.sleep(0.05)
                return 0

            async def fake_recover(pb, directory):
                return None

            executor = JobExecutor(
                settings,
                run_job_fn=fake_run_job,
                recover_fn=fake_recover,
                pb_factory=lambda client, root: FakePB({}),
            )
            app = create_app(settings, executor=executor)
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "ok")

                unauth = client.post("/internal/jobs/run", json={"job_id": "abcde1234567890"})
                self.assertEqual(unauth.status_code, 401)

                ok = client.post(
                    "/internal/jobs/run",
                    headers={"Authorization": "Bearer secret-123"},
                    json={"job_id": "abcde1234567890"},
                )
                self.assertEqual(ok.status_code, 202)
                self.assertFalse(ok.json()["deduplicated"])

                dup = client.post(
                    "/internal/jobs/run",
                    headers={"Authorization": "Bearer secret-123"},
                    json={"job_id": "abcde1234567890"},
                )
                self.assertEqual(dup.status_code, 202)
                self.assertTrue(dup.json()["deduplicated"])

                jobs = client.get("/internal/jobs", headers={"Authorization": "Bearer secret-123"})
                self.assertEqual(jobs.status_code, 200)
                self.assertEqual(len(jobs.json()["jobs"]), 1)

            self.assertEqual(len(ran), 1)


if __name__ == "__main__":
    unittest.main()
