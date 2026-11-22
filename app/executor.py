"""Background job executor and PocketBase queue poller for the FastAPI server."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
from pathlib import Path
import re
import time
from typing import Any

import httpx

from app.config import ServerSettings
from gemini_text import PocketBase, create_key_pool, recover, run_job

JOB_ID_RE = re.compile(r"^[a-z0-9]{15}$")


@dataclass
class JobRecord:
    job_id: str
    execution_id: str
    status: str = "queued"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None


class JobExecutor:
    def __init__(
        self,
        settings: ServerSettings,
        *,
        run_job_fn=run_job,
        recover_fn=recover,
        pb_factory=PocketBase,
        client_factory=None,
    ):
        self.settings = settings
        self._run_job = run_job_fn
        self._recover = recover_fn
        self._pb_factory = pb_factory
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(follow_redirects=False))

        self.client: httpx.AsyncClient | None = None
        self.pb: PocketBase | None = None
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._seq = itertools.count(1)
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._records: dict[str, JobRecord] = {}
        self._poller_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def startup(self) -> None:
        self.settings.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._client_factory()
        self.pb = self._pb_factory(self.client, self.settings.pb_url)
        await self.pb.authenticate(
            token=self.settings.pb_token,
            email=self.settings.pb_email,
            password=self.settings.pb_password,
        )
        await self.run_recovery()
        if self.settings.pb_poll_interval_seconds > 0:
            self._poller_task = asyncio.create_task(self._poll_loop())

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._poller_task is not None and not self._poller_task.done():
            self._poller_task.cancel()
            await asyncio.gather(self._poller_task, return_exceptions=True)
        if self._active_tasks:
            tasks = list(self._active_tasks.values())
            _, pending = await asyncio.wait(tasks, timeout=15)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def run_recovery(self) -> int:
        if self.pb is None:
            raise RuntimeError("Executor is not initialized")
        before = self.pending_recovery_count()
        await self._recover(self.pb, self.settings.recovery_dir)
        after = self.pending_recovery_count()
        return max(0, before - after)

    def pending_recovery_count(self) -> int:
        if not self.settings.recovery_dir.exists():
            return 0
        return sum(1 for p in self.settings.recovery_dir.glob("*.json") if not p.name.endswith(".progress.json"))

    def next_execution_id(self) -> str:
        return f"{int(time.time() * 1000)}:{next(self._seq)}"

    def submit_job(self, job_id: str) -> tuple[JobRecord, bool]:
        if not JOB_ID_RE.fullmatch(job_id):
            raise ValueError("Invalid job_id: expected 15-character lowercase alphanumeric PocketBase ID")
        if job_id in self._active_tasks and not self._active_tasks[job_id].done():
            return self._records[job_id], False
        if self.pb is None or self.client is None:
            raise RuntimeError("Executor is not initialized")

        execution_id = self.next_execution_id()
        record = JobRecord(job_id=job_id, execution_id=execution_id, status="queued")
        self._records[job_id] = record
        task = asyncio.create_task(self._execute_job(job_id, execution_id, record))
        self._active_tasks[job_id] = task
        return record, True

    async def _execute_job(self, job_id: str, execution_id: str, record: JobRecord) -> None:
        assert self.pb is not None and self.client is not None
        try:
            async with self.semaphore:
                record.status = "running"
                keys = list(self.settings.gemini_api_keys)

                async def pool_factory(options: dict[str, Any]):
                    if not keys:
                        raise RuntimeError("GEMINI_API_KEYS is required for Gemini/Antigravity jobs")
                    return await create_key_pool(keys, options, self.settings.gemini_key_groups)

                async def fast_pool_factory(options: dict[str, Any]):
                    if not keys:
                        raise RuntimeError("GEMINI_API_KEYS is required")
                    return await create_key_pool(keys, {"rpmPerKey": 15, "rpdPerKey": 500}, self.settings.gemini_key_groups)

                code = await self._run_job(
                    self.pb,
                    self.client,
                    job_id,
                    execution_id,
                    pool_factory,
                    self.settings.recovery_dir,
                    fast_pool_factory=fast_pool_factory if keys else None,
                    server_settings=self.settings,
                    timing_basis="Self-hosted FastAPI server wall clock",
                )
                record.exit_code = code
                record.status = "completed" if code == 0 else "failed"
        except asyncio.CancelledError:
            record.status = "canceled"
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)[:500]
        finally:
            record.finished_at = datetime.now(timezone.utc).isoformat()
            self._active_tasks.pop(job_id, None)

    async def _poll_loop(self) -> None:
        assert self.client is not None and self.pb is not None
        interval = max(1.0, self.settings.pb_poll_interval_seconds)
        while not self._stop_event.is_set():
            try:
                resp = await self.client.get(
                    f"{self.settings.pb_url}/api/collections/aw_gemini_jobs/records",
                    headers={"Authorization": self.pb.token},
                    params={
                        "filter": '(status="queued" || status="dispatching") && cancelRequested=false',
                        "sort": "created",
                        "perPage": 20,
                        "fields": "id,status",
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    items = resp.json().get("items") or []
                    for item in items:
                        jid = str(item.get("id") or "").strip()
                        if JOB_ID_RE.fullmatch(jid) and jid not in self._active_tasks:
                            self.submit_job(jid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[server.poller] poll_error error_type={type(exc).__name__}", flush=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "job_id": r.job_id,
                "execution_id": r.execution_id,
                "status": r.status,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "exit_code": r.exit_code,
                "error": r.error,
                "active": r.job_id in self._active_tasks,
            }
            for r in list(self._records.values())[-100:]
        ]

    def active_count(self) -> int:
        return len(self._active_tasks)
