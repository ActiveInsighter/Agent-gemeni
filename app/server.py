"""FastAPI entrypoint for self-hosted AnyWorkflow Gemini + Codex execution server."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.codex_app_server import CodexAppServerClient
from app.config import ServerSettings
from app.executor import JobExecutor


class RunJobRequest(BaseModel):
    job_id: str = Field(..., min_length=15, max_length=15, pattern=r"^[a-z0-9]{15}$")


def create_app(
    settings: ServerSettings | None = None,
    *,
    executor: JobExecutor | None = None,
) -> FastAPI:
    cfg = settings or ServerSettings.from_env()
    exec_instance = executor or JobExecutor(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.executor = exec_instance
        await exec_instance.startup()
        try:
            yield
        finally:
            await exec_instance.shutdown()

    app = FastAPI(
        title="AnyWorkflow Agent Server",
        version="1.0.0",
        description="Self-hosted FastAPI worker for PocketBase Gemini, Antigravity, and Codex App Server jobs.",
        lifespan=lifespan,
    )

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        token = cfg.server_auth_token
        if not token:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        provided = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided, token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "pb_url": cfg.pb_url,
            "active_jobs": exec_instance.active_count(),
            "max_concurrent_jobs": cfg.max_concurrent_jobs,
            "pending_recovery": exec_instance.pending_recovery_count(),
            "gemini_keys_configured": len(cfg.gemini_api_keys),
            "pb_polling_enabled": cfg.pb_poll_interval_seconds > 0,
            "pb_polling_ready": exec_instance.poll_ready,
        }

    @app.get("/codex/status", dependencies=[Depends(require_auth)])
    async def codex_status() -> dict[str, Any]:
        try:
            async with CodexAppServerClient(codex_bin=cfg.codex_bin, cwd=cfg.codex_cwd) as codex:
                account = (await codex.account_read()).get("account") or {}
                models_resp = await codex.model_list()
                raw_models = models_resp.get("models") or models_resp.get("data") or []
                model_ids = [
                    str(m.get("id") or m.get("model") or "")
                    for m in raw_models
                    if isinstance(m, dict) and (m.get("id") or m.get("model"))
                ]
                return {
                    "available": True,
                    "authenticated": bool(account),
                    "account_type": account.get("type"),
                    "models": model_ids,
                }
        except Exception as exc:
            return {
                "available": False,
                "authenticated": False,
                "error": str(exc)[:300],
                "models": [],
            }

    @app.post(
        "/internal/jobs/run",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_auth)],
    )
    async def run_job_endpoint(payload: RunJobRequest) -> dict[str, Any]:
        try:
            record, created = exec_instance.submit_job(payload.job_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return {
            "accepted": True,
            "deduplicated": not created,
            "job_id": record.job_id,
            "execution_id": record.execution_id,
            "status": record.status,
        }

    @app.get("/internal/jobs", dependencies=[Depends(require_auth)])
    async def list_jobs() -> dict[str, Any]:
        return {"jobs": exec_instance.list_jobs()}

    @app.post("/internal/recover", dependencies=[Depends(require_auth)])
    async def trigger_recover() -> dict[str, Any]:
        try:
            flushed = await exec_instance.run_recovery()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return {
            "flushed": flushed,
            "pending_recovery": exec_instance.pending_recovery_count(),
        }

    return app


def get_app() -> FastAPI:
    return create_app()
