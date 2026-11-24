"""Server configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from app.native_sessions import CLAUDE_CODE, COMMAND_CODE


DEFAULT_AGY_MODEL = "gemini-4-hydrogen"

def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None, default: int, *, minimum: int = 0) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        return default


def _parse_float(raw: str | None, default: float, *, minimum: float = 0.0) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, float(raw.strip()))
    except ValueError:
        return default


def validate_pb_url(raw: str, *, allow_http: bool = True) -> str:
    root = (raw or "").strip().rstrip("/")
    pattern = r"https?://[^/?#@]+" if allow_http else r"https://[^/?#@]+"
    if not re.fullmatch(pattern, root):
        scheme = "HTTP(S)" if allow_http else "HTTPS"
        raise RuntimeError(f"GEMINI_PB_URL must be a valid {scheme} origin")
    return root


@dataclass(frozen=True)
class ServerSettings:
    pb_url: str
    pb_token: str = ""
    pb_email: str = ""
    pb_password: str = ""
    gemini_api_keys: tuple[str, ...] = ()
    gemini_key_groups: str = ""
    server_auth_token: str = ""
    max_concurrent_jobs: int = 8
    recovery_dir: Path = Path("work/gemini/recovery")
    pb_poll_interval_seconds: float = 0.0
    pb_poll_max_interval_seconds: float = 10.0
    pb_executor_queue: str = "codex"
    codex_bin: str = "codex"
    codex_cwd: str = "work/codex/workspaces"
    codex_sandbox: str = "workspace-write"
    codex_approvals_reviewer: str = "user"
    codex_default_model: str = ""
    codex_turn_timeout_seconds: int = 1800
    agy_bin: str = "agy"
    agy_cwd: str = "work/antigravity-cli/workspaces"
    agy_default_model: str = DEFAULT_AGY_MODEL
    agy_api_key: str = ""
    agy_turn_timeout_seconds: int = 1800
    allow_insecure_pb: bool = True
    claude_code_bin: str = "claude"
    claude_code_cwd: str = "work/claude-code/workspaces"
    claude_code_turn_timeout_seconds: int = 1800
    modelflare_claude_key: str = ""
    modelflare_gpt_key: str = ""
    command_code_bin: str = "cmd"
    command_code_cwd: str = "work/command-code/workspaces"
    command_code_turn_timeout_seconds: int = 1800
    command_code_api_key: str = ""
    command_code_shell_guard: str = ""

    @classmethod
    def from_env(cls) -> "ServerSettings":
        pb_url = validate_pb_url(
            os.environ.get("GEMINI_PB_URL", "https://pb.any1.tech"),
            allow_http=True,
        )
        keys = tuple(
            k.strip()
            for k in os.environ.get("GEMINI_API_KEYS", "").splitlines()
            if k.strip()
        )
        sandbox = (os.environ.get("CODEX_SANDBOX", "workspace-write") or "workspace-write").strip()
        if sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            sandbox = "workspace-write"
        reviewer = (os.environ.get("CODEX_APPROVALS_REVIEWER", "user") or "user").strip()
        if reviewer not in {"auto_review", "guardian_subagent", "user"}:
            reviewer = "user"
        queue = os.environ.get("PB_EXECUTOR_QUEUE", "codex").strip()
        if queue not in {"codex", "gemini", "agy", COMMAND_CODE, CLAUDE_CODE}:
            raise RuntimeError("PB_EXECUTOR_QUEUE must be codex, gemini, agy, command-code or claude-code")
        if queue == "gemini" and not keys:
            raise RuntimeError("GEMINI_API_KEYS is required for the Gemini queue")
        agy_api_key = os.environ.get("AGY_GEMINI_API_KEY", "").strip()
        if queue == "agy" and not (agy_api_key or keys):
            raise RuntimeError("AGY_GEMINI_API_KEY or GEMINI_API_KEYS is required for the Antigravity CLI queue")
        claude_key = os.environ.get("MODELFLARE_CLAUDE_KEY", "").strip()
        gpt_key = os.environ.get("MODELFLARE_GPT_KEY", "").strip()
        if queue == CLAUDE_CODE and not (claude_key and gpt_key):
            raise RuntimeError("The Claude Code queue requires both ModelFlare keys")
        return cls(
            pb_url=pb_url,
            pb_token=os.environ.get("POCKETBASE_SERVER_TOKEN", "").strip(),
            pb_email=os.environ.get("GEMINI_PB_EMAIL", "").strip(),
            pb_password=os.environ.get("GEMINI_PB_PASSWORD", "").strip(),
            gemini_api_keys=keys,
            gemini_key_groups=os.environ.get("GEMINI_KEY_GROUPS", "").strip(),
            server_auth_token=os.environ.get("SERVER_AUTH_TOKEN", "").strip(),
            max_concurrent_jobs=_parse_int(os.environ.get("MAX_CONCURRENT_JOBS"), 8, minimum=1),
            recovery_dir=Path(os.environ.get("RECOVERY_DIR", "work/gemini/recovery")),
            pb_poll_interval_seconds=_parse_float(os.environ.get("PB_POLL_INTERVAL_SECONDS"), 0.0, minimum=0.0),
            pb_poll_max_interval_seconds=_parse_float(os.environ.get("PB_POLL_MAX_INTERVAL_SECONDS"), 10.0, minimum=1.0),
            pb_executor_queue=queue,
            codex_bin=os.environ.get("CODEX_BIN", "codex").strip() or "codex",
            codex_cwd=os.environ.get("CODEX_CWD", "").strip() or "work/codex/workspaces",
            codex_sandbox=sandbox,
            codex_approvals_reviewer=reviewer,
            codex_default_model=os.environ.get("CODEX_DEFAULT_MODEL", "").strip(),
            codex_turn_timeout_seconds=_parse_int(os.environ.get("CODEX_TURN_TIMEOUT_SECONDS"), 1800, minimum=30),
            agy_bin=os.environ.get("AGY_BIN", "").strip() or "agy",
            agy_cwd=os.environ.get("AGY_CWD", "").strip() or "work/antigravity-cli/workspaces",
            agy_default_model=os.environ.get("AGY_DEFAULT_MODEL", DEFAULT_AGY_MODEL).strip(),
            agy_api_key=agy_api_key,
            agy_turn_timeout_seconds=_parse_int(os.environ.get("AGY_TURN_TIMEOUT_SECONDS"), 1800, minimum=30),
            allow_insecure_pb=_parse_bool(os.environ.get("ALLOW_INSECURE_PB"), True),
            claude_code_bin=os.environ.get("CLAUDE_CODE_BIN", "").strip() or "claude",
            claude_code_cwd=os.environ.get("CLAUDE_CODE_CWD", "").strip() or "work/claude-code/workspaces",
            claude_code_turn_timeout_seconds=_parse_int(os.environ.get("CLAUDE_CODE_TURN_TIMEOUT_SECONDS"), 1800, minimum=30),
            modelflare_claude_key=claude_key,
            modelflare_gpt_key=gpt_key,
            command_code_bin=os.environ.get("COMMAND_CODE_BIN", "").strip() or "cmd",
            command_code_cwd=os.environ.get("COMMAND_CODE_CWD", "").strip() or "work/command-code/workspaces",
            command_code_turn_timeout_seconds=_parse_int(os.environ.get("COMMAND_CODE_TURN_TIMEOUT_SECONDS"), 1800, minimum=30),
            command_code_api_key=os.environ.get("COMMAND_CODE_API_KEY", "").strip(),
            command_code_shell_guard=os.environ.get("COMMAND_CODE_SHELL_GUARD", "").strip(),
        )
