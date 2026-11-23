"""Use only the session and workspace issued by PocketBase's send permit."""
from __future__ import annotations

from pathlib import Path
import re

ANTIGRAVITY_CLI = 'antigravity-cli'
COMMAND_CODE = 'command-code'


def is_antigravity_cli(model: str) -> bool:
    return model.strip().removeprefix('models/').lower() == ANTIGRAVITY_CLI


def is_command_code(model: str) -> bool:
    return model.strip().removeprefix('models/').lower() == COMMAND_CODE


def native_session(message, provider: str, root: str, id_field: str) -> tuple[str, str, str]:
    state = message.get('previousProviderState')
    session_id = ''
    if isinstance(state, dict) and state.get('provider') == provider:
        value = state.get(id_field)
        if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
            session_id = value
    workspace_id = state.get('workspaceId') if session_id else None
    if not isinstance(workspace_id, str) or not re.fullmatch(r'[a-z0-9]{15}', workspace_id):
        workspace_id = str(message.get('id') or '')
    if not re.fullmatch(r'[a-z0-9]{15}', workspace_id):
        raise RuntimeError('Native agent requires a valid PocketBase message ID')
    cwd = Path(root).resolve() / workspace_id
    cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cwd.resolve() != cwd:
        raise RuntimeError('Native agent workspace must be a directory inside its configured root')
    return session_id, workspace_id, str(cwd)


def native_prompt(job, message, *, include_history: bool) -> str:
    from gemini_agent import strip_image_markers

    chunks = []
    if include_history and isinstance(job.get('history'), list):
        lines = []
        for turn in job['history']:
            if not isinstance(turn, dict):
                continue
            text = strip_image_markers(str(turn.get('text') or ''))
            if text:
                role = str(turn.get('role') or 'user').strip().upper()
                lines.append(f'[{role}]\n{text}')
        if lines:
            chunks.append('Previous conversation:\n' + '\n\n'.join(lines))
    for value in (job.get('prompt'), message.get('inputText')):
        text = strip_image_markers(str(value or ''))
        if text:
            chunks.append(text)
    return '\n\n'.join(chunks).strip()
