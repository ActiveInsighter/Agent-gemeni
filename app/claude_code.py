"""Claude Code CLI transport for PocketBase Agent conversations."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
import re
import signal
import time

from agent_timeline import MAX_PARTS, bounded_part, tool_part
from app.native_sessions import CLAUDE_CODE, native_prompt, native_session
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES
from response_capture import sanitize

CLAUDE_MODELS = {'claude-sonnet-5-5', 'claude-opus-5-5'}
GPT_MODELS = {'gpt-6.2-sol', 'gpt-6.2-astra'}
EFFORTS = {'low', 'medium', 'high', 'xhigh', 'max'}
SESSION_ID = re.compile(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}')


class ClaudeTimeline:
    def __init__(self, progress, secrets):
        self.progress = progress
        self.secrets = secrets
        self.parts = {}
        self.calls = {}
        self.partial_text = {}

    def publish(self, key, part):
        if key not in self.parts and len(self.parts) >= MAX_PARTS:
            return
        if part['type'] == 'text':
            clean_text = sanitize(part['text'], secrets=self.secrets, text_limit=MAX_REPLY_BYTES, parse_strings=False)
            part = {**part, 'text': clean_text}
        else:
            part = bounded_part(sanitize(part, secrets=self.secrets))
        self.parts[key] = part
        if self.progress is not None:
            self.progress.publish(key, part)

    def consume(self, message, *, partial=False):
        for index, block in enumerate(message.get('content') or ()):
            kind = block.get('type')
            key = f"{message.get('id')}:{block.get('index', index)}"
            if kind == 'text':
                content = block.get(kind) or ''
                if partial:
                    if key not in self.parts and len(self.parts) >= MAX_PARTS:
                        continue
                    previous = self.partial_text.get(key, '')
                    if len(previous) >= 64000:
                        continue
                    content = previous + content
                    self.partial_text[key] = content
                else:
                    self.partial_text.pop(key, None)
                if len(content.encode()) > MAX_REPLY_BYTES:
                    raise RuntimeError('Claude Code reply exceeds 2 MiB')
                if content:
                    self.publish(key, {'type': 'text', 'text': content})
            elif kind == 'tool_use' and block.get('id'):
                call_id = block['id']
                self.publish(key, tool_part(block.get('name') or 'tool', call_id, CLAUDE_CODE, input=block.get('input')))
                if key in self.parts:
                    self.calls[call_id] = key
            elif kind == 'tool_result' and block.get('tool_use_id') in self.calls:
                call_id = block['tool_use_id']
                tool_key = self.calls[call_id]
                previous = self.parts.get(tool_key)
                if previous:
                    self.publish(tool_key, tool_part(
                        previous['toolName'], call_id, CLAUDE_CODE,
                        input=previous.get('input'), output=block.get('content'),
                        status='failed' if block.get('is_error') else 'completed',
                    ))


def token_usage(raw):
    fields = {
        'input_tokens': 'promptTokenCount',
        'output_tokens': 'candidatesTokenCount',
        'cache_read_input_tokens': 'cachedContentTokenCount',
    }
    usage = {target: raw[source] for source, target in fields.items()
             if type(raw.get(source)) is int and raw[source] >= 0}
    if 'promptTokenCount' in usage:
        for field in ('cache_creation_input_tokens', 'cache_read_input_tokens'):
            count = raw.get(field)
            if type(count) is int and count >= 0:
                usage['promptTokenCount'] += count
        usage['totalTokenCount'] = usage['promptTokenCount'] + usage.get('candidatesTokenCount', 0)
    return usage


async def generate_claude_code(job, message, *, checkpoint, progress, settings, images=()):
    options = job.get('options') or {}
    model = (options.get('agent') or {}).get('model')
    if model in CLAUDE_MODELS:
        key = settings.modelflare_claude_key
    elif model in GPT_MODELS:
        key = settings.modelflare_gpt_key
    else:
        raise RuntimeError('Unsupported Claude Code model')
    if not key:
        raise RuntimeError('ModelFlare credential missing for Claude Code model')
    if checkpoint is None:
        raise RuntimeError('Claude Code requires durable execution state')
    if (await checkpoint()).get('canceled'):
        raise AgentCanceled('Claude Code canceled')

    previous_id, workspace_id, cwd = native_session(message, CLAUDE_CODE, settings.claude_code_cwd, 'conversationId')
    if previous_id and not SESSION_ID.fullmatch(previous_id):
        raise RuntimeError('Invalid Claude Code session ID')
    prompt = native_prompt(job, message, include_history=not previous_id)
    if not prompt and not images:
        raise RuntimeError('Claude Code prompt is empty')

    args = [
        settings.claude_code_bin, '--print', '--bare', '--restricted', '--strict-mcp-config',
        '--permission-mode', 'acceptEdits', '--disable-slash-commands',
        '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
        '--include-partial-messages', '--model', model,
    ]
    if previous_id:
        args.extend(('--resume', previous_id))
    effort = options.get('thinkingLevel')
    if effort in EFFORTS:
        args.extend(('--effort', effort))
    instructions = str(job.get('systemPrompt') or '').strip()
    if instructions:
        args.extend(('--append-system-prompt', instructions))
    env = {name: value for name, value in os.environ.items()
           if name in {'PATH', 'HOME', 'LANG', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR'} or name.startswith('LC_')}
    env.update(ANTHROPIC_BASE_URL='https://modelflare.dev', ANTHROPIC_API_KEY=key,
               DISABLE_TELEMETRY='1', DISABLE_NON_ESSENTIAL_MODEL_CALLS='1')
    if effort == 'off':
        env['MAX_THINKING_TOKENS'] = '0'

    blocks = [{'type': 'text', 'text': prompt}] if prompt else []
    blocks.extend({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data}}
                  for mime, data in images)
    request = {'type': 'user', 'message': {'role': 'user', 'content': blocks}}
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, env=env, start_new_session=True, limit=16 * 1024 * 1024,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    secrets = (settings.modelflare_claude_key, settings.modelflare_gpt_key)
    timeline = ClaudeTimeline(progress, secrets)
    session_id = ''
    message_id = ''
    result = {}
    deadline = time.monotonic() + settings.claude_code_turn_timeout_seconds
    next_check = time.monotonic() + 2.5
    try:
        proc.stdin.write((json.dumps(request, ensure_ascii=False) + '\n').encode())
        proc.stdin.close()
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError('Claude Code turn timed out')
            if now >= next_check:
                if (await checkpoint()).get('canceled'):
                    raise AgentCanceled('Claude Code canceled')
                next_check = time.monotonic() + 2.5
                continue
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), min(deadline, next_check) - now)
            except TimeoutError:
                continue
            if not line:
                break
            event = json.loads(line)
            kind = event.get('type')
            if kind == 'system' and event.get('subtype') == 'init':
                session_id = event.get('session_id') or ''
                if not SESSION_ID.fullmatch(session_id) or (previous_id and previous_id != session_id):
                    raise RuntimeError('Claude Code did not open the requested session')
                state = {'provider': CLAUDE_CODE, 'conversationId': session_id, 'workspaceId': workspace_id}
                if (await checkpoint(state)).get('canceled'):
                    raise AgentCanceled('Claude Code canceled')
            elif kind == 'stream_event' and not event.get('parent_tool_use_id'):
                data = event.get('event') or {}
                if data.get('type') == 'message_start':
                    message_id = (data.get('message') or {}).get('id')
                elif data.get('type') == 'content_block_delta' and message_id:
                    delta = data.get('delta') or {}
                    if delta.get('type') == 'text_delta':
                        timeline.consume({'id': message_id, 'content': [
                            {'type': 'text', 'text': delta.get('text'), 'index': data.get('index')},
                        ]}, partial=True)
            elif kind in {'assistant', 'user'} and not event.get('parent_tool_use_id'):
                timeline.consume(event.get('message') or {})
            elif kind == 'result':
                result = event
        code = await asyncio.wait_for(proc.wait(), 5)
        reply = sanitize(str(result.get('result') or '').strip(), secrets=secrets,
                         text_limit=MAX_REPLY_BYTES, parse_strings=False)
        if len(reply.encode()) > MAX_REPLY_BYTES:
            raise RuntimeError('Claude Code reply exceeds 2 MiB')
        if code or not session_id or result.get('session_id') != session_id or result.get('is_error') or not reply:
            detail = str(result.get('subtype') or 'no result')
            raise RuntimeError(f'Claude Code could not complete the turn ({detail}, exit {code})')
        if not any(part['type'] == 'text' for part in timeline.parts.values()):
            timeline.publish('answer', {'type': 'text', 'text': reply})
        usage = token_usage(result.get('usage') or {})
        steps = sum(part['type'] == 'server_tool' for part in timeline.parts.values())
        if steps:
            usage['agentStepCount'] = steps
        return {'reply': reply, 'error': '', 'finishReason': 'STOP', 'protocol': CLAUDE_CODE,
                'parts': list(timeline.parts.values()), 'usage': usage}
    finally:
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), 3)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
