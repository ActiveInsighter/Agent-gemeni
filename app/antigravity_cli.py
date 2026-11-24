"""Google Antigravity headless CLI adapter for the PocketBase Gemini executor."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
import re
import signal
import time

from agent_timeline import MAX_PARTS, bounded_part, tool_part
from app.command_code import CommandCodeSession, run_acp_turn, stop_acp_process
from app.native_sessions import ANTIGRAVITY_CLI, native_prompt, native_session
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES
from image_attachments import acp_image_blocks
from response_capture import sanitize, snapshot

USAGE_FIELDS = {
    'input_tokens': 'promptTokenCount',
    'output_tokens': 'candidatesTokenCount',
    'thinking_tokens': 'thoughtsTokenCount',
    'cache_read_tokens': 'cachedContentTokenCount',
    'total_tokens': 'totalTokenCount',
}


async def generate_antigravity_cli(job, message, *, checkpoint, progress, settings, images=()):
    previous_id, workspace_id, cwd = native_session(message, ANTIGRAVITY_CLI, settings.agy_cwd, 'conversationId')
    prompt = native_prompt(job, message, include_history=not previous_id)
    if not prompt and not images:
        raise RuntimeError('Antigravity CLI prompt is empty')
    instructions = str(job.get('systemPrompt') or '').strip()
    if instructions:
        prompt = f'System instructions:\n{instructions}\n\nUser request:\n{prompt}'

    options = job.get('options') or {}
    model = (options.get('agent') or {}).get('model') or settings.agy_default_model
    effort = options.get('thinkingLevel')
    args = [
        settings.agy_bin, '--input-format', 'stream-json', '--output-format', 'stream-json',
        '--disable-slash-commands', '--print-timeout', f'{settings.agy_turn_timeout_seconds}s',
    ]
    args.append('--dangerously-skip-permissions' if settings.agy_skip_permissions else '--sandbox')
    if previous_id:
        args.extend(('--conversation', previous_id))
    if model:
        args.extend(('--model', model))
    if effort in {'low', 'medium', 'high', 'xhigh', 'max'}:
        args.extend(('--effort', effort))
    schema = options.get('responseJsonSchema') or options.get('responseSchema')
    if isinstance(schema, dict):
        args.extend(('--json-schema', json.dumps(schema)))

    key = settings.agy_api_key or next(iter(settings.gemini_api_keys), '')
    if not key:
        raise RuntimeError('Antigravity CLI requires a Gemini API key')
    args.extend(('--settings', '{"modelProvider":"gemini"}'))
    env = {name: value for name, value in os.environ.items()
           if name in {'PATH', 'HOME', 'LANG', 'TERM', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR'} or name.startswith('LC_')}
    env['GEMINI_API_KEY'] = key
    state = message.get('previousProviderState') or {}
    if images or state.get('transport') == 'acp':
        return await run_antigravity_acp(
            prompt, images, cwd, previous_id, workspace_id, model, effort, schema, env,
            key, checkpoint, progress, settings,
        )
    if await is_canceled(checkpoint):
        raise AgentCanceled('Antigravity CLI canceled')
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, env=env, start_new_session=True, limit=16 * 1024 * 1024,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(read_diagnostic(proc.stderr))
    try:
        request = {'event': 'user', 'message': {'content': prompt}}
        proc.stdin.write((json.dumps(request, ensure_ascii=False) + '\n').encode())
        with suppress(BrokenPipeError, ConnectionResetError):
            await proc.stdin.drain()
        parts = {}
        step_usage = {}
        result = {}
        conversation_id = ''
        deadline = time.monotonic() + settings.agy_turn_timeout_seconds + 5
        next_check = time.monotonic() + 2.5
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError('Antigravity CLI turn timed out')
            if now >= next_check:
                if await is_canceled(checkpoint):
                    raise AgentCanceled('Antigravity CLI canceled')
                next_check = time.monotonic() + 2.5
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=min(deadline, next_check) - now)
            except TimeoutError:
                continue
            if not line:
                break
            event = json.loads(line)
            kind = event.get('event')
            if kind == 'init':
                conversation_id = event.get('conversation_id')
                if not conversation_id or (previous_id and conversation_id != previous_id):
                    raise RuntimeError('Antigravity CLI did not open the requested conversation')
                state = {'provider': ANTIGRAVITY_CLI, 'conversationId': conversation_id, 'workspaceId': workspace_id}
                if await is_canceled(checkpoint, state):
                    raise AgentCanceled('Antigravity CLI canceled')
            elif kind == 'step_update':
                step = event.get('step_update') or {}
                index = step.get('step_index')
                if type(index) is int and 0 <= index < MAX_PARTS and isinstance(step.get('usage'), dict):
                    step_usage[index] = normalized_usage(step['usage'])
                consume_step(parts, progress, step, key)
            elif kind == 'result':
                result = event.get('result') or {}
                proc.stdin.close()
        code = await asyncio.wait_for(proc.wait(), timeout=5)
        if not result:
            diagnostic = (await stderr_task).replace(key, '[REDACTED]')
            raise RuntimeError(f'Antigravity CLI exited with code {code}: {diagnostic}')
        if result.get('conversation_id') != conversation_id:
            raise RuntimeError('Antigravity CLI result has an unexpected conversation ID')
        status = str(result.get('status') or 'ERROR')
        reply = str(result.get('response') or '').strip().replace(key, '[REDACTED]')
        if len(reply.encode()) > MAX_REPLY_BYTES:
            raise RuntimeError('Antigravity CLI reply exceeds 2 MiB')
        error = ''
        if code or status != 'SUCCESS' or not reply:
            fallback = f'Antigravity CLI ended with {status} (exit {code})'
            if not reply and result.get('denied_actions'):
                actions = [str(item.get('action')) for item in result['denied_actions']
                           if isinstance(item, dict) and item.get('action')]
                fallback = 'Antigravity CLI could not get tool approval in headless mode'
                if actions:
                    fallback += ' (' + ', '.join(dict.fromkeys(actions)) + ')'
            elif not reply:
                fallback += ': empty response'
            error = str(result.get('error') or fallback).replace(key, '[REDACTED]')[:1800]
        if reply and not any(part['type'] == 'text' for part in parts.values()):
            parts['answer'] = {'type': 'text', 'text': reply}
        # The result envelope includes earlier turns when using --conversation.
        usage = {} if previous_id else normalized_usage(result.get('usage') or {})
        if previous_id:
            for counters in step_usage.values():
                for field, count in counters.items():
                    usage[field] = usage.get(field, 0) + count
        steps = sum(part['type'] == 'server_tool' for part in parts.values())
        if steps:
            usage['agentStepCount'] = steps
        return {
            'reply': reply, 'error': error, 'finishReason': 'COMPLETED' if not error else 'ERROR',
            'protocol': ANTIGRAVITY_CLI, 'parts': list(parts.values()), 'usage': usage,
            'response': snapshot(ANTIGRAVITY_CLI, [], {'model': model}, secrets=(key,)),
        }
    finally:
        proc.stdin.close()
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)


class AntigravityAcpSession(CommandCodeSession):
    provider = ANTIGRAVITY_CLI
    label = 'Antigravity CLI'
    approval_error = 'Antigravity CLI could not get tool approval in headless mode'

    def can_use_tool(self, params):
        return False


async def run_antigravity_acp(
    prompt, images, cwd, previous_id, workspace_id, model, effort, schema, env,
    key, checkpoint, progress, settings,
):
    if await is_canceled(checkpoint):
        raise AgentCanceled('Antigravity CLI canceled')
    if isinstance(schema, dict):
        raise RuntimeError('Antigravity CLI ACP cannot enforce a JSON output schema')
    process = await asyncio.create_subprocess_exec(
        settings.agy_bin, 'acp', '--sandbox', cwd=cwd, env=env, start_new_session=True,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        limit=8 * 1024 * 1024,
    )
    session = AntigravityAcpSession(process, progress, key, cwd)
    try:
        async with asyncio.timeout(30):
            info = await session.request('initialize', {'protocolVersion': 1, 'clientCapabilities': {}})
            capabilities = info.get('agentCapabilities') or {}
            if info.get('protocolVersion') != 1 or (images and not capabilities.get('promptCapabilities', {}).get('image')):
                raise RuntimeError('Antigravity CLI ACP does not support image input')
            await session.request('authenticate', {'methodId': 'gemini-api-key'})
            params = {'cwd': cwd, 'mcpServers': []}
            if previous_id:
                resume_params = {**params, 'sessionId': previous_id}
                try:
                    opened = await session.request('session/resume', resume_params)
                except RuntimeError as error:
                    if 'load it with session/load' not in str(error):
                        raise
                    opened = await session.request('session/load', resume_params)
                session.session_id = previous_id
                session.resumed = True
            else:
                opened = await session.request('session/new', params)
                session.session_id = opened.get('sessionId') or ''
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session.session_id):
                raise RuntimeError('Antigravity CLI returned an invalid ACP session ID')
            config = opened.get('configOptions') or []
            if model:
                available = (opened.get('models') or {}).get('availableModels') or []
                if any(entry.get('modelId') == model for entry in available):
                    await session.request('session/set_model', {'sessionId': session.session_id, 'modelId': model})
                else:
                    updated = await session.request('session/set_config_option', {
                        'sessionId': session.session_id, 'configId': 'model', 'value': model,
                    })
                    config = updated.get('configOptions') or config
            levels = ('low', 'medium', 'high', 'xhigh', 'max')
            if effort in levels:
                options = next((entry.get('options') or [] for entry in config
                                if entry.get('id') == 'thought_level'), [])
                offered = {entry.get('value') for entry in options}
                if offered and effort not in offered:
                    effort = min((level for level in levels if level in offered),
                                 key=lambda level: abs(levels.index(level) - levels.index(effort)),
                                 default=effort)
                await session.request('session/set_config_option', {
                    'sessionId': session.session_id, 'configId': 'thought_level', 'value': effort,
                })
        state = {'provider': ANTIGRAVITY_CLI, 'conversationId': session.session_id,
                 'workspaceId': workspace_id, 'transport': 'acp'}
        if await is_canceled(checkpoint, state):
            raise AgentCanceled('Antigravity CLI canceled')
        content = ([{'type': 'text', 'text': prompt}] if prompt else []) + acp_image_blocks(images)
        return await run_acp_turn(session, content, checkpoint, settings.agy_turn_timeout_seconds)
    finally:
        await stop_acp_process(process)


def normalized_usage(raw):
    return {dst: raw[src] for src, dst in USAGE_FIELDS.items()
            if type(raw.get(src)) is int and raw[src] >= 0}


async def is_canceled(checkpoint, state=None):
    return checkpoint is not None and (await checkpoint(state)).get('canceled')


async def read_diagnostic(stream):
    diagnostic = ''
    async for line in stream:
        if not diagnostic:
            diagnostic = line.decode(errors='replace').strip()[:700]
    return diagnostic


def consume_step(parts, progress, step, key):
    index = step.get('step_index')
    if type(index) is not int or not 0 <= index < MAX_PARTS:
        return
    step_type = step.get('step_type')
    status = {'DONE': 'completed', 'ERROR': 'failed'}.get(step.get('state'), 'running')
    current = parts.get(index, {})
    if step_type == 'agent_response':
        text = current.get('text', '') + str(step.get('text_delta') or '')
        if len(text.encode()) > MAX_REPLY_BYTES:
            raise RuntimeError('Antigravity CLI reply exceeds 2 MiB')
        if not text:
            return
        # Keep fragments together until the full text can be redacted.
        part = {'type': 'text', 'text': text, 'status': status, 'providerType': step_type}
        parts[index] = part
    elif step_type == 'tool':
        info = step.get('tool_info') or {}
        name = str(step.get('tool_name') or info.get('name') or current.get('name') or 'tool')[:120]
        part = tool_part(name, str(index), step_type, status='failed' if info.get('error') else status,
                         input=info.get('parameters', current.get('input')),
                         output=info.get('output') or info.get('error') or current.get('output'))
        part = bounded_part(sanitize(part, secrets=(key,)))
        parts[index] = part
    else:
        return
    if progress is not None:
        progress.publish(f'step:{index}', sanitize(part, secrets=(key,)))
