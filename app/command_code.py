"""Run Command Code through its native ACP stdio transport."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from difflib import unified_diff
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time

from agent_timeline import MAX_PARTS, bounded_part, tool_part
from app.native_sessions import COMMAND_CODE, native_prompt, native_session
from gemini_agent import AgentCanceled, MAX_REPLY_BYTES, extract_image_urls
from response_capture import sanitize


class CommandCodeSession:
    def __init__(self, process, progress, secret, cwd, *, shell_enabled=False):
        self.process = process
        self.progress = progress
        self.secrets = (secret,) if secret else ()
        self.cwd = Path(cwd)
        self.shell_enabled = shell_enabled
        self.session_id = ''
        self.resumed = False
        self.request_id = 0
        self.parts = {}
        self.tools = {}
        self.denied = False
        self.reply_bytes = 0

    async def send(self, **message):
        self.process.stdin.write((json.dumps({'jsonrpc': '2.0', **message}) + '\n').encode())
        await self.process.stdin.drain()

    async def request(self, method, params):
        self.request_id += 1
        request_id = self.request_id
        await self.send(id=request_id, method=method, params=params)
        while line := await self.process.stdout.readline():
            message = json.loads(line)
            inbound = message.get('method')
            if inbound:
                params = message.get('params') or {}
                if 'id' in message:
                    await self.answer(inbound, message['id'], params)
                elif inbound == 'session/update' and method == 'session/prompt':
                    if params.get('sessionId') == self.session_id:
                        self.update(params.get('update') or {})
                continue
            if message.get('id') == request_id:
                if 'error' in message:
                    error = message['error'].get('message') or 'unknown error'
                    raise RuntimeError(f'Command Code {method}: {sanitize(error, secrets=self.secrets, text_limit=700)}')
                return message['result']
        raise RuntimeError(f'Command Code closed ACP while handling {method}')

    async def answer(self, method, request_id, params):
        if method == 'session/request_permission' and params.get('sessionId') == self.session_id:
            allowed = self.can_use_tool(params)
            kind = 'allow_once' if allowed else 'reject_once'
            option = next((item.get('optionId') for item in params.get('options', [])
                           if item.get('kind') == kind), None)
            self.denied |= not allowed or not option
            outcome = {'outcome': 'selected', 'optionId': option} if option else {'outcome': 'cancelled'}
            await self.send(id=request_id, result={'outcome': outcome})
        else:
            await self.send(id=request_id, error={'code': -32601, 'message': 'Client method unavailable'})

    def can_use_tool(self, params):
        call = params.get('toolCall') or {}
        raw = call.get('rawInput')
        if (self.shell_enabled and call.get('kind') == 'execute' and isinstance(raw, dict)
                and isinstance(raw.get('command'), str) and isinstance(raw.get('cwd'), str)):
            return self.in_workspace(raw['cwd'], include_root=True)
        return self.can_edit(params)

    def in_workspace(self, target, *, include_root=False):
        try:
            path = (self.cwd / target).resolve()
        except (OSError, ValueError, RuntimeError):
            return False
        return path.is_relative_to(self.cwd) and (include_root or path != self.cwd)

    def can_edit(self, params):
        # Approve only the explicit target of a file edit within the job workspace.
        call = params.get('toolCall') or {}
        if call.get('kind') != 'edit' or (params.get('_meta') or {}).get('risk'):
            return False
        raw = call.get('rawInput')
        if not isinstance(raw, dict):
            return False
        target = raw.get('file_path') or raw.get('path')
        locations = call.get('locations') or []
        if not isinstance(target, str) or len(locations) != 1 or locations[0].get('path') != target:
            return False
        return self.in_workspace(target)

    def update(self, update):
        kind = update.get('sessionUpdate')
        if kind in ('agent_message_chunk', 'agent_thought_chunk'):
            content = update.get('content') or {}
            field = 'text' if kind == 'agent_message_chunk' else 'reasoning'
            delta = content.get('text')
            if content.get('type') != 'text' or not isinstance(delta, str) or not delta:
                return
            if field == 'text':
                self.reply_bytes += len(delta.encode())
                if self.reply_bytes > MAX_REPLY_BYTES:
                    raise RuntimeError('Command Code reply exceeds 2 MiB')
            last_key = next(reversed(self.parts), None)
            if last_key is not None and self.parts[last_key]['type'] == field:
                key = last_key
                text = self.parts[key][field] + delta
            elif len(self.parts) < MAX_PARTS:
                key = f'step:{len(self.parts)}'
                text = delta
            else:
                return
            if field == 'reasoning':
                text = text[:65536]
            part = {'type': field, field: text}
        elif kind in ('tool_call', 'tool_call_update'):
            call_id = update.get('toolCallId')
            if not isinstance(call_id, str) or len(call_id) > 160:
                return
            if call_id not in self.tools:
                if len(self.parts) >= MAX_PARTS:
                    return
                self.tools[call_id] = (f'step:{len(self.parts)}', {})
            key, raw = self.tools[call_id]
            if update.get('title'):
                raw.setdefault('initialTitle', update['title'])
            if raw.get('status') != 'in_progress' and update.get('status') == 'in_progress':
                raw.setdefault('started', time.monotonic())
            if update.get('status') in ('completed', 'failed') and 'started' in raw:
                raw.setdefault('durationMs', round((time.monotonic() - raw['started']) * 1000))
            if update.get('content'):
                diffs = [item for item in update['content'] if item.get('type') == 'diff']
                if diffs:
                    raw['diffs'] = diffs
            raw.update({name: value for name, value in update.items() if value is not None})
            part = normalized_tool(raw, call_id, self.secrets)
        else:
            return
        self.parts[key] = part
        if self.progress is not None:
            self.progress.publish(key, sanitize(part, secrets=self.secrets))

    def result(self, final):
        stop = final.get('stopReason')
        if stop == 'cancelled':
            raise AgentCanceled('Command Code canceled')
        parts = list(self.parts.values())
        reply = '\n\n'.join(part['text'] for part in parts if part['type'] == 'text').strip()
        reply = sanitize(reply, secrets=self.secrets, text_limit=MAX_REPLY_BYTES, parse_strings=False)
        error = ''
        if not reply and self.denied:
            error = 'Command Code could not approve a tool outside its conversation workspace'
        elif stop not in ('end_turn', 'max_tokens', 'max_turn_requests') or not reply:
            error = f'Command Code ended with {stop or "an unknown reason"}' + (': empty response' if not reply else '')
        usage_fields = {'inputTokens': 'promptTokenCount', 'outputTokens': 'candidatesTokenCount',
                        'totalTokens': 'totalTokenCount', 'cachedReadTokens': 'cachedContentTokenCount'}
        raw_usage = (final.get('_meta') or {}).get('usage') or {}
        if not raw_usage and not self.resumed:
            raw_usage = final.get('usage') or {}
        if 'cacheReadTokens' in raw_usage and 'cachedReadTokens' not in raw_usage:
            raw_usage = {**raw_usage, 'cachedReadTokens': raw_usage['cacheReadTokens']}
        if 'totalTokens' not in raw_usage and all(type(raw_usage.get(key)) is int for key in ('inputTokens', 'outputTokens')):
            raw_usage = {**raw_usage, 'totalTokens': raw_usage['inputTokens'] + raw_usage['outputTokens']}
        usage = {dest: raw_usage[src] for src, dest in usage_fields.items()
                 if type(raw_usage.get(src)) is int and raw_usage[src] >= 0}
        if self.tools:
            usage['agentStepCount'] = len(self.tools)
        return {
            'reply': reply, 'error': error, 'finishReason': 'ERROR' if error else stop.upper(),
            'parts': parts,
            'usage': usage, 'protocol': COMMAND_CODE,
        }


def normalized_tool(raw, call_id, secrets):
    kind = raw.get('kind') or 'other'
    name = raw.get('name') or (raw.get('_meta') or {}).get('toolName')
    title = str(raw.get('title') or '')
    if not name:
        label = str(raw.get('initialTitle') or title).split(':', 1)[0]
        name = {
            ('edit', 'Write'): 'write_file', ('read', 'List'): 'list_dir',
            ('search', 'Glob'): 'glob', ('search', 'Search'): 'grep', ('think', 'Updating todos'): 'todo_write',
        }.get((kind, label)) or {
            'read': 'read_file', 'edit': 'edit_file', 'delete': 'delete_file',
            'move': 'edit_file', 'search': 'grep_search', 'execute': 'run_command', 'fetch': 'url_context',
        }.get(kind, label or 'command_code_tool')
    status = {'completed': 'completed', 'failed': 'failed'}.get(raw.get('status'), 'running')
    input_value = raw.get('rawInput')
    if isinstance(input_value, dict):
        input_value = dict(input_value)
    elif input_value is None:
        input_value = {}
    else:
        input_value = {'command' if kind == 'execute' else 'input': input_value}
    locations = raw.get('locations') or []
    if locations and isinstance(locations[0], dict) and locations[0].get('path'):
        input_value.setdefault('path', locations[0]['path'])
    if raw.get('title'):
        input_value['title'] = raw['title']
    output = raw.get('rawOutput')
    diffs = []
    texts = []
    for item in raw.get('diffs') or []:
        path = str(item.get('path') or 'file').lstrip('/')
        old = str(item.get('oldText') or '').splitlines()
        new = str(item.get('newText') or '').splitlines()
        diffs.extend(unified_diff(old, new, f'a/{path}', f'b/{path}', lineterm=''))
    for item in raw.get('content') or []:
        if item.get('type') == 'content' and (item.get('content') or {}).get('type') == 'text':
            texts.append(item['content'].get('text', ''))
    if diffs or raw.get('durationMs') is not None:
        if not isinstance(output, dict):
            output = {'output': output} if output is not None else {}
        if diffs:
            output['diff'] = '\n'.join(diffs)
        if raw.get('durationMs') is not None:
            output['durationMs'] = raw['durationMs']
    if texts and (output is None or isinstance(output, dict)):
        output = {**(output or {}), 'output': '\n'.join(texts)}
    name = str(name)[:120]
    part = tool_part(name, call_id, name, status=status, input=input_value, output=output)
    return bounded_part(sanitize(part, secrets=secrets))


async def watch_cancellation(checkpoint):
    while True:
        await asyncio.sleep(2.5)
        if (await checkpoint()).get('canceled'):
            return


@asynccontextmanager
async def workspace_lock(cwd, checkpoint):
    workspace = Path(cwd)
    with workspace.with_name(f'.{workspace.name}.lock').open('a') as lock:
        try:
            async with asyncio.timeout(30):
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if checkpoint is not None and (await checkpoint()).get('canceled'):
                            raise AgentCanceled('Command Code canceled')
                        await asyncio.sleep(1)
        except TimeoutError:
            raise RuntimeError('The previous Command Code turn is still using this workspace') from None
        yield


async def generate_command_code(job, message, *, checkpoint, progress, settings):
    if extract_image_urls(job, message):
        raise RuntimeError('Command Code currently supports text and Markdown input')
    previous_id, workspace_id, cwd = native_session(
        message, COMMAND_CODE, settings.command_code_cwd, 'conversationId',
    )
    prompt = native_prompt(job, message, include_history=not previous_id)
    if not prompt:
        raise RuntimeError('Command Code prompt is empty')
    instructions = str(job.get('systemPrompt') or '').strip()
    if instructions:
        prompt = f'System instructions:\n{instructions}\n\nUser request:\n{prompt}'
    if checkpoint is not None and (await checkpoint()).get('canceled'):
        raise AgentCanceled('Command Code canceled')

    async with workspace_lock(cwd, checkpoint):
        return await run_command_code(prompt, cwd, previous_id, workspace_id, checkpoint, progress, settings)


async def run_command_code(prompt, cwd, previous_id, workspace_id, checkpoint, progress, settings):
    env = {key: value for key, value in os.environ.items()
           if key in {'PATH', 'HOME', 'LANG', 'TERM', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}
           or key.startswith('LC_')}
    secret = settings.command_code_api_key
    guard = settings.command_code_shell_guard
    if guard:
        env['NODE_OPTIONS'] = '--require=' + guard
        env['AW_COMMAND_CODE_WORKSPACE'] = cwd
    if secret:
        env['COMMAND_CODE_API_KEY'] = secret
    process = await asyncio.create_subprocess_exec(
        settings.command_code_bin, 'acp', cwd=cwd, env=env, start_new_session=True,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        limit=8 * 1024 * 1024,
    )
    session = CommandCodeSession(process, progress, secret, cwd, shell_enabled=bool(guard))
    tasks = []
    try:
        async with asyncio.timeout(30):
            info = await session.request('initialize', {'protocolVersion': 1, 'clientCapabilities': {}})
            if info.get('protocolVersion') != 1:
                raise RuntimeError('Command Code returned an unsupported ACP version')
            params = {'cwd': cwd, 'mcpServers': []}
            if previous_id:
                capabilities = info.get('agentCapabilities') or {}
                supports_resume = 'resume' in (capabilities.get('sessionCapabilities') or {})
                method = 'session/resume' if supports_resume else 'session/load'
                opened = await session.request(method, {**params, 'sessionId': previous_id})
                session.session_id = previous_id
                session.resumed = True
            else:
                opened = await session.request('session/new', params)
                session.session_id = opened.get('sessionId') or ''
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session.session_id):
                raise RuntimeError('Command Code returned an invalid ACP session ID')
            session_settings = {'provider': COMMAND_CODE, 'conversationId': session.session_id, 'workspaceId': workspace_id}
        if checkpoint is not None and (await checkpoint(session_settings)).get('canceled'):
            raise AgentCanceled('Command Code canceled')

        model = (opened.get('models') or {}).get('currentModelId') or next(
            (option.get('currentValue') for option in opened.get('configOptions', [])
             if option.get('category') == 'model'), None,
        )
        if model and prompt == '/model':
            return {'reply': f'Command Code · `{model}`', 'error': '', 'finishReason': 'END_TURN',
                    'protocol': COMMAND_CODE}
        if model and not prompt.startswith('/'):
            prompt = f'Host runtime: Command Code. The current configured model is {model}.\n\n{prompt}'
        params = {'sessionId': session.session_id, 'prompt': [{'type': 'text', 'text': prompt}]}
        turn = asyncio.create_task(session.request('session/prompt', params))
        tasks.append(turn)
        cancellation = None
        if checkpoint is not None:
            cancellation = asyncio.create_task(watch_cancellation(checkpoint))
            tasks.append(cancellation)
        finished, _ = await asyncio.wait(tasks, timeout=settings.command_code_turn_timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
        if cancellation in finished or not finished:
            if cancellation in finished:
                cancellation.result()
            with suppress(BrokenPipeError, ConnectionResetError, RuntimeError, TimeoutError):
                async with asyncio.timeout(2):
                    await session.send(method='session/cancel', params={'sessionId': session.session_id})
                    await asyncio.shield(turn)
            if cancellation in finished:
                raise AgentCanceled('Command Code canceled')
            raise RuntimeError('Command Code turn timed out')
        return session.result(turn.result())
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        process.stdin.close()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
