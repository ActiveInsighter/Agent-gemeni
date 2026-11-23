from pathlib import Path
import sys
import tempfile
import unittest

from app.command_code import generate_command_code
from app.config import ServerSettings
from gemini_agent import AgentCanceled


ACP_FIXTURE = r"""
import json
import os
from pathlib import Path
import sys

def send(**payload):
    print(json.dumps({'jsonrpc': '2.0', **payload}), flush=True)

def read(method):
    message = json.loads(sys.stdin.readline())
    assert message['method'] == method, message
    return message

def update(**data):
    send(method='session/update', params={'sessionId': 'fixture-1', 'update': data})

assert sys.argv[1:] == ['acp']
assert os.environ['COMMAND_CODE_API_KEY'] == 'cmd-fixture-key'
assert 'GEMINI_PB_PASSWORD' not in os.environ
req = read('initialize')
send(id=req['id'], result={'protocolVersion': 1, 'agentCapabilities': {'sessionCapabilities': {'resume': {}}}})
req = json.loads(sys.stdin.readline())
assert req['params']['cwd'] == str(Path.cwd())
if req['method'] == 'session/resume':
    assert req['params']['sessionId'] == 'fixture-1'
    assert Path('example.txt').read_text() == 'original'
    send(id=req['id'], result={})
else:
    assert req['method'] == 'session/new'
    Path('example.txt').write_text('original')
    send(id=req['id'], result={'sessionId': 'fixture-1'})

prompt = read('session/prompt')
assert ('Previous conversation:' in prompt['params']['prompt'][0]['text']) == (req['method'] == 'session/new')
if 'cancel me' in prompt['params']['prompt'][0]['text']:
    read('session/cancel')
    send(id=prompt['id'], result={'stopReason': 'cancelled'})
else:
    for req_id, path, expected in [(100, 'example.txt', 'allow'), (101, '../other.txt', 'deny')]:
        send(id=req_id, method='session/request_permission', params={
            'sessionId': 'fixture-1',
            'toolCall': {'kind': 'edit', 'rawInput': {'file_path': path}, 'locations': [{'path': path}]},
            'options': [{'kind': 'allow_once', 'optionId': 'allow'}, {'kind': 'reject_once', 'optionId': 'deny'}],
        })
        assert json.loads(sys.stdin.readline())['result']['outcome']['optionId'] == expected
    for req_id, cwd, expected in [(102, str(Path.cwd()), 'allow'), (103, '..', 'deny')]:
        send(id=req_id, method='session/request_permission', params={
            'sessionId': 'fixture-1', 'toolCall': {'kind': 'execute', 'rawInput': {'command': 'git status', 'cwd': cwd}},
            'options': [{'kind': 'allow_once', 'optionId': 'allow'}, {'kind': 'reject_once', 'optionId': 'deny'}],
        })
        assert json.loads(sys.stdin.readline())['result']['outcome']['optionId'] == expected
    update(sessionUpdate='tool_call', toolCallId='edit-1', kind='edit', title='Write: example.txt',
           rawInput={'file_path': 'example.txt'},
           content=[{'type': 'diff', 'path': 'example.txt', 'oldText': 'original', 'newText': 'changed'}])
    update(sessionUpdate='tool_call_update', toolCallId='edit-1', status='in_progress', title='Writing file')
    update(sessionUpdate='tool_call_update', toolCallId='edit-1', status='completed',
           content=[{'type': 'content', 'content': {'type': 'text', 'text': 'saved'}}])
    for text in ('answer ', 'cmd-fixture-', 'key'):
        update(sessionUpdate='agent_message_chunk', content={'type': 'text', 'text': text})
    send(id=prompt['id'], result={'stopReason': 'end_turn', 'usage': {'totalTokens': 88},
                                '_meta': {'usage': {'inputTokens': 4, 'outputTokens': 3, 'cacheReadTokens': 2}}})
"""


class CommandCodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        launcher = root / 'cmd'
        launcher.write_text(f'#!{sys.executable}\n' + ACP_FIXTURE)
        launcher.chmod(0o700)
        self.settings = ServerSettings(
            pb_url='http://127.0.0.1:8091', command_code_bin=str(launcher),
            command_code_cwd=str(root / 'workspaces'), command_code_api_key='cmd-fixture-key',
            command_code_shell_guard=str(root / 'fixture-guard.cjs'),
        )
        self.first = {'id': 'a' * 15}

    async def test_turn_stream_and_native_resume(self):
        states = []

        async def checkpoint(state=None):
            if state:
                states.append(state)
            return {'canceled': False}

        job = {'prompt': 'question', 'history': [{'role': 'user', 'text': 'prior question'}]}
        options = dict(checkpoint=checkpoint, progress=None, settings=self.settings)
        first = await generate_command_code(job, self.first, **options)
        self.assertEqual(first['reply'], 'answer [REDACTED]')
        self.assertEqual(first['usage']['totalTokenCount'], 7)
        self.assertEqual(first['usage']['cachedContentTokenCount'], 2)
        tool = first['parts'][0]
        self.assertEqual((tool['providerType'], tool['status']), ('write_file', 'completed'))
        self.assertIn('\n-original\n+changed', tool['output']['diff'])
        self.assertIsInstance(tool['output']['durationMs'], int)
        self.assertEqual(states[0]['workspaceId'], self.first['id'])

        second = await generate_command_code(
            job, {'id': 'b' * 15, 'previousProviderState': states[0]}, **options,
        )
        self.assertEqual(second['reply'], first['reply'])
        self.assertEqual(states[1], states[0])

    async def test_checkpoint_cancellation(self):
        has_session = False

        async def checkpoint(state=None):
            nonlocal has_session
            if state:
                has_session = True
                return {'canceled': False}
            return {'canceled': has_session}

        with self.assertRaises(AgentCanceled):
            await generate_command_code(
                {'prompt': 'cancel me', 'history': [{'role': 'user', 'text': 'prior'}]},
                self.first, checkpoint=checkpoint, progress=None, settings=self.settings,
            )
