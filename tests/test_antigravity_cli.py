from pathlib import Path
import sys
import tempfile
import unittest

from app.antigravity_cli import generate_antigravity_cli
from app.config import ServerSettings


SCRIPT = r"""
import json, os, pathlib, sys

cwd = pathlib.Path.cwd()
request = json.loads(sys.stdin.readline())
previous = '--conversation' in sys.argv
assert request['event'] == 'user'
contents = request['message']['content']
assert ('Previous conversation:' in contents) is not previous
assert '--sandbox' in sys.argv and '--disable-slash-commands' in sys.argv
assert 'GEMINI_API_KEY' in os.environ and 'GEMINI_API_KEYS' not in os.environ
state = cwd / 'session.txt'
if previous:
    assert sys.argv[sys.argv.index('--conversation') + 1] == 'session-1'
    assert state.read_text() == 'session-1'
else:
    state.write_text('session-1')

def emit(event):
    print(json.dumps(event), flush=True)

emit({'event': 'init', 'conversation_id': 'session-1'})
if 'blocked' in contents:
    emit({'event': 'result', 'result': {
        'conversation_id': 'session-1', 'response': '', 'status': 'SUCCESS',
        'denied_actions': [{'action': 'command', 'display_name': 'Bash'}],
    }})
    assert sys.stdin.readline() == ''
    sys.exit(0)
emit({'event': 'step_update', 'step_update': {
    'step_index': 1, 'step_type': 'tool', 'state': 'DONE', 'tool_name': 'list_dir',
    'tool_info': {'parameters': {'path': '.'}, 'output': 'session.txt'},
}})
emit({'event': 'step_update', 'step_update': {
    'step_index': 2, 'step_type': 'agent_response', 'state': 'DONE', 'text_delta': 'answer',
    'usage': {'input_tokens': 3, 'output_tokens': 1},
}})
emit({'event': 'result', 'result': {
    'conversation_id': 'session-1', 'response': 'answer', 'status': 'SUCCESS',
    'usage': {'input_tokens': 6 if previous else 3, 'output_tokens': 2 if previous else 1},
}})
assert sys.stdin.readline() == ''
"""


class AntigravityCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_turn_and_resume_share_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / 'agy'
            launcher.write_text(f'#!{sys.executable}\n' + SCRIPT)
            launcher.chmod(0o700)
            settings = ServerSettings(pb_url='http://127.0.0.1:8090', agy_bin=str(launcher),
                                      agy_cwd=str(root / 'workspaces'), agy_api_key='fake-gemini-key')
            job = {'prompt': 'question', 'history': [{'role': 'user', 'text': 'prior question'}]}
            saved = []

            async def checkpoint(state):
                if state:
                    saved.append(state)
                return {'canceled': False}

            first = {'id': 'a' * 15}
            options = {'checkpoint': checkpoint, 'progress': None, 'settings': settings}
            result = await generate_antigravity_cli(job, first, **options)
            self.assertEqual(result['reply'], 'answer')
            self.assertEqual(result['usage']['agentStepCount'], 1)
            self.assertEqual(result['parts'][0]['output'], 'session.txt')
            self.assertEqual(saved[0]['workspaceId'], first['id'])
            result = await generate_antigravity_cli(job, {'id': 'b' * 15, 'previousProviderState': saved[0]}, **options)
            self.assertEqual(result['reply'], 'answer')
            self.assertEqual(result['usage']['promptTokenCount'], 3)
            self.assertEqual(saved[0], saved[1])
            self.assertTrue((root / 'workspaces' / first['id'] / 'session.txt').is_file())
            self.assertFalse((root / 'workspaces' / ('b' * 15)).exists())
            denied = await generate_antigravity_cli(
                {**job, 'prompt': 'blocked'}, {'id': 'c' * 15}, **options,
            )
            self.assertEqual(denied['finishReason'], 'ERROR')
            self.assertIn('tool approval in headless mode (command)', denied['error'])

    async def test_image_turn_uses_acp_and_text_followup_keeps_that_session(self):
        fixture = r"""
import json, os, sys
assert sys.argv[1:] == ['acp', '--sandbox']
assert os.environ['GEMINI_API_KEY'] == 'fake-key'

def emit(**message):
    print(json.dumps({'jsonrpc': '2.0', **message}), flush=True)

resumed = False
for line in sys.stdin:
    req = json.loads(line)
    method, params = req['method'], req['params']
    result = {}
    if method == 'initialize':
        result = {'protocolVersion': 1, 'agentCapabilities': {'promptCapabilities': {'image': True}}}
    elif method == 'authenticate':
        assert params['methodId'] == 'gemini-api-key'
    elif method == 'session/new':
        result = {'sessionId': 'agy-test'}
    elif method == 'session/resume':
        assert params['sessionId'] == 'agy-test'
        emit(id=req['id'], error={'code': -32603, 'message': 'Failed to read session; retry, or load it with session/load'})
        continue
    elif method == 'session/load':
        assert params['sessionId'] == 'agy-test'
        resumed = True
    elif method == 'session/set_config_option':
        assert params['configId'] == 'model' and params['value'] == 'gemini-4-hydrogen'
    elif method == 'session/prompt':
        expected = [{'type': 'text', 'text': 'question'}]
        if not resumed:
            expected.append({'type': 'image', 'mimeType': 'image/png', 'data': 'aW1hZ2U='})
        assert params['prompt'] == expected
        emit(method='session/update', params={'sessionId': 'agy-test', 'update': {
            'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'answer'},
        }})
        result = {'stopReason': 'end_turn'}
    else:
        raise AssertionError(method)
    emit(id=req['id'], result=result)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / 'agy'
            launcher.write_text(f'#!{sys.executable}\n' + fixture)
            launcher.chmod(0o700)
            settings = ServerSettings(pb_url='http://127.0.0.1:8090', agy_bin=str(launcher),
                                      agy_cwd=str(root / 'workspaces'), agy_api_key='fake-key')
            states = []

            async def checkpoint(state=None):
                if state:
                    states.append(state)
                return {'canceled': False}

            options = dict(checkpoint=checkpoint, progress=None, settings=settings)
            job = {'prompt': 'question'}
            first = await generate_antigravity_cli(job, {'id': 'a' * 15},
                                                   images=[('image/png', 'aW1hZ2U=')], **options)
            self.assertEqual(first['reply'], 'answer')
            self.assertEqual(states[0]['transport'], 'acp')
            followup = await generate_antigravity_cli(
                job, {'id': 'b' * 15, 'previousProviderState': states[0]}, **options,
            )
            self.assertEqual(followup['reply'], 'answer')
            self.assertEqual(states[0], states[1])
