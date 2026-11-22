import unittest

from agent_progress import MessageProgress
from agent_timeline import prepare_result
from app.codex_timeline import CodexTimeline
from response_capture import extract_structured_payload


class AgentTimelineTests(unittest.TestCase):
    def test_codex_interleaved_items_keep_order_and_public_data(self):
        progress = MessageProgress(secrets=['test-secret-value'])
        adapter = CodexTimeline(progress, 2 * 1024 * 1024)
        adapter.consume('item/reasoning/summaryTextDelta', {'itemId': 'r', 'summaryIndex': 0, 'delta': 'Inspect'})
        adapter.consume('item/reasoning/textDelta', {'itemId': 'r', 'delta': 'PRIVATE_RATIONALE'})
        adapter.update({'id': 'note', 'type': 'agentMessage', 'phase': 'commentary', 'text': 'I will check'}, completed=True)
        adapter.update({'id': 'cmd', 'type': 'commandExecution', 'command': 'printf ok', 'cwd': '/workspace'})
        adapter.consume('item/commandExecution/outputDelta', {'itemId': 'cmd', 'delta': 'test-secret-value ok'})
        live = progress.timeline_rows()
        self.assertIn('[REDACTED] ok', live[2]['part']['output']['aggregatedOutput'])
        adapter.update({'id': 'cmd', 'type': 'commandExecution', 'exitCode': 0, 'durationMs': 8}, completed=True)
        adapter.update({'id': 'diff', 'type': 'fileChange', 'changes': [{'path': 'readme.txt', 'diff': '+test'}]}, completed=True)
        adapter.update({'id': 'answer', 'type': 'agentMessage', 'phase': 'final_answer', 'text': '{ "done": true }'}, completed=True)
        result = prepare_result({'parts': adapter.parts(), 'reply': adapter.answer()}, 'codex-app-server', progress)
        parts = result['timeline']['parts']
        self.assertEqual([p['type'] for p in parts], ['reasoning', 'text', 'server_tool', 'server_tool', 'text'])
        self.assertEqual(parts[0]['reasoning'], 'Inspect')
        self.assertEqual(parts[2]['output']['exitCode'], 0)
        self.assertEqual(parts[3]['output']['changes'][0]['diff'], '+test')
        self.assertEqual(parts[4]['text'], '{ "done": true }')
        self.assertEqual(result['structuredOutput'], {'done': True})

    def test_interactions_keeps_tool_pair_and_block_local_reference(self):
        data = {'steps': [
            {'type': 'google_search_call', 'id': 'search', 'queries': ['query']},
            {'type': 'google_search_result', 'call_id': 'search', 'results': [{'title': 'Docs'}]},
            {'type': 'model_output', 'content': [{'type': 'text', 'text': 'Link',
              'annotations': [{'url': 'https://example.com', 'title': 'Docs', 'start_index': 0, 'end_index': 4}]}]},
        ]}
        progress = MessageProgress()
        progress.snapshot(data)
        live = [r['part'] for r in progress.timeline_rows()]
        self.assertEqual(live[0]['output'], {'results': [{'title': 'Docs'}]})
        result = prepare_result({'reply': 'Link', **extract_structured_payload('interactions', data)}, 'interactions', progress)
        self.assertEqual(result['timeline']['parts'][0]['output'], live[0]['output'])
        self.assertEqual(result['timeline']['parts'][1]['annotations'][0]['endIndex'], 4)

    def test_partial_output_and_large_tool_log_have_bounded_fallbacks(self):
        progress = MessageProgress()
        adapter = CodexTimeline(progress, 2 * 1024 * 1024)
        adapter.update({'id': 'cmd', 'type': 'commandExecution', 'command': 'printf ok', 'cwd': '/workspace'})
        adapter.consume('item/commandExecution/outputDelta', {'itemId': 'cmd', 'delta': '\\' * 60000})
        result = prepare_result({'reply': '', 'error': 'timed out'}, 'codex-app-server', progress)
        tool = result['timeline']['parts'][0]
        self.assertEqual(tool['status'], 'canceled')
        self.assertEqual(tool['input']['command'], 'printf ok')
        self.assertTrue(tool['output']['aggregatedOutput'])
        self.assertTrue(tool['truncated'])
