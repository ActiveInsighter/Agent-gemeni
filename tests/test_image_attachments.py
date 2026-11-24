import base64
import json
import unittest
from unittest.mock import AsyncMock

import httpx

from app.codex_app_server import CodexAppServerClient
from gemini_agent import build_agent_payload, generate_agent
from gemini_text import generate
from image_attachments import extract_image_urls, resolve_images_to_base64
from key_pool import KeyPool

URL = 'https://file.any1.tech/api/library/image-sources/00000000-0000-4000-8000-000000000001.png'
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII='


class ImageAttachmentsTests(unittest.IsolatedAsyncioTestCase):
    def test_extracts_structured_and_legacy_refs_once(self):
        marker = f'<!-- aw:images:{json.dumps([URL])} -->'
        job = {'options': {'images': [URL]}, 'prompt': 'Question\n' + marker}
        self.assertEqual(extract_image_urls(job, {}), [URL])
        self.assertEqual(extract_image_urls({}, {'inputText': marker}), [URL])
        for unsafe in (URL.replace('file.any1.tech', '127.0.0.1'), URL + '/..', 'file:///tmp/image.png'):
            with self.assertRaisesRegex(RuntimeError, '资料库'):
                extract_image_urls({'options': {'images': [unsafe]}}, {})

    async def test_rejects_redirect_and_mismatched_content_type(self):
        responses = [
            httpx.Response(302, headers={'Location': 'http://127.0.0.1/'}),
            httpx.Response(200, headers={'Content-Type': 'text/html'}, text='error'),
        ]
        for response in responses:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: response), follow_redirects=True,
            ) as client:
                with self.assertRaisesRegex(RuntimeError, '图片资料库'):
                    await resolve_images_to_base64(client, [URL])

    async def test_gemini_links_then_inline_on_explicit_media_rejection(self):
        events = []
        job = {'model': 'gemini-4-hydrogen', 'prompt': '',
               'options': {'images': [URL]}, 'history': [{'role': 'user', 'text': '', 'images': [URL]}]}

        async def outbound(request):
            if request.method == 'GET':
                self.assertEqual(str(request.url), URL)
                self.assertNotIn('x-goog-api-key', request.headers)
                events.append('download')
                return httpx.Response(200, headers={'Content-Type': 'image/png'}, content=base64.b64decode(PNG))
            events.append('model')
            contents = json.loads(request.content)['contents']
            if events == ['model']:
                for turn in contents:
                    self.assertEqual(turn['parts'], [{'fileData': {'mimeType': 'image/png', 'fileUri': URL}}])
                return httpx.Response(404, json={'error': {
                    'code': 404, 'status': 'NOT_FOUND',
                    'message': 'Cannot fetch content from the provided URL. The URL could not be found.',
                }})
            for turn in contents:
                self.assertEqual(turn['parts'], [{'inlineData': {'mimeType': 'image/png', 'data': PNG}}])
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                                                             'content': {'parts': [{'text': 'image seen'}]}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(outbound)) as client:
            pool = KeyPool(['key-one'], rpm_per_key=60, rpd_per_key=100)
            result = await generate(client, pool, job, {})
        self.assertEqual(result['reply'], 'image seen')
        self.assertEqual(events, ['model', 'download', 'model'])

    async def test_managed_agent_retries_rejected_image_url_before_checkpointing(self):
        requests = []
        state = {}
        url_part = {'type': 'image', 'mime_type': 'image/png', 'uri': URL}
        inline_part = {'type': 'image', 'mime_type': 'image/png', 'data': PNG}

        async def checkpoint(update=None):
            if update:
                state.update(update)
            return {'state': state, 'canceled': False}

        async def outbound(request):
            if str(request.url) == URL:
                requests.append('download')
                return httpx.Response(200, headers={'Content-Type': 'image/png'}, content=base64.b64decode(PNG))
            if request.method == 'GET':
                return httpx.Response(200, json={'id': 'image-agent', 'status': 'completed', 'output_text': 'image seen'})
            self.assertFalse(state)
            parts = json.loads(request.content)['input']
            if not requests:
                self.assertEqual(parts, [url_part])
                requests.append('url')
                return httpx.Response(403, json={'error': {'status': 'PERMISSION_DENIED'}})
            self.assertEqual(parts, [inline_part])
            requests.append('inline')
            return httpx.Response(200, json={'id': 'image-agent', 'status': 'completed', 'output_text': 'image seen'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(outbound)) as client:
            pool = KeyPool(['fixture-key'], rpm_per_key=1e6, rpd_per_key=100)
            result = await generate_agent(client, pool, {'model': 'antigravity', 'options': {'images': [URL]}},
                                          {}, checkpoint=checkpoint)
        self.assertEqual(result['reply'], 'image seen')
        self.assertEqual(requests, ['url', 'download', 'inline'])
        self.assertEqual(state['id'], 'image-agent')

    async def test_managed_and_codex_transport_use_native_image_blocks(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, headers={'Content-Type': 'image/png'}, content=base64.b64decode(PNG),
        ))) as client:
            images = await resolve_images_to_base64(client, [URL])
        managed = build_agent_payload({'model': 'Antigravity'}, {}, resolved_images=images)
        self.assertEqual(managed['input'], [{'type': 'image', 'mime_type': 'image/png', 'data': PNG}])
        codex = CodexAppServerClient()
        codex.request = AsyncMock(return_value={'turnId': 'turn'})
        await codex.turn_start('thread', '', images=tuple(images))
        self.assertEqual(codex.request.call_args.args, (
            'turn/start', {'threadId': 'thread', 'input': [{'type': 'image', 'url': f'data:image/png;base64,{PNG}'}]},
        ))
