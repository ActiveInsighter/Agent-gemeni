import json
import unittest
from datetime import datetime, timedelta, timezone

import httpx

from agent_environments import reclaim

ROOT = "https://generativelanguage.googleapis.com/v1beta/environments"
NOW = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)


def moment(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def transport(environments, messages=None, pages=None):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == 'GET' and request.url.path == '/v1beta/environments':
            if pages is not None:
                token = request.url.params.get('pageToken')
                return httpx.Response(200, json=pages.get(token or '', {}))
            return httpx.Response(200, json={'environments': environments})
        if request.method == 'DELETE' and request.url.path.startswith('/v1beta/environments/'):
            return httpx.Response(204)
        if request.method == 'GET' and '/api/collections/aw_gemini_messages/records' in request.url.path:
            return httpx.Response(200, json={'items': messages or [], 'totalPages': 1})
        raise AssertionError(f'unexpected request {request.method} {request.url}')

    return httpx.MockTransport(handler), calls


def state(environment_id):
    return {'providerState': json.dumps({'environmentId': environment_id}), 'status': 'failed',
            'created': '2026-11-23 14:00:00.000Z'}


class EnvironmentReclaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_keeps_referenced_and_recent_environments(self):
        client_transport, calls = transport([
            {'id': 'env-old', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(72)},
            {'id': 'env-kept', 'status': 'active', 'size_bytes': '200', 'last_accessed': moment(72)},
            {'id': 'env-recent', 'status': 'active', 'size_bytes': '300', 'last_accessed': moment(1)},
        ], [state('env-kept')])
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key'], groups='', root='https://pb.fixture',
                                    token='token', email='', password='', keep_hours=48,
                                    stale_hours=24, apply=False, use_pb=True,
                                    max_bytes=None, max_envs=None, now=NOW)
        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['storage_bytes'], 600)
        self.assertEqual(summary['kept'], 1)
        self.assertEqual(summary['candidates'], 1)
        self.assertEqual(summary['deleted'], 0)
        self.assertFalse(any(call.method == 'DELETE' for call in calls))

    async def test_apply_deletes_only_stale_unreferenced_environment(self):
        client_transport, calls = transport([
            {'id': 'env-old', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(72)},
            {'id': 'env-kept', 'status': 'active', 'size_bytes': '200', 'last_accessed': moment(72)},
            {'id': 'env-recent', 'status': 'active', 'size_bytes': '300', 'last_accessed': moment(1)},
        ], [state('env-kept')])
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key'], groups='', root='https://pb.fixture',
                                    token='token', email='', password='', keep_hours=48,
                                    stale_hours=24, apply=True, use_pb=True,
                                    max_bytes=None, max_envs=None, now=NOW)
        deletes = [call for call in calls if call.method == 'DELETE']
        self.assertEqual([call.url.path for call in deletes], ['/v1beta/environments/env-old'])
        self.assertEqual(deletes[0].headers['x-goog-api-key'], 'key')
        self.assertEqual(summary['deleted'], 1)
        self.assertEqual(summary['reclaimed_bytes'], 100)

    async def test_duplicate_projects_are_counted_once(self):
        client_transport, calls = transport([
            {'id': 'env-shared', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(72)},
        ])
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key-a', 'key-b'], groups='', root='https://pb.fixture',
                                    token='', email='', password='', keep_hours=48,
                                    stale_hours=24, apply=False, use_pb=False,
                                    max_bytes=None, max_envs=None, now=NOW)
        lists = [call for call in calls if call.url.path == '/v1beta/environments']
        self.assertEqual(len(lists), 2)
        self.assertEqual(summary['total'], 1)
        self.assertEqual(summary['storage_bytes'], 100)

    async def test_groups_list_one_key_per_project(self):
        client_transport, calls = transport([
            {'id': 'env-shared', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(72)},
        ])
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key-a', 'key-b'], groups='project\nproject',
                                    root='https://pb.fixture', token='', email='', password='',
                                    keep_hours=48, stale_hours=24, apply=False, use_pb=False,
                                    max_bytes=None, max_envs=None, now=NOW)
        self.assertEqual(len([call for call in calls if call.url.path == '/v1beta/environments']), 1)
        self.assertEqual(summary['total'], 1)

    async def test_budget_limits_the_deletion_count(self):
        client_transport, calls = transport([
            {'id': 'env-1', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(72)},
            {'id': 'env-2', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(71)},
            {'id': 'env-3', 'status': 'active', 'size_bytes': '100', 'last_accessed': moment(70)},
        ])
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key'], groups='', root='https://pb.fixture',
                                    token='', email='', password='', keep_hours=48,
                                    stale_hours=24, apply=True, use_pb=False,
                                    max_bytes=150, max_envs=None, now=NOW)
        self.assertEqual(summary['candidates'], 3)
        self.assertEqual(summary['selected'], 2)
        self.assertEqual(summary['deleted'], 2)
        self.assertEqual(len([call for call in calls if call.method == 'DELETE']), 2)

    async def test_list_environments_follows_pagination(self):
        pages = {
            '': {'environments': [{'id': 'env-1', 'size_bytes': '10'}], 'nextPageToken': 'next'},
            'next': {'environments': [{'id': 'env-2', 'size_bytes': '20'}]},
        }
        client_transport, calls = transport([], pages=pages)
        async with httpx.AsyncClient(transport=client_transport) as client:
            summary = await reclaim(client, keys=['key'], groups='', root='https://pb.fixture',
                                    token='', email='', password='', keep_hours=48,
                                    stale_hours=24, apply=False, use_pb=False,
                                    max_bytes=None, max_envs=None, now=NOW)
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['storage_bytes'], 30)
        self.assertEqual(len([call for call in calls if call.url.path == '/v1beta/environments']), 2)
