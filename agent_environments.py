"""Report and reclaim remote sandbox environments for Gemini managed agents."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from key_pool import parse_key_groups

ROOT = "https://generativelanguage.googleapis.com/v1beta/environments"
PAGE_SIZE = 100
KEEP_QUERY_FIELDS = "providerState,status,created"


@dataclass
class RemoteEnvironment:
    id: str
    status: str = ""
    file_count: int = 0
    size_bytes: int = 0
    created: datetime | None = None
    updated: datetime | None = None
    last_accessed: datetime | None = None
    keys: set[int] = field(default_factory=set)

    @property
    def last_activity(self) -> datetime | None:
        return self.last_accessed or self.updated or self.created

    def merge(self, other: "RemoteEnvironment") -> None:
        self.keys.update(other.keys)
        self.size_bytes = max(self.size_bytes, other.size_bytes)
        self.file_count = max(self.file_count, other.file_count)
        values = [value for value in (self.last_activity, other.last_activity) if value is not None]
        if values:
            self.last_accessed = max(values)


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_environment(item: dict[str, Any], key_index: int) -> RemoteEnvironment | None:
    env_id = item.get("id")
    if not isinstance(env_id, str) or not env_id:
        return None
    return RemoteEnvironment(
        id=env_id,
        status=str(item.get("status") or ""),
        file_count=as_int(item.get("file_count")),
        size_bytes=as_int(item.get("size_bytes")),
        created=parse_timestamp(item.get("created")),
        updated=parse_timestamp(item.get("updated")),
        last_accessed=parse_timestamp(item.get("last_accessed")),
        keys={key_index},
    )


async def list_environments(client: httpx.AsyncClient, key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token = ""
    while True:
        params = {"pageSize": str(PAGE_SIZE)}
        if token:
            params["pageToken"] = token
        response = await client.get(ROOT, params=params,
                                    headers={"x-goog-api-key": key}, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"environment list failed (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("environment list returned invalid JSON") from None
        batch = data.get("environments") if isinstance(data, dict) else None
        if batch is None:
            batch = []
        if not isinstance(batch, list):
            raise RuntimeError("environment list returned an invalid payload")
        items.extend(item for item in batch if isinstance(item, dict))
        token = str(data.get("nextPageToken") or "")
        if not token:
            return items


def effective_keys(keys: list[str], raw_groups: str) -> dict[str, list[int]]:
    groups = parse_key_groups(len(keys), raw_groups)
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    return grouped


async def collect_environments(client: httpx.AsyncClient, keys: list[str], raw_groups: str) -> tuple[dict[str, RemoteEnvironment], list[str], dict[str, set[str]]]:
    environments: dict[str, RemoteEnvironment] = {}
    errors: list[str] = []
    group_envs: dict[str, set[str]] = {}
    for group, indexes in effective_keys(keys, raw_groups).items():
        last_error = ""
        for index in indexes:
            try:
                batch = await list_environments(client, keys[index])
            except (RuntimeError, httpx.HTTPError) as error:
                last_error = f"{group}: {type(error).__name__}: {error}"
                continue
            group_envs[group] = set()
            for item in batch:
                env = build_environment(item, index)
                if env is None:
                    continue
                group_envs[group].add(env.id)
                current = environments.get(env.id)
                if current is None:
                    environments[env.id] = env
                else:
                    current.merge(env)
            break
        else:
            errors.append(last_error or f"{group}: no usable key")
        await asyncio.sleep(0.2)
    return environments, errors, group_envs


def environment_ids(items: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for item in items:
        state = item.get("providerState")
        if isinstance(state, str) and state.strip():
            try:
                state = json.loads(state)
            except ValueError:
                continue
        if isinstance(state, dict) and isinstance(state.get("environmentId"), str):
            found.add(state["environmentId"])
    return found


async def fetch_message_fields(client: httpx.AsyncClient, root: str, token: str, filter_text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        response = await client.get(f"{root}/api/collections/aw_gemini_messages/records",
                                    headers={"Authorization": token},
                                    params={"perPage": 500, "page": page,
                                            "filter": filter_text, "fields": KEEP_QUERY_FIELDS},
                                    timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"PocketBase message query failed (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("PocketBase message query returned invalid JSON") from None
        batch = data.get("items") if isinstance(data, dict) else None
        if not isinstance(batch, list):
            raise RuntimeError("PocketBase message query returned an invalid payload")
        items.extend(item for item in batch if isinstance(item, dict))
        total_pages = as_int(data.get("totalPages"))
        if page >= total_pages:
            return items
        page += 1


async def keep_environments(client: httpx.AsyncClient, *, root: str, token: str,
                            email: str, password: str, keep_hours: float,
                            now: datetime) -> tuple[set[str], int]:
    from gemini_text import PocketBase
    pb = PocketBase(client, root)
    await pb.authenticate(token=token, email=email, password=password)
    cutoff = (now - timedelta(hours=keep_hours)).strftime("%Y-%m-%d %H:%M:%S.000Z")
    recent = await fetch_message_fields(client, root, pb.token, f'created>="{cutoff}"')
    active = await fetch_message_fields(client, root, pb.token,
                                        'status="running"||status="queued"')
    return environment_ids(recent) | environment_ids(active), len(recent) + len(active)


async def delete_environment(client: httpx.AsyncClient, key: str, env_id: str) -> None:
    response = await client.delete(f"{ROOT}/{env_id}", headers={"x-goog-api-key": key}, timeout=60)
    if response.status_code not in (200, 202, 204, 404):
        raise RuntimeError(f"environment delete failed (HTTP {response.status_code})")


def select_deletions(candidates: list[RemoteEnvironment], total_bytes: int, total_count: int,
                     max_bytes: int | None, max_envs: int | None) -> list[RemoteEnvironment]:
    selected: list[RemoteEnvironment] = []
    for env in candidates:
        if max_bytes is not None and total_bytes <= max_bytes:
            break
        if max_envs is not None and total_count <= max_envs:
            break
        selected.append(env)
        total_bytes -= env.size_bytes
        total_count -= 1
    if max_bytes is None and max_envs is None:
        return candidates
    remaining = len(candidates) - len(selected)
    if (max_bytes is not None and total_bytes > max_bytes) or (max_envs is not None and total_count > max_envs):
        print(f"[env] budget still exceeded; {remaining} stale environments would remain after this pass", flush=True)
    return selected


async def reclaim(client: httpx.AsyncClient, *, keys: list[str], groups: str, root: str,
                  token: str, email: str, password: str, keep_hours: float, stale_hours: float,
                  apply: bool, use_pb: bool, max_bytes: int | None, max_envs: int | None,
                  allow_empty_pb: bool = False, top: int = 5,
                  now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    environments, errors, group_envs = await collect_environments(client, keys, groups)
    project_totals = {
        group: {
            "envs": len(ids),
            "bytes": sum(environments[env_id].size_bytes for env_id in ids if env_id in environments),
        }
        for group, ids in group_envs.items()
    }
    kept: set[str] = set()
    message_count = 0
    if use_pb:
        kept, message_count = await keep_environments(
            client, root=root, token=token, email=email, password=password,
            keep_hours=keep_hours, now=now,
        )
    if use_pb and message_count == 0 and apply and not allow_empty_pb:
        raise RuntimeError("PocketBase returned no messages; refusing to delete without --allow-empty-pb")
    total_bytes = sum(env.size_bytes for env in environments.values())
    total_count = len(environments)
    print(f"[env] environments={total_count} storage={total_bytes} bytes "
          f"kept={len(kept)} messages={message_count} keys={len(keys)}", flush=True)
    for group, totals in sorted(project_totals.items(), key=lambda item: item[1]["bytes"], reverse=True):
        print(f"[env] project={group} envs={totals['envs']} storage={totals['bytes']} bytes", flush=True)
    largest = sorted(environments.values(), key=lambda env: env.size_bytes, reverse=True)[:max(0, top)]
    for env in largest:
        activity = env.last_activity.strftime("%Y-%m-%dT%H:%M:%SZ") if env.last_activity else "unknown"
        print(f"[env] largest id={env.id} bytes={env.size_bytes} files={env.file_count} "
              f"last={activity}", flush=True)
    for error in errors:
        print(f"[env] list_error {error}", flush=True)
    stale_cutoff = now - timedelta(hours=stale_hours)
    candidates = [
        env for env in environments.values()
        if env.id not in kept and env.last_activity is not None and env.last_activity < stale_cutoff
    ]
    candidates.sort(key=lambda env: env.last_activity or now)
    selected = select_deletions(candidates, total_bytes, total_count, max_bytes, max_envs)
    if not apply:
        for env in selected:
            activity = env.last_activity.strftime("%Y-%m-%dT%H:%M:%SZ") if env.last_activity else "unknown"
            print(f"[env] would_delete id={env.id} status={env.status} "
                  f"bytes={env.size_bytes} files={env.file_count} last={activity}", flush=True)
        print(f"[env] dry_run candidates={len(selected)} stale={len(candidates)}", flush=True)
        return {"total": total_count, "storage_bytes": total_bytes, "kept": len(kept),
                "messages": message_count, "candidates": len(candidates),
                "selected": len(selected), "deleted": 0, "errors": errors,
                "projects": project_totals,
                "largest": [{"id": env.id, "bytes": env.size_bytes, "files": env.file_count}
                            for env in largest]}
    deleted = 0
    reclaimed = 0
    delete_errors: list[str] = []
    for env in selected:
        key = keys[min(env.keys)]
        try:
            await delete_environment(client, key, env.id)
        except (RuntimeError, httpx.HTTPError) as error:
            delete_errors.append(f"{env.id}: {type(error).__name__}: {error}")
            print(f"[env] delete_error id={env.id} {type(error).__name__}: {error}", flush=True)
            continue
        deleted += 1
        reclaimed += env.size_bytes
        print(f"[env] deleted id={env.id} bytes={env.size_bytes}", flush=True)
    print(f"[env] done deleted={deleted} reclaimed_bytes={reclaimed} "
          f"remaining_envs={total_count - deleted} remaining_bytes={total_bytes - reclaimed}", flush=True)
    return {"total": total_count, "storage_bytes": total_bytes, "kept": len(kept),
            "messages": message_count, "candidates": len(candidates),
            "selected": len(selected), "deleted": deleted,
            "reclaimed_bytes": reclaimed, "errors": errors, "delete_errors": delete_errors,
            "projects": project_totals,
            "largest": [{"id": env.id, "bytes": env.size_bytes, "files": env.file_count}
                        for env in largest]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete stale environments instead of only reporting")
    parser.add_argument("--keep-hours", type=float, default=48.0,
                        help="Keep environments referenced by messages newer than this")
    parser.add_argument("--stale-hours", type=float, default=24.0,
                        help="Only delete environments idle for at least this long")
    parser.add_argument("--no-pb", action="store_true",
                        help="Skip PocketBase; delete purely by idle age")
    parser.add_argument("--allow-empty-pb", action="store_true",
                        help="Allow deletion when PocketBase returns no messages")
    parser.add_argument("--max-bytes", type=int, help="Target total storage across all projects")
    parser.add_argument("--max-envs", type=int, help="Target total environment count")
    parser.add_argument("--top", type=int, default=5, help="Report the largest N environments")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    keys = [key.strip() for key in os.environ.get("GEMINI_API_KEYS", "").splitlines() if key.strip()]
    if not keys:
        raise RuntimeError("GEMINI_API_KEYS is required")
    root = os.environ.get("GEMINI_PB_URL", "").rstrip("/")
    if not args.no_pb and not root:
        raise RuntimeError("GEMINI_PB_URL is required unless --no-pb is set")
    async with httpx.AsyncClient(follow_redirects=False) as client:
        summary = await reclaim(
            client, keys=keys, groups=os.environ.get("GEMINI_KEY_GROUPS", ""), root=root,
            token=os.environ.get("POCKETBASE_SERVER_TOKEN", ""),
            email=os.environ.get("GEMINI_PB_EMAIL", ""),
            password=os.environ.get("GEMINI_PB_PASSWORD", ""),
            keep_hours=args.keep_hours, stale_hours=args.stale_hours,
            apply=args.apply, use_pb=not args.no_pb,
            max_bytes=args.max_bytes, max_envs=args.max_envs,
            allow_empty_pb=args.allow_empty_pb, top=args.top,
        )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, httpx.HTTPError) as error:
        print(f"[env] setup failed: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1) from None
