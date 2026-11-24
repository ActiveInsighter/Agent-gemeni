"""Resolve RikkaHub image-library references into provider attachments."""
from __future__ import annotations

import asyncio
import base64
import json
import re

import httpx

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_INLINE_BYTES = 48 * 1024 * 1024
MAX_IMAGES = 6
R2_IMAGE = re.compile(
    r'https://file\.any1\.tech/api/library/image-sources/'
    r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}\.(jpg|png|webp|gif)'
)
IMAGE_MIMES = {'jpg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp', 'gif': 'image/gif'}
IMAGE_MARKER = re.compile(r'\n?<!--\s*aw:images:(\[.*?\])\s*-->', re.DOTALL)


def strip_image_markers(text: str) -> str:
    return IMAGE_MARKER.sub('', text or '').strip()


def image_mime(url: str) -> str:
    match = R2_IMAGE.fullmatch(url)
    if not match:
        raise RuntimeError('图片须先上传至图片资料库')
    return IMAGE_MIMES[match[1]]


def extract_image_urls(job: dict, message: dict) -> list[str]:
    urls = []
    options = job.get('options') or {}
    for container in (options, job, message):
        raw = container.get('images')
        if isinstance(raw, list):
            urls.extend(raw)
    for field in (job.get('prompt'), message.get('inputText')):
        for marker in IMAGE_MARKER.finditer(field or ''):
            try:
                parsed = json.loads(marker[1])
                if not isinstance(parsed, list):
                    raise ValueError
                urls.extend(parsed)
            except ValueError:
                raise RuntimeError('图片附件格式无效') from None
    unique = {}
    for url in urls:
        if isinstance(url, dict):
            url = url.get('url') or url.get('sourceUrl')
        if not isinstance(url, str):
            raise RuntimeError('图片附件格式无效')
        url = url.strip()
        image_mime(url)
        unique[url] = None
    if len(unique) > MAX_IMAGES:
        raise RuntimeError('每条消息最多添加 6 张图片')
    return list(unique)


async def resolve_images_to_base64(client: httpx.AsyncClient, urls: list[str]) -> list[tuple[str, str]]:
    images = []
    remaining = MAX_INLINE_BYTES
    for url in urls:
        mime = image_mime(url)
        try:
            async with asyncio.timeout(45):
                async with client.stream('GET', url, follow_redirects=False, timeout=15) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f'无法读取图片资料库附件（HTTP {response.status_code}）')
                    if response.headers.get('content-type', '').split(';', 1)[0].strip().lower() != mime:
                        raise RuntimeError('图片资料库返回的文件类型无效')
                    data = bytearray()
                    limit = min(remaining, MAX_IMAGE_BYTES)
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(data) + len(chunk) > limit:
                            raise RuntimeError('图片超过单张 25 MiB 或单次内联 48 MiB 的大小限制')
                        data.extend(chunk)
        except (httpx.HTTPError, TimeoutError):
            raise RuntimeError('读取图片资料库附件失败，请重试') from None
        if not data:
            raise RuntimeError('图片资料库返回了空图片')
        remaining -= len(data)
        images.append((mime, base64.b64encode(data).decode('ascii')))
    return images


def external_image_rejected(status: int, data: dict, credential_scope: str | None) -> bool:
    if status in (403, 413) or (status == 400 and not credential_scope):
        return True
    error = data.get('error')
    message = error.get('message') if isinstance(error, dict) else None
    return status in (404, 410) and isinstance(message, str) and (
        'cannot fetch content from the provided url' in message.lower()
        or 'url could not be found' in message.lower()
    )


def acp_image_blocks(images: list[tuple[str, str]]) -> list[dict]:
    return [{'type': 'image', 'mimeType': mime, 'data': data} for mime, data in images]
