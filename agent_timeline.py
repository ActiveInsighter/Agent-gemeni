"""Public, provider-neutral timeline shared by progress and completed messages."""
from __future__ import annotations

import json

from response_capture import _secret_values, _try_parse_structured_output, sanitize

VERSION = 1
MAX_TIMELINE_BYTES = 5 * 1024 * 1024
MAX_BLOCK_BYTES = 64 * 1024
MAX_PARTS = 9998


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode())


def bounded_part(part, *, maximum=MAX_BLOCK_BYTES):
    original = part
    part = dict(part)
    encoded = json.dumps(part, ensure_ascii=False)
    if '_captureOmitted' in encoded or '[truncated]' in encoded:
        part['truncated'] = True
    if encoded_size(part) <= maximum:
        return part
    part['truncated'] = True
    if part['type'] in ('tool', 'server_tool'):
        for budget in (maximum // 3, maximum // 12, 1024):
            for key in ('input', 'output'):
                if key in original:
                    part[key] = compact_details(original[key], budget)
            if encoded_size(part) <= maximum:
                return part
    for key in ('output', 'input', 'url', 'reasoning', 'text', 'annotations'):
        if key not in part or encoded_size(part) <= maximum:
            continue
        value = original[key]
        if isinstance(value, str) and key != 'url':
            # Bound escaped JSON bytes as well as UTF-8 bytes.
            part[key] = ''
            start, end = 0, len(value)
            while start < end:
                middle = (start + end + 1) // 2
                part[key] = value[:middle]
                if encoded_size(part) <= maximum:
                    start = middle
                else:
                    end = middle - 1
            part[key] = value[:start]
        else:
            part.pop(key)
    return part



def compact_details(value, max_text):
    if isinstance(value, str):
        return value.encode()[:max_text].decode('utf-8', errors='ignore')
    if isinstance(value, dict):
        return {key: compact_details(item, max_text) for key, item in list(value.items())[:40]}
    if isinstance(value, list):
        return [compact_details(item, max_text) for item in value[:30]]
    return value


def citations(items):
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get('url') or item.get('uri')
        if not isinstance(url, str) or not url.lower().startswith(('https://', 'http://')) or len(url) > 4096:
            continue
        title = str(item.get('title') or url)[:1024]
        start = item.get('startIndex', item.get('start_index'))
        end = item.get('endIndex', item.get('end_index'))
        offsets = {'startIndex': start, 'endIndex': end} if type(start) is int and type(end) is int and 0 <= start < end <= 2**31 - 1 else {}
        key = (url, start, end) if offsets else (url,)
        if key not in seen:
            result.append({'type': 'url_citation', 'title': title, 'url': url, **offsets})
            seen.add(key)
        if len(result) >= 200:
            break
    return result


def tool_part(name, call_id, provider_type, *, status='running', input=None, output=None, client=False):
    return {'type': 'tool' if client else 'server_tool', 'toolName': name, 'toolCallId': call_id,
            'providerType': provider_type, 'status': status, 'input': input, 'output': output}


def normalize_part(part, *, terminal=False):
    kind = part['type']
    normalized = {key: part[key] for key in ('type', 'providerType', 'truncated') if key in part}
    status = {'in_progress': 'running'}.get(part.get('status'), part.get('status'))
    if status not in ('running', 'completed', 'failed', 'canceled'):
        status = 'completed' if kind not in ('tool', 'server_tool') or part.get('output') else 'running'
    if terminal and status == 'running':
        status = 'canceled'
    normalized['status'] = status
    if kind in ('tool', 'server_tool'):
        normalized.update({key: part[key] for key in ('toolCallId', 'toolName', 'input', 'output') if key in part})
        # Older capture helpers prepared UIMessagePart.Tool directly. Keep the
        # wire data as JSON for both tool types.
        if kind == 'tool':
            for key in ('input', 'output'):
                value = normalized.get(key)
                if key == 'output' and isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and value[0].get('type') == 'text':
                    value = value[0].get('text')
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                normalized[key] = value
    elif kind == 'reasoning':
        normalized.update(reasoning=part.get('reasoning', ''), reasoningType='summary_text')
    elif kind == 'text':
        normalized['text'] = part.get('text', '')
        if part.get('annotations'):
            normalized['annotations'] = citations(part['annotations'])
    elif kind in ('image', 'document'):
        normalized.update(url=part.get('url') or part.get('uri') or '', mimeType=part.get('mimeType') or '')
        if part.get('fileName'):
            normalized['fileName'] = part['fileName']
    elif kind == 'annotations':
        normalized['annotations'] = citations(part.get('annotations') or [])
    return normalized


def make_timeline(parts, *, annotations=(), secrets=()):
    blocks = []
    size = 128
    omitted = 0
    for part in parts:
        clean = normalize_part(part, terminal=True)
        text = clean.pop('text', None)
        clean = sanitize(clean, secrets=secrets)
        if text is not None:
            clean['text'] = sanitize(text, secrets=secrets, text_limit=2 * 1024 * 1024, parse_strings=False)
        limit = 4 * 1024 * 1024 if clean['type'] == 'text' else MAX_BLOCK_BYTES
        clean = bounded_part(clean, maximum=limit)
        clean['sequence'] = len(blocks) + 1
        weight = encoded_size(clean) + 2
        if len(blocks) >= MAX_PARTS or size + weight > MAX_TIMELINE_BYTES - 300 * 1024:
            omitted += 1
            continue
        blocks.append(clean)
        size += weight
    refs = sanitize(citations(annotations), secrets=secrets)
    while encoded_size(refs) > 256 * 1024:
        refs.pop()
    return {'version': VERSION, 'parts': blocks, 'annotations': refs, 'omittedParts': omitted}


def prepare_result(result, protocol, progress):
    """Build the only terminal payload written to PocketBase, including partial failures."""
    result = {**result}
    secrets = _secret_values(progress.redactor.secrets)
    parts = result.pop('parts', None)
    annotations = result.pop('annotations', None)
    if not parts:
        parts = [row['part'] for row in progress.timeline_rows()]
    refs = list(annotations or ())
    for part in parts:
        if part['type'] == 'annotations':
            refs.extend(part.get('annotations') or [])
    display_parts = [part for part in parts if part['type'] not in ('status', 'annotations')]
    if result.get('reply') and not any(p['type'] == 'text' and p.get('text') for p in display_parts):
        display_parts.append({'type': 'text', 'text': result['reply']})
    result['timeline'] = make_timeline(display_parts, annotations=refs, secrets=secrets)
    result.setdefault('protocol', protocol)
    if 'structuredOutput' not in result:
        result['structuredOutput'] = _try_parse_structured_output(result.get('reply', ''), secrets=secrets)
    for key in ('thinkingSummary', 'toolCalls', 'codeExecutions', 'grounding', 'response'):
        result.pop(key, None)
    return result


def wire_row(row):
    """Translate the executor's progress buffer into a timeline upsert."""
    kind = row['type']
    payload = row.get('payload')
    if isinstance(payload, dict) and payload.get('type') in ('server_tool', 'tool', 'text', 'reasoning', 'image', 'document'):
        part = {**payload, 'status': row['status']}
    elif kind in ('tool_call', 'tool_result'):
        if payload is None and row['content']:
            try:
                payload = json.loads(row['content'])
            except ValueError:
                payload = row['content']
        part = tool_part(row.get('toolName') or row.get('providerType') or 'tool',
                         row.get('callId') or row['key'], row.get('providerType') or kind,
                         status=row['status'], client=row.get('toolKind') == 'client',
                         input=payload if kind == 'tool_call' else None, output=payload if kind == 'tool_result' else None)
    elif kind in ('image', 'annotations'):
        try:
            data = json.loads(row['content'])
        except ValueError:
            data = {} if kind == 'image' else []
        part = normalize_part({'type': kind, 'status': row['status'], **(data if kind == 'image' else {'annotations': data})})
    elif kind == 'reasoning_summary':
        part = {'type': 'reasoning', 'reasoning': row['content'], 'reasoningType': 'summary_text', 'status': row['status']}
    else:
        part = {'type': kind, 'text': row['content'], 'status': row['status']}
    if row.get('providerType') and not part.get('providerType'):
        part['providerType'] = row['providerType']
    if row.get('truncated'):
        part['truncated'] = True
    return {'key': row['key'], 'sequence': row['sequence'], 'part': bounded_part(part, maximum=60 * 1024)}
