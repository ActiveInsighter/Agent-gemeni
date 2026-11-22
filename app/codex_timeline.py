"""Convert the public item lifecycle of a single Codex turn into ordered parts."""
from __future__ import annotations

from agent_timeline import tool_part

TOOL_FIELDS = {
    'commandExecution': (('command', 'cwd'), ('aggregatedOutput', 'exitCode', 'durationMs')),
    'fileChange': (('changes',), ('changes', 'aggregatedOutput')),
    'mcpToolCall': (('server', 'tool', 'arguments'), ('result', 'error', 'durationMs')),
    'dynamicToolCall': (('tool', 'arguments'), ('contentItems', 'success', 'durationMs')),
    'toolCall': (('tool', 'arguments'), ('result', 'error')),
    'webSearch': (('query', 'action'), ()),
    'plan': ((), ('text', 'plan', 'explanation')),
    'collabToolCall': (('tool', 'prompt', 'receiverThreadIds'), ('agentStatus', 'agentsStates')),
    'imageView': (('path',), ()),
}
DELTAS = {
    'item/agentMessage/delta': ('agentMessage', 'text'),
    'item/reasoning/summaryTextDelta': ('reasoning', 'summary'),
    'item/commandExecution/outputDelta': ('commandExecution', 'aggregatedOutput'),
    'item/fileChange/outputDelta': ('fileChange', 'aggregatedOutput'),
    'item/plan/delta': ('plan', 'text'),
}


def item_text(item):
    if isinstance(item.get('text'), str):
        return item['text']
    content = item.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(block['text'] for block in content if isinstance(block, dict) and isinstance(block.get('text'), str))
    return ''


def pick(item, fields):
    return {key: item[key] for key in fields if item.get(key) is not None} or None


class CodexTimeline:
    def __init__(self, progress, reply_limit):
        self.progress = progress
        self.reply_limit = reply_limit
        self.items = {}

    def consume(self, method, params):
        if method in ('item/started', 'item/completed'):
            item = params.get('item')
            if isinstance(item, dict):
                self.update(item, completed=method == 'item/completed')
        elif method == 'turn/plan/updated' and isinstance(params.get('plan'), list):
            self.update({'id': 'turn-plan', 'type': 'plan', **(pick(params, ('plan', 'explanation')) or {})},
                        completed=all(isinstance(p, dict) and p.get('status') == 'completed' for p in params['plan']))
        elif method in DELTAS and isinstance(params.get('delta'), str):
            kind, field = DELTAS[method]
            item_id = params.get('itemId') or ('answer' if kind == 'agentMessage' else '')
            if not item_id:
                return
            current = self.items.get(item_id, {})
            if current.get('_completed'):
                return
            if field == 'summary':
                index = params.get('summaryIndex', 0)
                if type(index) is not int or not 0 <= index < 500:
                    return
                sections = {**current.get('_sections', {})}
                sections[index] = (sections.get(index, '') + params['delta'])[:64000]
                delta = {'_sections': sections, 'summary': [sections[k] for k in sorted(sections)]}
            else:
                text = str(current.get(field) or '') + params['delta']
                if kind == 'agentMessage' and len(text.encode()) > self.reply_limit:
                    raise ValueError('Codex reply exceeds 2 MiB')
                delta = {field: text if kind == 'agentMessage' else text[:64000]}
                if kind != 'agentMessage' and len(text) > 64000:
                    delta['_truncated'] = True
            self.update({'id': item_id, 'type': kind, **delta})
            return field == 'aggregatedOutput' and not current.get(field)

    def update(self, item, *, completed=False):
        kind, item_id = item.get('type'), item.get('id')
        if (kind not in ('agentMessage', 'reasoning') and kind not in TOOL_FIELDS) or not isinstance(item_id, str) or not item_id:
            return
        if len(self.items) >= 9998 and item_id not in self.items:
            return
        if kind == 'agentMessage':
            # Do not retain arbitrary message content once the visible text is extracted.
            text = item_text(item)
            item = {key: item[key] for key in ('type', 'id', 'phase') if key in item}
            if text:
                item['text'] = text
        elif kind == 'reasoning':
            item = {key: item[key] for key in ('type', 'id', 'summary', '_sections') if key in item}
            if not item.get('summary'):
                item.pop('summary', None)
        else:
            fields = TOOL_FIELDS[kind][0] + TOOL_FIELDS[kind][1]
            item = pick(item, ('type', 'id', 'status', '_truncated') + fields)
        previous = self.items.get(item_id)
        if previous is None:
            previous = {'_index': len(self.items)}
        if previous.get('_completed') and not completed and item_id != 'turn-plan':
            return
        if previous.get('aggregatedOutput') and not item.get('aggregatedOutput'):
            item.pop('aggregatedOutput', None)
        current = {**previous, **item, '_completed': completed}
        if kind == 'agentMessage' and len(item_text(current).encode()) > self.reply_limit:
            raise ValueError('Codex reply exceeds 2 MiB')
        self.items[item_id] = current
        if self.progress is not None:
            part = self.part(current)
            self.progress.publish(f"step:{current['_index']}", part)

    def part(self, item):
        kind = item['type']
        raw_status = item.get('status')
        status = 'completed' if item['_completed'] else 'running'
        if raw_status in ('failed', 'error', 'declined') or item.get('success') is False or item.get('exitCode') not in (None, 0):
            status = 'failed' if item['_completed'] else 'running'
        if kind == 'agentMessage':
            return {'type': 'text', 'text': item_text(item), 'providerType': item.get('phase') or kind, 'status': status}
        if kind == 'reasoning':
            summary = item.get('summary') or []
            if isinstance(summary, str):
                summary = [summary]
            text = '\n'.join(text for text in summary if isinstance(text, str))
            return {'type': 'reasoning', 'reasoning': text, 'reasoningType': 'summary_text', 'providerType': kind, 'status': status}
        name = str(item.get('tool') or kind)
        if kind == 'mcpToolCall' and item.get('server'):
            name = f"{item['server']}/{name}"
        input_fields, output_fields = TOOL_FIELDS[kind]
        if kind == 'fileChange' and item['_completed']:
            input_fields = ()
        elif kind == 'fileChange':
            output_fields = ('aggregatedOutput',)
        part = tool_part(name[:120], item['id'][:256], kind, status=status,
                         input=pick(item, input_fields), output=pick(item, output_fields), client=kind == 'dynamicToolCall')
        if item.get('_truncated'):
            part['truncated'] = True
        return part

    def answer(self):
        answers = [item for item in self.items.values() if item['type'] == 'agentMessage' and item_text(item)]
        explicit = [item_text(item) for item in answers if item.get('phase') in ('partial_answer', 'final_answer')]
        candidates = [item_text(item) for item in answers if item.get('phase') != 'commentary']
        return '\n\n'.join(explicit) if explicit else (candidates[-1] if candidates else '')

    def parts(self):
        parts = []
        for item in self.items.values():
            part = self.part(item)
            if part['type'] == 'text' and not part['text'] or part['type'] == 'reasoning' and not part['reasoning']:
                continue
            parts.append(part)
        return parts
