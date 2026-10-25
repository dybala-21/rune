import type { ToolCall } from '../types';
import { argString, normalizeToolName } from './tooling';
import { sourceUrl } from './searchActivity';

export type ToolState = 'pending' | 'waiting' | 'succeeded' | 'failed' | 'check-failed' | 'unconfirmed' | 'interrupted' | 'unknown';
export type ToolKind = 'file' | 'terminal' | 'web' | 'browser' | 'desktop' | 'code' | 'memory' | 'task' | 'clock' | 'connection' | 'tool';

const LABELS: Record<string, string> = {
  'file.read': 'Read file', 'file.write': 'Write file', 'file.edit': 'Edit file', 'file.delete': 'Delete file',
  'file.list': 'Browse files', 'file.search': 'Search files', 'file.query': 'Query data',
  bash: 'Run command', 'managed.service.status': 'Service status', 'managed.service.stop': 'Stop service',
  'web.fetch': 'Read web page', 'web.search': 'Web search',
  'browser.navigate': 'Navigate', 'browser.open': 'Open tab', 'browser.observe': 'Inspect page',
  'browser.act': 'Browser action', 'browser.screenshot': 'Capture page', 'browser.extract': 'Read page content',
  'browser.find': 'Find on page', 'browser.discover.apis': 'Inspect page requests',
  'browser.batch': 'Browser actions', 'browser.workflow': 'Browser workflow', 'browser.profile': 'Browser profile',
  'desktop.connect': 'Connect apps', 'desktop.apps': 'Available apps', 'desktop.open': 'Open app',
  'desktop.observe': 'Inspect app', 'desktop.wait': 'Wait for app', 'desktop.act': 'App action', 'desktop.phase': 'Switch work mode',
  'document.create': 'Create document', 'document.read': 'Read document', 'document.preview': 'Preview document',
  'document.bundle': 'Create documents', 'document.bundle.inspect': 'Inspect documents', 'document.bundle.update': 'Update documents',
  'code.analyze': 'Analyze code', 'code.find.def': 'Find definition', 'code.find.refs': 'Find references', 'code.impact': 'Check code impact',
  'project.map': 'Explore project', 'memory.search': 'Search memory', 'memory.save': 'Save memory',
  'memory.tune': 'Update memory settings', 'conversation.search': 'Search conversations',
  'delegate.task': 'Delegate task', 'delegate.orchestrate': 'Coordinate tasks',
  'task.create': 'Create task', 'task.update': 'Update task', 'task.list': 'List tasks', 'task.blocked': 'Report blocker',
  'cron.create': 'Create schedule', 'cron.update': 'Update schedule', 'cron.delete': 'Delete schedule', 'cron.list': 'List schedules',
  'connector.list': 'Available connectors', 'connector.request': 'Connector request',
  'service.connect': 'Connect service', 'service.status': 'Service status', 'service.list': 'Available services',
  'service.disconnect': 'Disconnect service', 'service.reconnect': 'Reconnect service', 'credential.save': 'Save credential',
  'skill.search': 'Find skill', 'skill.load': 'Read skill', 'skill.create': 'Create skill', 'skill.promote': 'Publish skill',
  'safety.tune': 'Update safety settings', think: 'Plan next steps', 'ask.user': 'Question',
};

const ACTIONS: Record<string, string> = {
  click: 'Click', type: 'Enter text', scroll: 'Scroll', select: 'Select option', check: 'Check', uncheck: 'Uncheck',
  press: 'Press control', set_value: 'Set value', key: 'Press key', drag: 'Drag', publish: 'Share saved file',
};

export function toolState(tool: ToolCall, active: boolean): ToolState {
  if (tool.success === false) return 'failed';
  if (tool.checkStatus === 'fail') return 'check-failed';
  if (tool.checkStatus === 'inconclusive') return 'unconfirmed';
  if (tool.result === undefined) {
    if (tool.completedAt != null) return 'unknown';
    if (!active) return 'interrupted';
    return normalizeToolName(tool.toolName) === 'desktop.connect' ? 'waiting' : 'pending';
  }
  return tool.success === true ? 'succeeded' : 'unknown';
}

export const STATE_LABELS: Record<ToolState, string> = {
  pending: 'Running', waiting: 'Waiting for you', succeeded: 'Succeeded', failed: 'Failed',
  'check-failed': 'Checks failed', unconfirmed: 'Checks unconfirmed', interrupted: 'Interrupted', unknown: 'Status unavailable',
};

export function toolKind(name: string): ToolKind {
  const normalized = normalizeToolName(name);
  const family = normalized.split('.')[0];
  if (normalized === 'bash' || normalized.startsWith('managed.service.')) return 'terminal';
  if (family === 'document' || family === 'project') return 'file';
  if (['service', 'connector', 'credential'].includes(family)) return 'connection';
  if (family === 'cron') return 'clock';
  if (family === 'delegate') return 'task';
  if (family === 'conversation') return 'memory';
  if (['file', 'web', 'browser', 'desktop', 'code', 'memory', 'task'].includes(family)) return family as ToolKind;
  return 'tool';
}

export function toolLabel(name: string): string {
  const normalized = normalizeToolName(name);
  const label = normalized.replace(/[.]+/g, ' ').trim();
  return LABELS[normalized] ?? (label.charAt(0).toUpperCase() + label.slice(1));
}

export interface ToolPresentation {
  label: string;
  kind: ToolKind;
  target: string;
  context: string;
  href?: string;
  monospace?: boolean;
}

export function presentTool(tool: ToolCall): ToolPresentation {
  const name = normalizeToolName(tool.toolName);
  const args = tool.args;
  const kind = toolKind(name);
  const label = toolLabel(name);
  const path = argString(args, 'path', 'file_path', 'directory') ?? '';
  const filename = path.replace(/\\/g, '/').replace(/\/$/, '').split('/').pop() || path;
  if (kind === 'terminal') return { label, kind, target: argString(args, 'command', 'cmd', 'service_id') ?? '', context: argString(args, 'cwd') ?? '', monospace: true };
  if (kind === 'file' || kind === 'code') {
    const query = argString(args, 'pattern', 'name', 'symbol_name');
    const searching = ['file.search', 'code.find.def', 'code.find.refs'].includes(name);
    return { label, kind, target: (searching ? query : filename) || query || argString(args, 'title') || '', context: path };
  }
  if (kind === 'web' || kind === 'browser') {
    const href = sourceUrl(argString(args, 'url') ?? '');
    const action = argString(args, 'action') ?? '';
    const method = argString(args, 'method');
    return { label: name === 'browser.act' ? ACTIONS[action] ?? label : label, kind,
      target: href ? new URL(href).host : argString(args, 'selector', 'query', 'taskHint', 'name') ?? '',
      context: [method && method !== 'GET' ? method : '', href ? new URL(href).pathname : ''].filter(Boolean).join(' · '), href: href ?? undefined };
  }
  if (kind === 'desktop') {
    const action = argString(args, 'action') ?? '';
    const modifiers = Array.isArray(args.modifiers) ? args.modifiers.filter(v => typeof v === 'string') : [];
    const target = action === 'key' ? [...modifiers, argString(args, 'key')].filter(Boolean).join(' + ')
      : argString(args, 'app', 'ref', 'phase') ?? '';
    return { label: name === 'desktop.act' ? ACTIONS[action] ?? label : label, kind, target, context: 'This Mac' };
  }
  if (kind === 'connection') return { label, kind, target: argString(args, 'connector', 'service', 'name') ?? '',
    context: [argString(args, 'method'), path].filter(Boolean).join(' ') };
  return { label, kind, target: argString(args, 'query', 'title', 'name', 'goal', 'task_id', 'job_id', 'skill_name') ?? '', context: '' };
}

export function activityStatus(tools: ToolCall[], active: boolean) {
  const counts = new Map<ToolState, number>();
  for (const tool of tools) {
    const state = toolState(tool, active);
    counts.set(state, (counts.get(state) ?? 0) + 1);
  }
  const order: ToolState[] = ['failed', 'check-failed', 'waiting', 'pending', 'interrupted', 'unconfirmed', 'unknown', 'succeeded'];
  const state = order.find(value => counts.has(value)) ?? 'unknown';
  const text = tools.length === 1 ? STATE_LABELS[state] : order.filter(value => counts.has(value)).map(value => {
    const count = counts.get(value)!;
    if (value === 'unknown') return `${count} status unavailable`;
    if (value === 'check-failed') return `${count} with failed checks`;
    if (value === 'unconfirmed') return `${count} with unconfirmed checks`;
    return `${count} ${STATE_LABELS[value].toLowerCase()}`;
  }).join(' · ');
  return { state, text, running: counts.has('pending') || counts.has('waiting') };
}
