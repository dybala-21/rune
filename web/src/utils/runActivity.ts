import type { ToolCall } from '../types';
import { normalizeToolName } from './tooling';

const LABELS: Record<string, string> = {
  'file.read': 'Reading files…', 'file.write': 'Writing files…', 'file.edit': 'Editing files…',
  'file.list': 'Looking through files…', 'file.search': 'Searching files…',
  bash: 'Running a command…', 'web.search': 'Searching the web…', 'web.fetch': 'Reading a source…',
  'browser.navigate': 'Opening a page…', 'browser.observe': 'Reading the page…',
  'browser.act': 'Working in the browser…', 'desktop.observe': 'Looking at the app…',
  'desktop.act': 'Working in the app…', 'document.create': 'Creating a document…',
  'memory.search': 'Checking saved context…', 'memory.save': 'Saving context…',
  think: 'Planning the next step…',
};

export function runActivity(calls: ToolCall[], runId: string): { state: 'thinking' | 'working'; label: string } {
  const pending = runId ? [...calls].reverse().find(call => call.runId === runId
    && call.result === undefined && call.completedAt === undefined) : undefined;
  const name = pending && normalizeToolName(pending.toolName);
  return {
    state: name && name !== 'think' ? 'working' : 'thinking',
    label: name ? LABELS[name] ?? 'Working on your request…' : 'Preparing a response…',
  };
}
