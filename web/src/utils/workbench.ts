import type { FileChange, ToolCall } from '../types';
import { inferActivityMode, normalizeToolName } from './tooling';

export function desktopAttention(calls: ToolCall[], running: boolean): 'connection' | 'action' | null {
  if (!running) return null;
  const runId = calls[calls.length - 1]?.runId;
  const pending = [...calls].reverse().find(call => call.runId === runId && call.result === undefined);
  const name = pending && normalizeToolName(pending.toolName);
  return name === 'desktop.connect' ? 'connection' : name === 'desktop.act' ? 'action' : null;
}

export function hasFileEdits(calls: ToolCall[]): boolean {
  return calls.some(call => ['file.write', 'file.edit', 'file.delete'].includes(normalizeToolName(call.toolName)));
}

export function shouldOpenWorkbench(calls: ToolCall[], changes: FileChange[]): boolean {
  return changes.length > 0 || hasFileEdits(calls)
    || calls.some(call => normalizeToolName(call.toolName) === 'bash')
    || calls.some(call => normalizeToolName(call.toolName).startsWith('browser.'))
    || calls.some(call => normalizeToolName(call.toolName).startsWith('desktop.'))
    || inferActivityMode(calls) === 'research';
}

export function preferredWorkbenchTab(calls: ToolCall[], changes: FileChange[]): 'diff' | 'activity' | 'progress' | 'computer' {
  if (changes.length > 0 || hasFileEdits(calls)) return 'diff';
  if (calls.some(call => normalizeToolName(call.toolName).startsWith('browser.'))) return 'computer';
  if (calls.some(call => normalizeToolName(call.toolName).startsWith('desktop.'))) return 'computer';
  return calls.some(call => normalizeToolName(call.toolName) === 'bash') ? 'activity' : 'progress';
}
