import type { FileChange, ToolCall } from '../types';
import { inferActivityMode, normalizeToolName } from './tooling';

export function hasFileEdits(calls: ToolCall[]): boolean {
  return calls.some(call => ['file.write', 'file.edit', 'file.delete'].includes(normalizeToolName(call.toolName)));
}

export function shouldOpenWorkbench(calls: ToolCall[], changes: FileChange[]): boolean {
  return changes.length > 0 || hasFileEdits(calls)
    || calls.some(call => normalizeToolName(call.toolName) === 'bash')
    || inferActivityMode(calls) === 'research';
}

export function preferredWorkbenchTab(calls: ToolCall[], changes: FileChange[]): 'diff' | 'activity' | 'progress' {
  if (changes.length > 0 || hasFileEdits(calls)) return 'diff';
  return calls.some(call => normalizeToolName(call.toolName) === 'bash') ? 'activity' : 'progress';
}
