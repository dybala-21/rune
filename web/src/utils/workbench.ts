import type { FileChange, ToolCall, TrustInfo } from '../types';
import { argString, inferActivityMode, normalizeToolName } from './tooling';

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
  return changes.length > 0 || hasFileEdits(calls) || hasDocumentWork(calls)
    || calls.some(call => normalizeToolName(call.toolName) === 'bash')
    || calls.some(call => normalizeToolName(call.toolName).startsWith('browser.'))
    || calls.some(call => normalizeToolName(call.toolName).startsWith('desktop.'))
    || inferActivityMode(calls) === 'research';
}

export function preferredWorkbenchTab(calls: ToolCall[], changes: FileChange[]): 'diff' | 'activity' | 'progress' | 'computer' | 'outputs' {
  if (hasDocumentWork(calls)) return 'outputs';
  if (changes.length > 0 || hasFileEdits(calls)) return 'diff';
  if (calls.some(call => normalizeToolName(call.toolName).startsWith('browser.'))) return 'computer';
  if (calls.some(call => normalizeToolName(call.toolName).startsWith('desktop.'))) return 'computer';
  return calls.some(call => normalizeToolName(call.toolName) === 'bash') ? 'activity' : 'progress';
}


function hasDocumentWork(calls: ToolCall[]): boolean {
  return calls.some(call => ['document.create', 'document.bundle', 'document.bundle.update'].includes(normalizeToolName(call.toolName)));
}

export interface WorkbenchOutput {
  path: string;
  sessionId: string;
  state: 'created' | 'pending' | 'failed' | 'unknown';
}

export function workbenchOutputs(calls: ToolCall[], sessionId: string, trust?: TrustInfo | null): WorkbenchOutput[] {
  const outputs = new Map<string, WorkbenchOutput>();
  for (const call of calls) {
    if (normalizeToolName(call.toolName) !== 'document.create') continue;
    const path = argString(call.args, 'path');
    if (!path) continue;
    outputs.set(path, { path, sessionId, state: call.result === undefined ? 'pending'
      : call.success === true ? 'created' : call.success === false ? 'failed' : 'unknown' });
  }
  for (const receipt of trust?.artifactReceipts ?? []) {
    const artifacts = receipt.scope === 'download' ? [receipt] : receipt.artifacts;
    for (const artifact of artifacts) {
      if (!outputs.has(artifact.path)) outputs.set(artifact.path, {
        path: artifact.path, sessionId: receipt.scope === 'download' ? receipt.sessionId : sessionId,
        state: receipt.scope === 'download' && !receipt.verified ? 'unknown' : 'created',
      });
    }
  }
  return [...outputs.values()];
}
