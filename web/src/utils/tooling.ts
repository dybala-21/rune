import type { ActivitySummary, ToolCall, TrustInfo } from '../types';

export function normalizeToolName(name: string): string {
  return name.replace(/_/g, '.');
}

export function isToolName(name: string, canonicalName: string): boolean {
  return normalizeToolName(name) === canonicalName;
}

export function isBrowserToolName(name: string): boolean {
  return normalizeToolName(name).startsWith('browser.');
}

export type WorkPhase = 'analyzing' | 'implementing' | 'verifying';

export function inferWorkPhase(toolCalls: ToolCall[]): WorkPhase {
  let hasWrites = false;
  let hasVerification = false;

  for (const tc of toolCalls) {
    const name = normalizeToolName(tc.toolName);
    if (name === 'file.write' || name === 'file.edit' || name === 'file.delete') {
      hasWrites = true;
    }
    if (hasWrites && name === 'bash') {
      hasVerification = true;
    }
  }

  if (hasVerification) return 'verifying';
  if (hasWrites) return 'implementing';
  return 'analyzing';
}

/**
 * The single run-verdict rule shared by every surface (status pip, workbench,
 * chat card) so they never disagree. Prefer the real trust result; count a
 * "verified" only when an Evidence Gate check actually ran — a plain completion
 * with no check must not claim verified, so it falls back to the tool-activity
 * heuristic. Returns null when there's no verdict to show yet.
 */
export function computeRunVerdict(
  trust: TrustInfo | null | undefined,
  activitySummary: ActivitySummary | null | undefined,
): boolean | null {
  if (trust && !trust.verified) return false;
  if (trust?.verified && trust.evidenceGate?.hasCheck) return true;
  return activitySummary ? activitySummary.success : null;
}

export function computeActivitySummary(
  toolCalls: ToolCall[],
  totalDurationMs = 0,
  success = true,
): ActivitySummary {
  return {
    success,
    totalToolCalls: toolCalls.length,
    filesRead: toolCalls.filter(tc => isToolName(tc.toolName, 'file.read')).length,
    filesWritten: toolCalls.filter(tc => isToolName(tc.toolName, 'file.write') || isToolName(tc.toolName, 'file.edit')).length,
    bashExecutions: toolCalls.filter(tc => isToolName(tc.toolName, 'bash')).length,
    webSearches: toolCalls.filter(tc => isToolName(tc.toolName, 'web.search') || isToolName(tc.toolName, 'web.fetch')).length,
    browserActions: toolCalls.filter(tc => isBrowserToolName(tc.toolName)).length,
    totalDurationMs,
  };
}
