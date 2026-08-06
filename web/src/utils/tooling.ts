import type { ActivitySummary, ToolCall, TrustInfo } from '../types';

/**
 * Canonical tool name: wire names use underscores (`file_read`), and the shell
 * tool ships as `bash_execute` — canonicalize the whole bash family to `bash`
 * here so every consumer can compare against one name.
 */
export function normalizeToolName(name: string): string {
  const n = name.replace(/_/g, '.');
  return n === 'bash' || n.startsWith('bash.') ? 'bash' : n;
}

export function isToolName(name: string, canonicalName: string): boolean {
  return normalizeToolName(name) === canonicalName;
}

export function isBrowserToolName(name: string): boolean {
  return normalizeToolName(name).startsWith('browser.');
}

/** File-mutation + shell tools — the "working in a folder" signal. */
export function isCodingToolName(normalized: string): boolean {
  return normalized === 'file.read' || normalized === 'file.write'
    || normalized === 'file.edit' || normalized === 'file.delete'
    || normalized === 'bash';
}

/** First non-empty string arg among the given keys — the one arg-key priority
    convention (`path`/`file_path`/… , `command`/`cmd`/…) shared by every
    tool-label surface. */
export function argString(args: Record<string, unknown>, ...keys: string[]): string | null {
  for (const k of keys) {
    const v = args[k];
    if (typeof v === 'string' && v.trim()) return v;
  }
  return null;
}

export function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) + '…' : text;
}

export function basename(path: string): string {
  const parts = path.replace(/\/+$/, '').split('/');
  return parts[parts.length - 1] || path;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

/** Evidence Gate pass count; older payloads used the `passed` key. */
export function passedChecks(trust: TrustInfo | null | undefined): number {
  const counts = trust?.evidenceGate?.verdictCounts;
  return (counts?.pass ?? counts?.passed ?? 0);
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

export type ActivityMode = 'coding' | 'research' | 'generic';

/**
 * What kind of work this run is, from the tool-call composition — structured
 * data, not text guessing. Drives which evidence the progress pane shows
 * (files/diff vs sources/queries). Stays 'generic' until enough signal
 * accumulates so the pane doesn't flap early in a run.
 */
export function inferActivityMode(toolCalls: ToolCall[]): ActivityMode {
  let coding = 0;
  let research = 0;
  for (const tc of toolCalls) {
    const name = normalizeToolName(tc.toolName);
    if (name === 'file.write' || name === 'file.edit' || name === 'file.delete' || name === 'bash') {
      coding++;
    } else if (name === 'web.search' || name === 'web.fetch' || isBrowserToolName(name)) {
      research++;
    }
  }
  if (coding + research < 2) return 'generic';
  if (coding >= research) return 'coding';
  return 'research';
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
