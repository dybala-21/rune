import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type {
  ActivitySummary,
  OrchestrationState,
  StepInfo,
  ToolCall,
  TrustInfo,
} from '../types';
import {
  normalizeToolName,
  isBrowserToolName,
  isBashToolName,
  inferActivityMode,
  computeRunVerdict,
} from '../utils/tooling';

/**
 * Progress pane: the supervision view of a run, designed to answer one
 * question at a glance — "what is it doing / how did it end?". Top-down:
 * a status band (live action while running, verdict once finished), the
 * delegated-task checklist when one exists, the step timeline of what
 * actually happened, and the evidence the run touched (files for coding,
 * sources for research). Status is always glyph + color + text — never emoji.
 */

interface ProgressPaneProps {
  toolCalls: ToolCall[];
  isRunning: boolean;
  currentStep: StepInfo | null;
  trust?: TrustInfo | null;
  activitySummary: ActivitySummary | null;
  orchestration: OrchestrationState | null;
  /** The agent is blocked on the user — takes priority over every other band. */
  awaiting?: 'approval' | 'question' | null;
  onOpenFile?: (path: string) => void;
}

function argString(args: Record<string, unknown>, ...keys: string[]): string | null {
  for (const k of keys) {
    const v = args[k];
    if (typeof v === 'string' && v.trim()) return v;
  }
  return null;
}

function basename(path: string): string {
  const parts = path.replace(/\/+$/, '').split('/');
  return parts[parts.length - 1] || path;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) + '…' : text;
}

/** Last two path segments — enough to identify a file without the noise of a
    full absolute path (which stays available as the tooltip / open target). */
function shortPath(path: string): string {
  const parts = path.replace(/\/+$/, '').split('/').filter(Boolean);
  return parts.length > 2 ? parts.slice(-2).join('/') : path;
}

/** One human-readable "verb target" fragment for a tool call. */
function callLabel(tc: ToolCall): string {
  const name = normalizeToolName(tc.toolName);
  const path = argString(tc.args, 'path', 'file_path', 'file', 'target');
  if (isBashToolName(name)) {
    const cmd = argString(tc.args, 'command', 'cmd', 'script');
    return cmd ? `run ${truncate(cmd, 40)}` : 'run command';
  }
  switch (name) {
    case 'file.read': return path ? `read ${basename(path)}` : 'read file';
    case 'file.write': return path ? `write ${basename(path)}` : 'write file';
    case 'file.edit': return path ? `edit ${basename(path)}` : 'edit file';
    case 'file.delete': return path ? `delete ${basename(path)}` : 'delete file';
    case 'web.search': {
      const q = argString(tc.args, 'query', 'q');
      return q ? `search "${truncate(q, 32)}"` : 'search web';
    }
    case 'web.fetch': {
      const url = argString(tc.args, 'url');
      return url ? `fetch ${hostOf(url)}` : 'fetch page';
    }
    default: {
      if (isBrowserToolName(name)) {
        const url = argString(tc.args, 'url');
        return url ? `browse ${hostOf(url)}` : name.replace('browser.', 'browser ');
      }
      return name;
    }
  }
}

interface StepGroup {
  step: number;
  calls: ToolCall[];
}

function groupBySteps(toolCalls: ToolCall[]): StepGroup[] {
  const groups: StepGroup[] = [];
  for (const tc of toolCalls) {
    const step = tc.step ?? 0;
    const last = groups[groups.length - 1];
    if (last && last.step === step) last.calls.push(tc);
    else groups.push({ step, calls: [tc] });
  }
  return groups;
}

/** Condense a step's calls to at most two labels plus a "+n" tail. */
function stepSummary(calls: ToolCall[]): string {
  const labels: string[] = [];
  for (const tc of calls) {
    const l = callLabel(tc);
    if (!labels.includes(l)) labels.push(l);
  }
  const shown = labels.slice(0, 2).join(' · ');
  const extra = labels.length - 2;
  return extra > 0 ? `${shown} +${extra}` : shown;
}

type Glyph = 'done' | 'failed' | 'running' | 'pending';

function StatusGlyph({ kind }: { kind: Glyph }) {
  if (kind === 'running') {
    return <span className="spinner" style={{ width: 11, height: 11, flexShrink: 0 }} />;
  }
  const map: Record<Exclude<Glyph, 'running'>, [string, string]> = {
    done: ['✓', 'var(--success)'],
    failed: ['✗', 'var(--danger)'],
    pending: ['○', 'var(--text-muted)'],
  };
  const [glyph, color] = map[kind];
  return <span style={{ color, flexShrink: 0, width: 14, textAlign: 'center' }}>{glyph}</span>;
}

/** Section header: quiet label + count, so the pane scans as an outline. */
function SectionTitle({ label, count }: { label: string; count?: number }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'baseline',
      gap: 6,
      fontSize: 10,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: 'var(--text-muted)',
      margin: '16px 0 6px',
    }}>
      <span>{label}</span>
      {count !== undefined && count > 0 && (
        <span style={{ fontFamily: 'var(--font-mono)' }}>{count}</span>
      )}
    </div>
  );
}

/**
 * The one-glance answer at the top of the pane. Blocked on the user: an amber
 * call to action. While running: what the agent is doing right now. Finished:
 * the run's verify-or-fail-honestly verdict as a full-width band, expandable
 * to the actual Evidence Gate output — the receipt, not just the claim.
 */
function StatusBand({ isRunning, awaiting, nowLabel, stepNumber, verdictOk, trust }: {
  isRunning: boolean;
  awaiting: 'approval' | 'question' | null;
  nowLabel: string | null;
  stepNumber: number | null;
  verdictOk: boolean | null;
  trust?: TrustInfo | null;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const evidence = trust?.evidenceGate?.lastEvidence?.trim() || null;

  const evidenceDisclosure = evidence && (
    <>
      <button
        type="button"
        onClick={() => setShowEvidence(v => !v)}
        style={{
          alignSelf: 'flex-start',
          background: 'none', border: 'none', padding: '0 0 0 21px', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 10.5, fontFamily: 'var(--font-mono)',
        }}
      >
        {showEvidence ? '▾ hide evidence' : '▸ show evidence'}
      </button>
      {showEvidence && (
        <pre style={{
          margin: '2px 0 0 21px', padding: '6px 8px',
          background: 'var(--bg-primary)', borderRadius: 6,
          border: '1px solid var(--border-subtle, var(--border))',
          color: 'var(--text-secondary, var(--text-primary))',
          fontSize: 10.5, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          maxHeight: 140, overflow: 'auto',
        }}>
          {trust?.evidenceGate?.lastVerdict ? `verdict: ${trust.evidenceGate.lastVerdict}\n` : ''}{evidence}
        </pre>
      )}
    </>
  );

  const band = (color: string, bg: string, body: ReactNode) => (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 3,
      padding: '9px 12px',
      borderRadius: 8,
      border: `1px solid ${color}`,
      background: bg,
    }}>
      {body}
    </div>
  );

  if (awaiting) {
    return band('var(--warning)', 'var(--warning-subtle, var(--bg-secondary))', (
      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
        <span style={{ color: 'var(--warning)' }}>◐</span>
        <span style={{ color: 'var(--text-primary)', fontSize: 12.5, fontWeight: 600 }}>
          {awaiting === 'approval' ? 'Waiting for your approval' : 'Waiting for your answer'}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 'auto' }}>
          respond in the chat
        </span>
      </div>
    ));
  }
  if (isRunning) {
    return band('var(--border)', 'var(--bg-secondary)', (
      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
        <span className="spinner" style={{ width: 12, height: 12, flexShrink: 0 }} />
        <span style={{
          color: 'var(--text-primary)', fontSize: 12.5, flex: 1,
          fontFamily: 'var(--font-mono)', wordBreak: 'break-word',
        }}>
          {nowLabel ?? 'working…'}
        </span>
        {stepNumber !== null && (
          <span style={{ color: 'var(--text-muted)', fontSize: 10.5, flexShrink: 0 }}>
            step {stepNumber}
          </span>
        )}
      </div>
    ));
  }
  if (verdictOk === null) return null;
  if (verdictOk) {
    return band('var(--success)', 'var(--success-subtle)', (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <span style={{ color: 'var(--success)' }}>✓</span>
          <span style={{ color: 'var(--text-primary)', fontSize: 12.5, fontWeight: 600 }}>
            Verified
          </span>
          {trust?.evidenceGate && (trust.evidenceGate.verdictCounts?.pass ?? 0) > 0 && (
            <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 'auto' }}>
              {trust.evidenceGate.verdictCounts.pass} checks passed
            </span>
          )}
        </div>
        {evidenceDisclosure}
      </>
    ));
  }
  const honest = Boolean(trust && !trust.verified);
  return band(
    honest ? 'var(--warning)' : 'var(--danger)',
    honest ? 'var(--warning-subtle, var(--danger-subtle))' : 'var(--danger-subtle)',
    (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <span style={{ color: honest ? 'var(--warning)' : 'var(--danger)' }}>✗</span>
          <span style={{ color: 'var(--text-primary)', fontSize: 12.5, fontWeight: 600 }}>
            {honest ? 'Not verified — honest stop' : 'Failed'}
          </span>
        </div>
        {trust?.honestNote && (
          <div style={{ color: 'var(--text-primary)', fontSize: 11.5, paddingLeft: 21 }}>
            {trust.honestNote}
          </div>
        )}
        {trust?.escalationHint && (
          <div style={{ color: 'var(--text-muted)', fontSize: 11, paddingLeft: 21 }}>
            {trust.escalationHint}
          </div>
        )}
        {evidenceDisclosure}
      </>
    ),
  );
}

export const ProgressPane = memo(function ProgressPane({
  toolCalls, isRunning, currentStep, trust, activitySummary, orchestration, awaiting = null, onOpenFile,
}: ProgressPaneProps) {
  const mode = inferActivityMode(toolCalls);
  const groups = useMemo(() => groupBySteps(toolCalls), [toolCalls]);
  const verdictOk = computeRunVerdict(trust, activitySummary);

  // What the agent is doing right now: the newest still-pending call, falling
  // back to the newest call of any state.
  const nowLabel = useMemo(() => {
    if (toolCalls.length === 0) return null;
    const pending = [...toolCalls].reverse().find(tc => tc.result === undefined);
    return callLabel(pending ?? toolCalls[toolCalls.length - 1]);
  }, [toolCalls]);

  // Files the run touched, with the verbs it used on them, in first-touch order.
  const files = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const tc of toolCalls) {
      const name = normalizeToolName(tc.toolName);
      if (!name.startsWith('file.')) continue;
      const path = argString(tc.args, 'path', 'file_path', 'file', 'target');
      if (!path) continue;
      if (!map.has(path)) map.set(path, new Set());
      map.get(path)!.add(name.replace('file.', ''));
    }
    return [...map.entries()];
  }, [toolCalls]);

  // Sources consulted: searches, and fetched/browsed pages grouped by
  // host+path — paginated fetches (same page, different query string) collapse
  // into one row with a ×N count instead of reading as duplicates.
  const sources = useMemo(() => {
    const queries: string[] = [];
    const pages = new Map<string, { url: string; count: number }>();
    for (const tc of toolCalls) {
      const name = normalizeToolName(tc.toolName);
      if (name === 'web.search') {
        const q = argString(tc.args, 'query', 'q');
        if (q && !queries.includes(q)) queries.push(q);
      } else if (name === 'web.fetch' || isBrowserToolName(name)) {
        const url = argString(tc.args, 'url');
        if (!url) continue;
        let key = url;
        try {
          const u = new URL(url);
          key = u.host + u.pathname;
        } catch { /* non-URL string: group by the raw value */ }
        const entry = pages.get(key);
        if (entry) entry.count += 1;
        else pages.set(key, { url, count: 1 });
      }
    }
    return { queries, pages: [...pages.entries()] };
  }, [toolCalls]);

  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = listRef.current;
    if (el && isRunning) el.scrollTop = el.scrollHeight;
  }, [toolCalls.length, isRunning]);

  if (toolCalls.length === 0 && !orchestration) {
    return (
      <div style={{ flex: 1, padding: 14, color: 'var(--text-muted)', fontSize: 12.5 }}>
        Waiting for the agent to start working…
      </div>
    );
  }

  const rowStyle = {
    display: 'flex',
    gap: 8,
    alignItems: 'baseline',
    padding: '3px 0',
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
  };

  return (
    <div ref={listRef} style={{
      flex: 1,
      overflow: 'auto',
      padding: '12px 14px',
      fontSize: 12,
      lineHeight: 1.55,
    }}>
      <StatusBand
        isRunning={isRunning}
        awaiting={awaiting}
        nowLabel={nowLabel}
        stepNumber={currentStep?.stepNumber ?? null}
        verdictOk={verdictOk}
        trust={trust}
      />

      {/* Delegated-task checklist — real plan data, only when it exists. */}
      {orchestration && (
        <>
          <SectionTitle label="Tasks" count={orchestration.total} />
          {orchestration.description && (
            <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>
              {truncate(orchestration.description, 120)}
            </div>
          )}
          <div style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 4 }}>
            {orchestration.completed}/{orchestration.total} complete
          </div>
          {orchestration.tasks.map(t => (
            <div key={t.taskId} style={rowStyle}>
              <StatusGlyph kind={
                t.success === true ? 'done'
                  : t.success === false ? 'failed'
                  : isRunning ? 'running' : 'pending'
              } />
              <span style={{ color: 'var(--text-primary)', flex: 1 }}>
                {t.description || t.taskId}
              </span>
              {t.role && <span style={{ color: 'var(--text-muted)', fontSize: 10.5 }}>{t.role}</span>}
              {t.retries > 0 && (
                <span style={{ color: 'var(--warning)', fontSize: 10.5 }}>retry {t.retries}</span>
              )}
            </div>
          ))}
          {orchestration.tasks.length < orchestration.total && (
            <div style={{ ...rowStyle, color: 'var(--text-muted)' }}>
              <StatusGlyph kind="pending" />
              <span>{orchestration.total - orchestration.tasks.length} more queued</span>
            </div>
          )}
        </>
      )}

      {/* Step timeline — what actually happened, on a vertical rail. */}
      <SectionTitle label="Steps" count={groups.length} />
      {groups.length === 0 ? (
        <div style={{ color: 'var(--text-muted)' }}>No tool activity yet.</div>
      ) : (
        <div style={{ position: 'relative' }}>
          {/* the rail: connects step dots so the list reads as a timeline */}
          {groups.length > 1 && (
            <div style={{
              position: 'absolute', left: 6, top: 10, bottom: 10,
              width: 1, background: 'var(--border)',
            }} />
          )}
          {groups.map((g, i) => {
            const isLast = i === groups.length - 1;
            const live = isLast && (isRunning || awaiting !== null);
            // A failed call inside a step the run recovered from is an
            // annotation, not a verdict — ✗ is reserved for the step the run
            // actually died on (last step of a not-verified/failed run).
            const failedCalls = g.calls.filter(tc => tc.success === false).length;
            const runEndedHere = isLast && !isRunning && awaiting === null && verdictOk === false;
            const kind: Glyph = live
              ? (isRunning ? 'running' : 'pending')
              : runEndedHere ? 'failed' : 'done';
            return (
              <div key={`${g.step}-${i}`} style={{
                ...rowStyle,
                position: 'relative',
                paddingLeft: 0,
                background: live ? 'var(--bg-secondary)' : undefined,
                borderRadius: live ? 6 : undefined,
              }}>
                <span style={{
                  position: 'relative', zIndex: 1,
                  background: 'var(--code-bg)', borderRadius: '50%',
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <StatusGlyph kind={kind} />
                </span>
                <span style={{
                  color: 'var(--text-muted)', flexShrink: 0, fontSize: 10.5,
                  fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums',
                  minWidth: 14, textAlign: 'right',
                }}>
                  {i + 1}
                </span>
                {/* time gradient: past steps recede, the live one stays bright */}
                <span style={{
                  color: live ? 'var(--text-primary)' : 'var(--text-muted)',
                  flex: 1,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11.5,
                }}>
                  {stepSummary(g.calls)}
                </span>
                {failedCalls > 0 && (
                  <span style={{ color: 'var(--warning)', fontSize: 10.5, flexShrink: 0 }}>
                    {failedCalls} failed
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Evidence — adaptive: files for coding work, sources for research. */}
      {(mode !== 'research' && files.length > 0) && (
        <>
          <SectionTitle label="Files" count={files.length} />
          {files.map(([path, verbs]) => (
            <div key={path} style={rowStyle}>
              {onOpenFile ? (
                <button
                  type="button"
                  onClick={() => onOpenFile(path)}
                  title={path}
                  style={{
                    color: 'var(--text-primary)', flex: 1, textAlign: 'left',
                    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                    fontFamily: 'var(--font-mono)', fontSize: 11.5,
                    textDecoration: 'underline dotted', textUnderlineOffset: 3,
                    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                  }}
                >{shortPath(path)}</button>
              ) : (
                <span title={path} style={{
                  color: 'var(--text-primary)', flex: 1, wordBreak: 'break-all',
                  fontFamily: 'var(--font-mono)', fontSize: 11.5,
                }}>{shortPath(path)}</span>
              )}
              <span style={{ color: 'var(--text-muted)', fontSize: 10.5, flexShrink: 0 }}>
                {[...verbs].join(' · ')}
              </span>
            </div>
          ))}
        </>
      )}
      {(mode !== 'coding' && (sources.queries.length > 0 || sources.pages.length > 0)) && (
        <>
          <SectionTitle label="Sources" count={sources.queries.length + sources.pages.length} />
          {sources.queries.map(q => (
            <div key={`q-${q}`} style={rowStyle}>
              <span style={{ color: 'var(--text-muted)', flexShrink: 0, fontSize: 10.5 }}>search</span>
              <span style={{ color: 'var(--text-primary)', flex: 1 }}>{truncate(q, 80)}</span>
            </div>
          ))}
          {sources.pages.map(([key, { url, count }]) => (
            <div key={`u-${key}`} style={rowStyle}>
              <span style={{ color: 'var(--text-muted)', flexShrink: 0, fontSize: 10.5 }}>page</span>
              <span style={{
                color: 'var(--text-primary)', flex: 1, wordBreak: 'break-all',
                fontFamily: 'var(--font-mono)', fontSize: 11.5,
              }} title={url}>
                {truncate(key, 60)}
              </span>
              {count > 1 && (
                <span style={{ color: 'var(--text-muted)', fontSize: 10.5, flexShrink: 0 }}>
                  ×{count}
                </span>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  );
});

export default ProgressPane;
