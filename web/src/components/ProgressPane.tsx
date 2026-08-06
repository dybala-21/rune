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
  computeRunVerdict,
  passedChecks,
  argString,
  truncate,
  basename,
  hostOf,
  type ActivityMode,
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
  /** Computed once by WorkbenchPanel and shared with the tab bar. */
  mode: ActivityMode;
  isRunning: boolean;
  currentStep: StepInfo | null;
  trust?: TrustInfo | null;
  activitySummary: ActivitySummary | null;
  orchestration: OrchestrationState | null;
  /** The agent is blocked on the user — takes priority over every other band. */
  awaiting?: 'approval' | 'question' | null;
  onOpenFile?: (path: string) => void;
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
  switch (name) {
    case 'bash': {
      const cmd = argString(tc.args, 'command', 'cmd', 'script');
      return cmd ? `run ${truncate(cmd, 40)}` : 'run command';
    }
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
  summary: string;
  failedCount: number;
}

interface Derived {
  groups: StepGroup[];
  /** path → verbs used on it, in first-touch order */
  files: Array<[string, string[]]>;
  queries: string[];
  /** host+path → {sample url, hit count} — pagination collapses to ×N */
  pages: Array<[string, { url: string; count: number }]>;
  /** what the agent is doing right now (newest pending, else newest call) */
  nowLabel: string | null;
}

/** Everything the pane renders, derived in a single pass over the calls. */
function derive(toolCalls: ToolCall[]): Derived {
  const groups: Array<StepGroup & { labels: Set<string>; extra: number }> = [];
  const files = new Map<string, Set<string>>();
  const queries: string[] = [];
  const pages = new Map<string, { url: string; count: number }>();
  let pendingLabel: string | null = null;
  for (const tc of toolCalls) {
    const label = callLabel(tc);
    if (tc.result === undefined) pendingLabel = label;

    const step = tc.step ?? 0;
    let g = groups[groups.length - 1];
    if (!g || g.step !== step) {
      g = { step, summary: '', failedCount: 0, labels: new Set(), extra: 0 };
      groups.push(g);
    }
    if (!g.labels.has(label)) {
      if (g.labels.size < 2) g.labels.add(label);
      else g.extra++;
    }
    if (tc.success === false) g.failedCount++;

    const name = normalizeToolName(tc.toolName);
    if (name.startsWith('file.')) {
      const path = argString(tc.args, 'path', 'file_path', 'file', 'target');
      if (path) {
        if (!files.has(path)) files.set(path, new Set());
        files.get(path)!.add(name.replace('file.', ''));
      }
    } else if (name === 'web.search') {
      const q = argString(tc.args, 'query', 'q');
      if (q && !queries.includes(q)) queries.push(q);
    } else if (name === 'web.fetch' || isBrowserToolName(name)) {
      const url = argString(tc.args, 'url');
      if (url) {
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
  }
  const last = toolCalls[toolCalls.length - 1];
  return {
    groups: groups.map(g => ({
      step: g.step,
      failedCount: g.failedCount,
      summary: [...g.labels].join(' · ') + (g.extra > 0 ? ` +${g.extra}` : ''),
    })),
    files: [...files.entries()].map(([p, verbs]) => [p, [...verbs]]),
    queries,
    pages: [...pages.entries()],
    nowLabel: pendingLabel ?? (last ? callLabel(last) : null),
  };
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

const CAP_NOTE = 'stopped at tool budget';
const CAP_DETAIL =
  'The tool-round budget ran out — some planned steps never executed, so the answer may be incomplete.';

/** One band shape for every terminal state, so the layout can't drift. */
interface BandSpec {
  color: string;
  bg: string;
  glyph: ReactNode;
  title: string;
  titleMono?: boolean;
  note?: string;
  details?: string[];
  showEvidence?: boolean;
}

/**
 * The one-glance answer at the top of the pane. Blocked on the user: an amber
 * call to action. While running: what the agent is doing right now. Finished:
 * the run's verify-or-fail-honestly verdict, expandable to the actual
 * Evidence Gate output — the receipt, not just the claim.
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
  const capped = Boolean(trust?.budgetExhausted);
  const passes = passedChecks(trust);

  let spec: BandSpec | null = null;
  if (awaiting) {
    spec = {
      color: 'var(--warning)', bg: 'var(--warning-subtle, var(--bg-secondary))',
      glyph: <span style={{ color: 'var(--warning)' }}>◐</span>,
      title: awaiting === 'approval' ? 'Waiting for your approval' : 'Waiting for your answer',
      note: 'respond in the chat',
    };
  } else if (isRunning) {
    spec = {
      color: 'var(--border)', bg: 'var(--bg-secondary)',
      glyph: <span className="spinner" style={{ width: 12, height: 12, flexShrink: 0 }} />,
      title: nowLabel ?? 'working…',
      titleMono: true,
      note: stepNumber !== null ? `step ${stepNumber}` : undefined,
    };
  } else if (verdictOk === null) {
    return null;
  } else if (verdictOk && trust?.evidenceGate?.hasCheck) {
    spec = {
      color: 'var(--success)', bg: 'var(--success-subtle)',
      glyph: <span style={{ color: 'var(--success)' }}>✓</span>,
      title: 'Verified',
      note: passes > 0 ? `${passes} checks passed` : undefined,
      showEvidence: true,
    };
  } else if (verdictOk) {
    // Completion is not verification — a run that finished without any
    // Evidence Gate check gets a neutral band, amber when it was cut off
    // at the tool budget (the answer may omit steps that never ran).
    spec = {
      color: capped ? 'var(--warning)' : 'var(--border)',
      bg: capped ? 'var(--warning-subtle, var(--bg-secondary))' : 'var(--bg-secondary)',
      glyph: <span style={{ color: capped ? 'var(--warning)' : 'var(--text-muted)' }}>✓</span>,
      title: 'Completed',
      note: capped ? CAP_NOTE : 'no verification checks ran',
      details: capped ? [CAP_DETAIL] : undefined,
    };
  } else {
    const honest = Boolean(trust && !trust.verified);
    spec = {
      color: honest ? 'var(--warning)' : 'var(--danger)',
      bg: honest ? 'var(--warning-subtle, var(--danger-subtle))' : 'var(--danger-subtle)',
      glyph: <span style={{ color: honest ? 'var(--warning)' : 'var(--danger)' }}>✗</span>,
      title: honest ? 'Not verified — honest stop' : 'Failed',
      details: [
        ...(trust?.honestNote ? [trust.honestNote] : []),
        ...(capped ? [CAP_DETAIL] : []),
        ...(trust?.escalationHint ? [trust.escalationHint] : []),
      ],
      showEvidence: true,
    };
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 3,
      padding: '9px 12px',
      borderRadius: 8,
      border: `1px solid ${spec.color}`,
      background: spec.bg,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
        {spec.glyph}
        <span style={{
          color: 'var(--text-primary)', flex: 1, wordBreak: 'break-word',
          fontSize: 12.5,
          fontWeight: spec.titleMono ? undefined : 600,
          fontFamily: spec.titleMono ? 'var(--font-mono)' : undefined,
        }}>
          {spec.title}
        </span>
        {spec.note && (
          <span style={{ color: 'var(--text-muted)', fontSize: 11, flexShrink: 0 }}>
            {spec.note}
          </span>
        )}
      </div>
      {spec.details?.map(d => (
        <div key={d} style={{ color: 'var(--text-primary)', fontSize: 11.5, paddingLeft: 21 }}>
          {d}
        </div>
      ))}
      {spec.showEvidence && evidence && (
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
      )}
    </div>
  );
}

export const ProgressPane = memo(function ProgressPane({
  toolCalls, mode, isRunning, currentStep, trust, activitySummary, orchestration, awaiting = null, onOpenFile,
}: ProgressPaneProps) {
  const verdictOk = computeRunVerdict(trust, activitySummary);
  const { groups, files, queries, pages, nowLabel } = useMemo(() => derive(toolCalls), [toolCalls]);

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
                  {g.summary}
                </span>
                {g.failedCount > 0 && (
                  <span style={{ color: 'var(--warning)', fontSize: 10.5, flexShrink: 0 }}>
                    {g.failedCount} failed
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {isRunning && currentStep && (
        <div style={{ color: 'var(--text-muted)', fontSize: 10.5, marginTop: 2 }}>
          step {currentStep.stepNumber} in progress
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
                {verbs.join(' · ')}
              </span>
            </div>
          ))}
        </>
      )}
      {(mode !== 'coding' && (queries.length > 0 || pages.length > 0)) && (
        <>
          <SectionTitle label="Sources" count={queries.length + pages.length} />
          {queries.map(q => (
            <div key={`q-${q}`} style={rowStyle}>
              <span style={{ color: 'var(--text-muted)', flexShrink: 0, fontSize: 10.5 }}>search</span>
              <span style={{ color: 'var(--text-primary)', flex: 1 }}>{truncate(q, 80)}</span>
            </div>
          ))}
          {pages.map(([key, { url, count }]) => (
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
