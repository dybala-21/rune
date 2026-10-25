import { memo, useEffect, useMemo, useRef, type ReactNode } from 'react';
import { describeTrust, trustColors } from '../utils/trust';
import { RunEvidence } from './RunEvidence';
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
  argString,
  truncate,
  basename,
  hostOf,
  type ActivityMode,
} from '../utils/tooling';

interface ProgressPaneProps {
  toolCalls: ToolCall[];
  /** Computed once by WorkbenchPanel and shared with the tab bar. */
  mode: ActivityMode;
  isRunning: boolean;
  currentStep: StepInfo | null;
  trust?: TrustInfo | null;
  activitySummary: ActivitySummary | null;
  orchestration: OrchestrationState | null;
  /** Pending input takes priority over progress. */
  awaiting?: 'approval' | 'question' | null;
  onOpenFile?: (path: string) => void;
}

/** Shorten the label while keeping the full path in the tooltip. */
function shortPath(path: string): string {
  const parts = path.replace(/\/+$/, '').split('/').filter(Boolean);
  return parts.length > 2 ? parts.slice(-2).join('/') : path;
}

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

interface StepSource {
  kind: 'search' | 'page';
  label: string;
  url?: string;
  count: number;
}

interface StepGroup {
  step: number;
  /** Step numbers repeat across runs. */
  run: number;
  summary: string;
  failedCount: number;
  /** Sources in order of first use. */
  sources: StepSource[];
}

interface Derived {
  groups: StepGroup[];
  /** Paths and actions in order of first use. */
  files: Array<[string, string[]]>;
  /** Latest pending call, falling back to the latest call. */
  nowLabel: string | null;
}

function derive(toolCalls: ToolCall[]): Derived {
  const groups: Array<StepGroup & { labels: Set<string>; extra: number; srcKeys: Set<string> }> = [];
  const files = new Map<string, Set<string>>();
  let pendingLabel: string | null = null;
  for (const tc of toolCalls) {
    const label = callLabel(tc);
    if (tc.result === undefined) pendingLabel = label;

    const step = tc.step ?? 0;
    const run = tc.run ?? 0;
    let g = groups[groups.length - 1];
    // Step numbers restart each run; group by both run and step.
    if (!g || g.step !== step || g.run !== run) {
      g = { step, run, summary: '', failedCount: 0, labels: new Set(), extra: 0, sources: [], srcKeys: new Set() };
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
      if (q && !g.srcKeys.has(`s:${q}`)) {
        g.srcKeys.add(`s:${q}`);
        g.sources.push({ kind: 'search', label: q, count: 1 });
      }
    } else if (name === 'web.fetch' || isBrowserToolName(name)) {
      const url = argString(tc.args, 'url');
      if (url) {
        let key = url;
        try {
          const u = new URL(url);
          key = u.host + u.pathname;
        } catch { /* non-URL string: group by the raw value */ }
        const existing = g.sources.find(s => s.kind === 'page' && s.label === key);
        if (existing) existing.count += 1;
        else { g.srcKeys.add(`p:${key}`); g.sources.push({ kind: 'page', label: key, url, count: 1 }); }
      }
    }
  }
  const last = toolCalls[toolCalls.length - 1];
  return {
    groups: groups.map(g => ({
      step: g.step,
      run: g.run,
      failedCount: g.failedCount,
      summary: [...g.labels].join(' · ') + (g.extra > 0 ? ` +${g.extra}` : ''),
      sources: g.sources,
    })),
    files: [...files.entries()].map(([p, verbs]) => [p, [...verbs]]),
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

function SectionTitle({ label, count }: { label: string; count?: number }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'baseline',
      gap: 6,
      fontSize: 12,
      fontWeight: 550,
      color: 'var(--text-muted)',
      margin: '16px 0 6px',
    }}>
      <span>{label}</span>{' '}
      {count !== undefined && count > 0 && (
        <span style={{ fontFamily: 'var(--font-mono)' }}>({count})</span>
      )}
    </div>
  );
}
interface BandSpec {
  color: string;
  bg: string;
  glyph: ReactNode;
  title: string;
  titleMono?: boolean;
  note?: string;
  details?: string[];
  live?: boolean;
}

/** Show current activity or a result that needs attention. */
function StatusBand({ isRunning, awaiting, nowLabel, stepNumber, verdictOk, trust }: {
  isRunning: boolean;
  awaiting: 'approval' | 'question' | null;
  nowLabel: string | null;
  stepNumber: number | null;
  verdictOk: boolean | null;
  trust?: TrustInfo | null;
}) {
  const view = trust ? describeTrust(trust) : null;

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
      color: 'var(--accent)', bg: 'var(--accent-subtle)',
      glyph: <span className="spinner" style={{ width: 12, height: 12, flexShrink: 0 }} />,
      title: nowLabel ?? 'working…',
      titleMono: true,
      note: stepNumber !== null ? `step ${stepNumber}` : undefined,
      live: true,
    };
  } else if (view ? !view.showCard : verdictOk === null) {
    return null;
  } else if (trust && view) {
    const colors = trustColors(view.tone);
    spec = {
      color: colors.accent, bg: colors.background,
      glyph: <span>{view.glyph}</span>,
      title: view.title,
      note: view.note,
      details: trust.escalationHint ? [trust.escalationHint] : [],
    };
  } else {
    spec = {
      color: verdictOk ? 'var(--border)' : 'var(--danger)',
      bg: verdictOk ? 'var(--bg-secondary)' : 'var(--danger-subtle)',
      glyph: <span>{verdictOk ? '✓' : '✗'}</span>,
      title: verdictOk ? 'Completed' : 'Failed',
      note: verdictOk ? 'No verification status was reported.' : undefined,
    };
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
      minWidth: 0,
      overflowWrap: 'anywhere',
      padding: '9px 12px',
      borderRadius: 8,
      border: `1px solid ${spec.color}`,
      background: spec.bg,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 9, minWidth: 0 }}>
        <span aria-hidden="true" style={{ flexShrink: 0 }}>{spec.glyph}</span>
        {spec.live && (
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
            color: 'var(--accent)', background: 'var(--accent-subtle)',
            padding: '1px 5px', borderRadius: 4, flexShrink: 0,
            fontFamily: 'var(--font-mono)',
          }}>
            NOW
          </span>
        )}
        <span style={{
          color: 'var(--text-primary)', flex: 1, minWidth: 0,
          fontSize: 12.5,
          fontWeight: spec.titleMono ? undefined : 600,
          fontFamily: spec.titleMono ? 'var(--font-mono)' : undefined,
        }}>
          {spec.title}
        </span>
        {spec.live && spec.note && (
          <span style={{ color: 'var(--text-muted)', fontSize: 12, flexShrink: 0 }}>
            {spec.note}
          </span>
        )}
      </div>
      {!spec.live && spec.note && (
        <div style={{ color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.6 }}>
          {spec.note}
        </div>
      )}
      {spec.details?.map(d => (
        <div key={d} style={{ color: 'var(--text-primary)', fontSize: 12, paddingLeft: 21 }}>
          {d}
        </div>
      ))}
    </div>
  );
}

export const ProgressPane = memo(function ProgressPane({
  toolCalls, mode, isRunning, currentStep, trust, activitySummary, orchestration, awaiting = null, onOpenFile,
}: ProgressPaneProps) {
  const verdictOk = computeRunVerdict(trust, activitySummary);
  const { groups, files, nowLabel } = useMemo(() => derive(toolCalls), [toolCalls]);

  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = listRef.current;
    if (el && isRunning) el.scrollTop = el.scrollHeight;
  }, [toolCalls.length, isRunning]);

  if (toolCalls.length === 0 && !orchestration && !trust && !activitySummary && !isRunning && !awaiting) {
    return (
      <div style={{ flex: 1, padding: 14, color: 'var(--text-muted)', fontSize: 12.5 }}>
        Task progress will appear here when Rune starts working.
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
      minWidth: 0,
      minHeight: 0,
      overflow: 'auto',
      padding: '16px',
      fontSize: 13,
      lineHeight: 1.6,
    }}>
      <StatusBand
        isRunning={isRunning}
        awaiting={awaiting}
        nowLabel={nowLabel}
        stepNumber={currentStep?.stepNumber ?? null}
        verdictOk={verdictOk}
        trust={trust}
      />
      {!isRunning && !awaiting && trust && <RunEvidence trust={trust} />}

      {orchestration && (
        <>
          <SectionTitle label="Tasks" count={orchestration.total} />
          {orchestration.description && (
            <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>
              {truncate(orchestration.description, 120)}
            </div>
          )}
          <div style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 4 }}>
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
              {t.role && <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{t.role}</span>}
              {t.retries > 0 && (
                <span style={{ color: 'var(--warning)', fontSize: 12 }}>retry {t.retries}</span>
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

      <SectionTitle label="Steps" count={groups.length} />
      {groups.length === 0 ? (
        <div style={{ color: 'var(--text-muted)' }}>No tool activity yet.</div>
      ) : (
        <div style={{ position: 'relative' }}>
          {groups.length > 1 && (
            <div style={{
              position: 'absolute', left: 6, top: 10, bottom: 10,
              width: 1, background: 'var(--border)',
            }} />
          )}
          {groups.map((g, i) => {
            const isLast = i === groups.length - 1;
            const live = isLast && (isRunning || awaiting !== null);
            // Recovered tool failures do not make the whole step fail.
            const runEndedHere = isLast && !isRunning && awaiting === null && verdictOk === false;
            const kind: Glyph = live
              ? (isRunning ? 'running' : 'pending')
              : runEndedHere ? 'failed' : 'done';
            return (
              <div key={`${g.step}-${i}`} style={{
                position: 'relative',
                padding: live ? '2px 0 4px 8px' : undefined,
                background: live ? 'var(--accent-subtle)' : undefined,
                borderRadius: live ? 6 : undefined,
                borderLeft: live ? '2px solid var(--accent)' : undefined,
              }}>
                <div style={rowStyle}>
                  <span style={{
                    position: 'relative', zIndex: 1,
                    background: 'var(--code-bg)', borderRadius: '50%',
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <StatusGlyph kind={kind} />
                  </span>
                  <span style={{
                    color: 'var(--text-muted)', flexShrink: 0, fontSize: 12,
                    fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums',
                    minWidth: 14, textAlign: 'right',
                  }}>
                    {i + 1}
                  </span>
                  <span style={{
                    color: live ? 'var(--text-primary)' : 'var(--text-muted)',
                    flex: 1,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 12,
                  }}>
                    {g.summary}
                  </span>
                  {g.failedCount > 0 && (
                    <span style={{ color: 'var(--warning)', fontSize: 12, flexShrink: 0 }}>
                      {g.failedCount} failed
                    </span>
                  )}
                </div>
                {g.sources.length > 0 && (
                  <div style={{ paddingLeft: 30, display: 'flex', flexDirection: 'column', gap: 1, marginTop: 1 }}>
                    {g.sources.map(src => (
                      <div key={`${src.kind}:${src.label}`} style={{
                        display: 'flex', gap: 6, alignItems: 'baseline',
                        fontSize: 12, color: 'var(--text-muted)',
                      }}>
                        <span style={{ flexShrink: 0, opacity: 0.8 }}>
                          {src.kind === 'search' ? '⌕' : '↳'}
                        </span>
                        <span
                          title={src.url ?? src.label}
                          style={{
                            flex: 1, wordBreak: 'break-all',
                            fontFamily: src.kind === 'page' ? 'var(--font-mono)' : undefined,
                            color: 'var(--text-secondary)',
                          }}
                        >
                          {truncate(src.label, src.kind === 'page' ? 56 : 72)}
                        </span>
                        {src.count > 1 && (
                          <span style={{ flexShrink: 0, opacity: 0.7 }}>×{src.count}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {isRunning && currentStep && (
        <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 2 }}>
          step {currentStep.stepNumber} in progress
        </div>
      )}

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
                    fontFamily: 'var(--font-mono)', fontSize: 12,
                    textDecoration: 'underline dotted', textUnderlineOffset: 3,
                    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                  }}
                >{shortPath(path)}</button>
              ) : (
                <span title={path} style={{
                  color: 'var(--text-primary)', flex: 1, wordBreak: 'break-all',
                  fontFamily: 'var(--font-mono)', fontSize: 12,
                }}>{shortPath(path)}</span>
              )}
              <span style={{ color: 'var(--text-muted)', fontSize: 12, flexShrink: 0 }}>
                {verbs.join(' · ')}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
});

export default ProgressPane;
