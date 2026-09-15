import { lazy, memo, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ActivitySummary, FileChange, OrchestrationState, StepInfo, ToolCall, TrustInfo } from '../types';
import { normalizeToolName, isCodingToolName, argString, inferWorkPhase, inferActivityMode, computeRunVerdict, type WorkPhase } from '../utils/tooling';
import { RuneMark, type MarkState } from './RuneMark';
import { checkSummary, describeTrust, trustColors } from '../utils/trust';
import { fetchWorkspaceDiff, readWorkspaceFile } from '../api';
import { HighlightedCode } from './Code';
import { Markdown } from './Markdown';
import { langFromPath } from '../utils/highlight';
// Terminal pulls in xterm (~330 KB). Load it only when the tab is opened.
const TerminalPane = lazy(() => import('./TerminalPane').then(m => ({ default: m.TerminalPane })));
import { ProgressPane } from './ProgressPane';
import { DiffText, FileChangesPane } from './FileChangesPane';
import { desktopAttention, hasFileEdits, preferredWorkbenchTab } from '../utils/workbench';
import { ComputerPane } from './ComputerPane';

/** Saved code changes, execution progress, and live workspace tools. */

interface WorkbenchPanelProps {
  sessionId: string;
  toolCalls: ToolCall[];
  fileChanges?: FileChange[];
  isRunning: boolean;
  activitySummary: ActivitySummary | null;
  /** The run's verify-or-fail verdict — the real Evidence Gate result, same
      data the chat trust card uses. Preferred over the activity heuristic. */
  trust?: TrustInfo | null;
  currentStep?: StepInfo | null;
  orchestration?: OrchestrationState | null;
  /** The agent is blocked on the user (approval or question) — surfaced in the
      status band and as an attention dot on the Progress tab. */
  awaiting?: 'approval' | 'question' | null;
  connected?: boolean;
  historical?: boolean;
  onClose: () => void;
}

const PHASE_LABEL: Record<WorkPhase, string> = {
  analyzing: 'analyzing',
  implementing: 'implementing',
  verifying: 'verifying',
};

function fileVerb(name: string): string {
  switch (name) {
    case 'file.read': return 'read';
    case 'file.write': return 'write';
    case 'file.edit': return 'edit';
    case 'file.delete': return 'delete';
    default: return name;
  }
}

const DIFF_MAX_LINES = 6;

function diffLines(text: string): { lines: string[]; truncated: boolean } {
  const all = text.split('\n');
  return { lines: all.slice(0, DIFF_MAX_LINES), truncated: all.length > DIFF_MAX_LINES };
}

/**
 * Renders file.edit's search/replace args — not a real file diff. An empty
 * search is a pure insertion (show only + lines); an empty replace is a pure
 * deletion (show only − lines).
 */
function EditDiff({ search, replace }: { search: string; replace: string }) {
  const del = search ? diffLines(search) : null;
  const add = replace ? diffLines(replace) : null;
  if (!del && !add) return null;
  const row = (mark: string, color: string, bg: string, line: string, i: number) => (
    <div key={`${mark}${i}`} style={{
      display: 'flex',
      background: bg,
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-all',
    }}>
      <span style={{ width: 16, textAlign: 'center', color, flexShrink: 0 }}>{mark}</span>
      <span style={{ color, flex: 1, padding: '1px 6px 1px 0' }}>{line}</span>
    </div>
  );
  return (
    <div style={{
      margin: '4px 0 4px 20px',
      border: '1px solid var(--border-subtle)',
      borderRadius: 6,
      overflow: 'hidden',
      fontSize: 11.5,
    }}>
      {del?.lines.map((l, i) => row('−', 'var(--danger)', 'var(--danger-subtle)', l, i))}
      {del?.truncated && row('−', 'var(--text-muted)', 'var(--danger-subtle)', '…', -1)}
      {add?.lines.map((l, i) => row('+', 'var(--success)', 'var(--success-subtle)', l, i))}
      {add?.truncated && row('+', 'var(--text-muted)', 'var(--success-subtle)', '…', -2)}
    </div>
  );
}

// memo: the panel's 1s elapsed-timer re-render shouldn't re-diff unchanged commands.
const CommandLine = memo(function CommandLine({ tc, onOpenFile }: { tc: ToolCall; onOpenFile?: (path: string) => void }) {
  const name = normalizeToolName(tc.toolName);
  const pending = tc.result === undefined;
  const ok = tc.success !== false;
  const isBash = name === 'bash';
  const target = isBash
    ? argString(tc.args, 'command', 'cmd', 'script')
    : argString(tc.args, 'path', 'file_path', 'file', 'target');

  const head = isBash
    ? target ?? '(command)'
    : `${fileVerb(name)}  ${target ?? ''}`.trim();

  const isEdit = name === 'file.edit';
  const search = isEdit ? (argString(tc.args, 'search') ?? '') : null;
  const replace = isEdit ? (argString(tc.args, 'replace') ?? '') : null;

  const resultText =
    typeof tc.result === 'string' && tc.result.trim()
      ? tc.result.length > 240 ? tc.result.slice(0, 240) + '…' : tc.result
      : null;

  return (
    <div style={{ padding: '3px 0' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        <span style={{ color: isBash ? 'var(--accent)' : 'var(--text-muted)', flexShrink: 0 }}>
          {isBash ? '❯' : '✎'}
        </span>
        {!isBash && target && onOpenFile ? (
          <button
            type="button"
            onClick={() => onOpenFile(target)}
            title="Open file"
            style={{
              color: 'var(--text-primary)', flex: 1, textAlign: 'left',
              background: 'none', border: 'none', padding: 0, cursor: 'pointer',
              fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline dotted',
              textUnderlineOffset: 3,
            }}
          >{head}</button>
        ) : (
          <span style={{ color: 'var(--text-primary)', flex: 1 }}>{head}</span>
        )}
        {pending ? (
          <span className="spinner" style={{ width: 11, height: 11 }} />
        ) : (
          <span style={{ color: ok ? 'var(--success)' : 'var(--danger)', flexShrink: 0 }}>
            {ok ? '✓' : '✗'}
          </span>
        )}
      </div>
      {isEdit && (
        <EditDiff search={search ?? ''} replace={replace ?? ''} />
      )}
      {resultText && !(isEdit && ok) && (
        <div style={{
          color: 'var(--text-muted)',
          paddingLeft: 20,
          marginTop: 1,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          maxHeight: 84,
          overflow: 'hidden',
        }}>
          {resultText}
        </div>
      )}
    </div>
  );
});

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${s % 60}s`;
}

// Leaf component owns the 1s tick so the timer re-renders one text node, not
// the whole panel (which would re-derive phase/mode/coding every second).
function Elapsed({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <span style={{
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
      color: 'var(--warning)',
      fontVariantNumeric: 'tabular-nums',
    }}>
      {formatElapsed(now - startedAt)}
    </span>
  );
}

type BenchTab = 'progress' | 'activity' | 'diff' | 'file' | 'terminal' | 'computer';

const TABS: Array<[BenchTab, string]> = [
  ['progress', 'Progress'],
  ['computer', 'Computer'],
  ['activity', 'Activity'],
  ['diff', 'Diff'],
  ['file', 'File'],
  ['terminal', 'Terminal'],
];

const NO_CHANGES: FileChange[] = [];

export function WorkbenchPanel({ sessionId, toolCalls, fileChanges = NO_CHANGES, isRunning, activitySummary, trust, currentStep = null, orchestration = null, awaiting = null, connected = true, historical = false, onClose }: WorkbenchPanelProps) {
  // Share the verdict with chat and status indicators; null means no verdict yet.
  const verdictOk = computeRunVerdict(trust, activitySummary);
  const phase = useMemo(() => inferWorkPhase(toolCalls), [toolCalls]);
  const desktopWait = desktopAttention(toolCalls, isRunning && !historical);
  const waitingLabel = awaiting === 'approval' ? 'Waiting for approval'
    : awaiting === 'question' ? 'Waiting for your answer'
    : desktopWait === 'connection' ? 'Waiting for app access'
    : desktopWait === 'action' ? 'Review app action' : null;
  const mode = useMemo(() => inferActivityMode(toolCalls), [toolCalls]);
  const coding = useMemo(
    () => toolCalls.filter(tc => isCodingToolName(normalizeToolName(tc.toolName))),
    [toolCalls],
  );

  const preferredTab = preferredWorkbenchTab(toolCalls, fileChanges);
  const [tab, setTab] = useState<BenchTab>(preferredTab);
  const [workspaceDiff, setWorkspaceDiff] = useState(false);
  const [diffText, setDiffText] = useState('');
  const [diffLoading, setDiffLoading] = useState(false);
  const [filePath, setFilePath] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [fileError, setFileError] = useState('');
  const [fileLoaded, setFileLoaded] = useState(false);
  const [filePreview, setFilePreview] = useState(true);

  const loadDiff = useCallback(() => {
    if (historical) return;
    setWorkspaceDiff(true);
    setDiffLoading(true);
    fetchWorkspaceDiff()
      .then(r => setDiffText(r.diff))
      .catch(e => setDiffText(`diff unavailable: ${e instanceof Error ? e.message : e}`))
      .finally(() => setDiffLoading(false));
  }, [historical]);

  const openFile = useCallback((path: string) => {
    if (historical) return;
    setTab('file');
    setFilePath(path);
    setFileError('');
    setFileContent('');
    setFileLoaded(false);
    readWorkspaceFile(path)
      .then(r => { setFileContent(r.content); setFileLoaded(true); })
      .catch(e => setFileError(e instanceof Error ? e.message : String(e)));
  }, [historical]);

  // Follow opens code changes, but a tab chosen by the user stays put.
  const [follow, setFollow] = useState(true);
  const activityCount = toolCalls.length;
  useEffect(() => {
    // Don't yank the user out of an interactive terminal.
    if (follow && isRunning && activityCount > 0) {
      setTab(t => (t === 'terminal' ? t : preferredTab));
    }
  }, [follow, isRunning, activityCount, preferredTab]);

  // Diff is a coding surface; when the run turns out to be research work the
  // tab disappears, so a user parked there falls back to Progress.
  const showDiffTab = fileChanges.length > 0 || hasFileEdits(toolCalls) || !historical && mode !== 'research';
  useEffect(() => {
    if (!showDiffTab) setTab(t => (t === 'diff' ? 'progress' : t));
  }, [showDiffTab]);

  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [coding.length, isRunning]);

  const startedAt = toolCalls.length > 0 ? toolCalls[0].timestamp : null;

  let petState: MarkState = 'idle';
  if (waitingLabel) petState = 'warning';
  else if (isRunning) petState = phase === 'verifying' ? 'thinking' : 'working';
  else if (verdictOk !== null) petState = verdictOk ? 'passed' : 'failed';

  const trustView = trust ? describeTrust(trust) : null;
  const verdictTitle = trustView?.title ?? (verdictOk ? 'Completed' : 'Failed');
  const verdictColors = trustColors(trustView?.tone ?? (verdictOk ? 'neutral' : 'danger'));
  // Choose the footer label and color from the same state.
  const foot = awaiting || waitingLabel
    ? { text: waitingLabel || 'waiting for you', color: 'var(--warning)' }
    : isRunning
      ? { text: `${PHASE_LABEL[phase]}…`, color: 'var(--warning)' }
      : verdictOk === null
        ? { text: 'ready', color: 'var(--text-muted)' }
        : { text: verdictTitle.toLowerCase(), color: trustView?.tone === 'neutral' ? 'var(--text-muted)' : verdictColors.accent };

  return (
    <aside style={{
      width: '100%',
      height: '100%',
      borderLeft: '1px solid var(--border)',
      background: 'var(--code-bg)',
      display: 'flex',
      flexDirection: 'column',
      minWidth: 0,
    }}>
      {/* Header */}
      <div className="workbench-heading" style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '9px 14px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
      }}>
        <RuneMark state={petState} size={18} title={`RUNE workbench (${petState})`} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>
          Workbench
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}>
          {waitingLabel || (isRunning ? PHASE_LABEL[phase] : trust?.completionStatus === 'cancelled' ? 'stopped' : toolCalls.length || trust ? 'finished' : 'ready')}
        </span>
        {isRunning && startedAt !== null && <Elapsed startedAt={startedAt} />}
        <button
          onClick={onClose}
          title="Collapse workbench (⌘J)"
          style={{
            marginLeft: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-muted)',
            background: 'none',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '3px 9px',
            cursor: 'pointer',
          }}
        >
          Collapse
        </button>
      </div>

      {/* Tabs */}
      <div className="workbench-tabs" style={{
        display: 'flex', gap: 2, padding: '6px 10px 0',
        borderBottom: '1px solid var(--border)',
      }}>
        {TABS.filter(([key]) => historical
          ? key === 'progress' || key === 'activity' || key === 'diff' && showDiffTab
          : key !== 'diff' || showDiffTab).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => { setTab(key); setFollow(false); }}
            aria-pressed={tab === key}
            style={{
              padding: '5px 12px', fontSize: 12, cursor: 'pointer',
              color: tab === key ? 'var(--text-primary)' : 'var(--text-muted)',
              background: tab === key ? 'var(--code-bg)' : 'none',
              border: '1px solid', borderColor: tab === key ? 'var(--border)' : 'transparent',
              borderBottom: 'none', borderRadius: '8px 8px 0 0',
            }}
          >
            {label}
            {key === 'progress' && awaiting && tab !== 'progress' && (
              <span aria-label="needs your attention" style={{
                display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
                background: 'var(--warning)', marginLeft: 5, verticalAlign: 'middle',
              }} />
            )}
          </button>
        ))}
        {tab === 'diff' && !historical ? (
          <button
            type="button"
            onClick={() => workspaceDiff ? setWorkspaceDiff(false) : loadDiff()}
            title={workspaceDiff ? 'Show saved task changes' : 'Inspect current workspace Git diff'}
            style={{
              marginLeft: 'auto', background: 'none', border: 'none',
              color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer', paddingBottom: 4,
            }}
          >
            {diffLoading ? 'Loading…' : workspaceDiff ? 'Task changes' : 'Workspace diff'}
          </button>
        ) : !historical && (
          <button
            type="button"
            onClick={() => setFollow(f => !f)}
            title="Follow code changes and tool activity"
            aria-pressed={follow}
            style={{
              marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
              background: 'none', border: 'none', cursor: 'pointer',
              color: follow ? 'var(--accent)' : 'var(--text-muted)',
              fontSize: 11, paddingBottom: 4, fontFamily: 'var(--font-mono)',
            }}
          >
            Follow
            <span style={{
              width: 24, height: 14, borderRadius: 99, position: 'relative',
              background: follow ? 'var(--accent)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border)', transition: 'background 0.15s',
            }}>
              <span style={{
                position: 'absolute', top: 1, width: 10, height: 10, borderRadius: '50%',
                background: follow ? '#0A1319' : 'var(--text-muted)',
                left: follow ? 12 : 2, transition: 'left 0.15s',
              }} />
            </span>
          </button>
        )}
      </div>

      {/* Progress view — checklist, step timeline, evidence, verdict */}
      {tab === 'progress' && (
        <ProgressPane
          toolCalls={toolCalls}
          mode={mode}
          isRunning={isRunning}
          currentStep={currentStep}
          trust={trust}
          activitySummary={activitySummary}
          orchestration={orchestration}
          awaiting={awaiting}
          onOpenFile={historical ? undefined : openFile}
        />
      )}

      {/* Diff view */}
      {!historical && tab === 'computer' && <ComputerPane key={sessionId} sessionId={sessionId} />}
      {tab === 'diff' && !workspaceDiff && <FileChangesPane changes={fileChanges} toolCalls={toolCalls} historical={historical} />}
      {tab === 'diff' && workspaceDiff && !historical && (
        <div style={{
          flex: 1, overflow: 'auto', padding: 14,
          fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6,
        }}>
          {diffText ? <DiffText text={diffText.replace(/^```diff\n|\n```$/g, '')} /> : (
            <div style={{ color: 'var(--text-muted)' }}>{diffLoading ? 'Loading diff…' : 'No diff yet.'}</div>
          )}
        </div>
      )}

      {/* Terminal — mounted only when selected so no PTY opens otherwise */}
      {!historical && tab === 'terminal' && (
        <Suspense fallback={<div className="wb-loading">Loading terminal…</div>}>
          <TerminalPane />
        </Suspense>
      )}

      {/* File view */}
      {tab === 'file' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <form
            onSubmit={e => { e.preventDefault(); if (filePath.trim()) openFile(filePath.trim()); }}
            style={{ display: 'flex', gap: 6, padding: '8px 12px', borderBottom: '1px solid var(--border-subtle, var(--border))' }}
          >
            <input
              value={filePath}
              onChange={e => setFilePath(e.target.value)}
              placeholder="relative/path/in/workspace"
              style={{
                flex: 1, background: 'var(--bg-primary)', color: 'var(--text-primary)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                padding: '5px 8px', fontSize: 11.5, fontFamily: 'var(--font-mono)',
              }}
            />
            <button type="submit" style={{
              background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              padding: '5px 10px', fontSize: 11.5, cursor: 'pointer',
            }}>Open</button>
            {langFromPath(filePath) === 'markdown' && fileContent && (
              <button type="button" onClick={() => setFilePreview(p => !p)} style={{
                background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                padding: '5px 10px', fontSize: 11.5, cursor: 'pointer',
              }}>{filePreview ? 'Raw' : 'Preview'}</button>
            )}
          </form>
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            {fileError
              ? <div style={{ padding: 14, color: 'var(--danger)', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{fileError}</div>
              : fileContent
                ? (langFromPath(filePath) === 'markdown' && filePreview
                    ? <div style={{ padding: '16px 18px', fontSize: 14, lineHeight: 1.7, color: 'var(--text-primary)' }}><Markdown content={fileContent} /></div>
                    : <HighlightedCode code={fileContent} lang={langFromPath(filePath)} lineNumbers />)
                : fileLoaded
                  ? <div style={{ padding: 14, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>(empty file)</div>
                  : <div style={{ padding: 14, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>Open a file from the Activity tab or enter a path.</div>}
          </div>
        </div>
      )}

      {/* Command log */}
      {tab === 'activity' && (
      <div ref={logRef} style={{
        flex: 1,
        overflow: 'auto',
        padding: '14px',
        fontFamily: 'var(--font-mono)',
        fontSize: 12.5,
        lineHeight: 1.5,
      }}>
        {!isRunning && (verdictOk !== null) && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            margin: '0 0 10px', padding: '8px 11px', borderRadius: 8,
            border: `1px solid ${verdictColors.accent}`,
            background: verdictColors.background,
            fontSize: 12,
          }}>
            <span aria-hidden="true">{trustView?.glyph ?? (verdictOk ? '✓' : '⚠')}</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {verdictTitle}
            </span>
            <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {trust?.evidenceGate?.hasCheck
                ? checkSummary(trust)
                : activitySummary && activitySummary.filesWritten > 0
                  ? `${activitySummary.filesWritten} edited`
                  : ''}
            </span>
          </div>
        )}
        {coding.length === 0 ? (
          <div style={{ color: 'var(--text-muted)' }}>Waiting for the first edit or command…</div>
        ) : (
          coding.map(tc => <CommandLine key={tc.id} tc={tc} onOpenFile={historical ? undefined : openFile} />)
        )}
      </div>
      )}

      {/* Status footer */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '9px 14px',
        borderTop: '1px solid var(--border)',
        background: 'var(--bg-secondary)',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--text-muted)',
      }}>
        <span style={{ color: foot.color }}>{foot.text}</span>
        {activitySummary && (
          <span>
            {activitySummary.filesWritten > 0 && `${activitySummary.filesWritten} edited  `}
            {activitySummary.bashExecutions > 0 && `${activitySummary.bashExecutions} ran`}
          </span>
        )}
        <span style={{ marginLeft: 'auto', color: connected ? undefined : 'var(--danger)' }}>
          {connected ? 'daemon · 127.0.0.1' : 'daemon · offline'}
        </span>
      </div>
    </aside>
  );
}

export default WorkbenchPanel;
