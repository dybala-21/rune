import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { ActivitySummary, ToolCall } from '../types';
import { normalizeToolName, inferWorkPhase, type WorkPhase } from '../utils/tooling';
import { PixelWolf, type WolfState } from './PixelWolf';
import { fetchWorkspaceDiff, readWorkspaceFile } from '../api';
import { TerminalPane } from './TerminalPane';

/**
 * Coding workbench panel: a command log of file edits / bash calls with a
 * verify status. Opened/closed by App based on inferWorkPhase.
 */

interface WorkbenchPanelProps {
  toolCalls: ToolCall[];
  isRunning: boolean;
  activitySummary: ActivitySummary | null;
  connected?: boolean;
  onClose: () => void;
}

const PHASE_LABEL: Record<WorkPhase, string> = {
  analyzing: 'analyzing',
  implementing: 'implementing',
  verifying: 'verifying',
};

const CODING_TOOLS = new Set(['file.read', 'file.write', 'file.edit', 'file.delete', 'bash']);

function fileVerb(name: string): string {
  switch (name) {
    case 'file.read': return 'read';
    case 'file.write': return 'write';
    case 'file.edit': return 'edit';
    case 'file.delete': return 'delete';
    default: return name;
  }
}

function argString(args: Record<string, unknown>, ...keys: string[]): string | null {
  for (const k of keys) {
    const v = args[k];
    if (typeof v === 'string' && v.trim()) return v;
  }
  return null;
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

type BenchTab = 'activity' | 'diff' | 'file' | 'terminal';

export function WorkbenchPanel({ toolCalls, isRunning, activitySummary, connected = true, onClose }: WorkbenchPanelProps) {
  const phase = inferWorkPhase(toolCalls);
  const coding = toolCalls.filter(tc => CODING_TOOLS.has(normalizeToolName(tc.toolName)));

  const [tab, setTab] = useState<BenchTab>('activity');
  const [diffText, setDiffText] = useState('');
  const [diffLoading, setDiffLoading] = useState(false);
  const [filePath, setFilePath] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [fileError, setFileError] = useState('');
  const [fileLoaded, setFileLoaded] = useState(false);

  const loadDiff = useCallback(() => {
    setDiffLoading(true);
    fetchWorkspaceDiff()
      .then(r => setDiffText(r.diff))
      .catch(e => setDiffText(`diff unavailable: ${e instanceof Error ? e.message : e}`))
      .finally(() => setDiffLoading(false));
  }, []);

  const openFile = useCallback((path: string) => {
    setTab('file');
    setFilePath(path);
    setFileError('');
    setFileContent('');
    setFileLoaded(false);
    readWorkspaceFile(path)
      .then(r => { setFileContent(r.content); setFileLoaded(true); })
      .catch(e => setFileError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => { if (tab === 'diff') loadDiff(); }, [tab, loadDiff]);

  // Follow: while the agent works, snap back to the live Activity feed when a
  // new tool call lands, so a user parked on Diff/File isn't left behind.
  const [follow, setFollow] = useState(true);
  const codingCount = coding.length;
  useEffect(() => {
    // Don't yank the user out of an interactive terminal.
    if (follow && isRunning && codingCount > 0) {
      setTab(t => (t === 'terminal' ? t : 'activity'));
    }
  }, [follow, isRunning, codingCount]);

  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [coding.length, isRunning]);

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isRunning) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [isRunning]);
  const startedAt = coding.length > 0 ? coding[0].timestamp : null;
  const elapsed = isRunning && startedAt ? formatElapsed(now - startedAt) : null;

  let petState: WolfState = 'idle';
  if (isRunning) petState = phase === 'verifying' ? 'thinking' : 'working';
  else if (activitySummary) petState = activitySummary.success ? 'passed' : 'failed';

  const footText = isRunning
    ? `${PHASE_LABEL[phase]}…`
    : activitySummary
      ? activitySummary.success ? 'verified' : 'not verified'
      : 'ready';
  const footColor = isRunning
    ? 'var(--warning)'
    : activitySummary
      ? activitySummary.success ? 'var(--success)' : 'var(--danger)'
      : 'var(--text-muted)';

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
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '9px 14px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
      }}>
        <PixelWolf state={petState} px={1.5} title={`RUNE workbench (${petState})`} />
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
          {PHASE_LABEL[phase]}
        </span>
        {elapsed && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--warning)',
            fontVariantNumeric: 'tabular-nums',
          }}>
            {elapsed}
          </span>
        )}
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
      <div style={{
        display: 'flex', gap: 2, padding: '6px 10px 0',
        borderBottom: '1px solid var(--border)',
      }}>
        {([['activity', 'Activity'], ['diff', 'Diff'], ['file', 'File'], ['terminal', 'Terminal']] as const).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            style={{
              padding: '5px 12px', fontSize: 12, cursor: 'pointer',
              color: tab === key ? 'var(--text-primary)' : 'var(--text-muted)',
              background: tab === key ? 'var(--code-bg)' : 'none',
              border: '1px solid', borderColor: tab === key ? 'var(--border)' : 'transparent',
              borderBottom: 'none', borderRadius: '8px 8px 0 0',
            }}
          >
            {label}
          </button>
        ))}
        {tab === 'diff' ? (
          <button
            type="button"
            onClick={loadDiff}
            title="Refresh diff"
            style={{
              marginLeft: 'auto', background: 'none', border: 'none',
              color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer', paddingBottom: 4,
            }}
          >
            {diffLoading ? 'loading…' : '↻ refresh'}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setFollow(f => !f)}
            title="Follow the agent — auto-switch to Activity as it works"
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

      {/* Diff view */}
      {tab === 'diff' && (
        <div style={{
          flex: 1, overflow: 'auto', padding: 14,
          fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6,
        }}>
          {diffText ? diffText.replace(/^```diff\n|\n```$/g, '').split('\n').map((l, i) => (
            <div key={i} style={{
              whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              color: l.startsWith('+') ? 'var(--success)'
                : l.startsWith('-') ? 'var(--danger)'
                : l.startsWith('@@') ? 'var(--accent)'
                : 'var(--text-muted)',
            }}>{l || '\u00A0'}</div>
          )) : (
            <div style={{ color: 'var(--text-muted)' }}>{diffLoading ? 'Loading diff…' : 'No diff yet.'}</div>
          )}
        </div>
      )}

      {/* Terminal — mounted only when selected so no PTY opens otherwise */}
      {tab === 'terminal' && <TerminalPane />}

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
          </form>
          <div style={{
            flex: 1, overflow: 'auto', padding: 14,
            fontFamily: 'var(--font-mono)', fontSize: 11.5, lineHeight: 1.6,
            whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: 'var(--text-primary)',
          }}>
            {fileError
              ? <span style={{ color: 'var(--danger)' }}>{fileError}</span>
              : fileContent
                ? fileContent
                : fileLoaded
                  ? <span style={{ color: 'var(--text-muted)' }}>(empty file)</span>
                  : <span style={{ color: 'var(--text-muted)' }}>Open a file from the Activity tab or enter a path.</span>}
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
        {/* Evidence Gate verdict — RUNE's honest-completion signal, shown
            once the run settles. */}
        {!isRunning && activitySummary && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            margin: '0 0 10px', padding: '8px 11px', borderRadius: 8,
            border: `1px solid ${activitySummary.success ? 'var(--success)' : 'var(--danger)'}`,
            background: activitySummary.success ? 'var(--success-subtle)' : 'var(--danger-subtle)',
            fontSize: 12,
          }}>
            <span aria-hidden="true">{activitySummary.success ? '✓' : '⚠'}</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              Evidence Gate — {activitySummary.success ? 'verified' : 'not verified'}
            </span>
            <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {activitySummary.filesWritten > 0 ? `${activitySummary.filesWritten} edited` : ''}
            </span>
          </div>
        )}
        {coding.length === 0 ? (
          <div style={{ color: 'var(--text-muted)' }}>Waiting for the first edit or command…</div>
        ) : (
          coding.map(tc => <CommandLine key={tc.id} tc={tc} onOpenFile={openFile} />)
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
        <span style={{ color: footColor }}>{footText}</span>
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
