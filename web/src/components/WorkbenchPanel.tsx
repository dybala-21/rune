import { useEffect, useRef, useState } from 'react';
import type { ActivitySummary, ToolCall } from '../types';
import { normalizeToolName, inferWorkPhase, type WorkPhase } from '../utils/tooling';
import { PixelWolf, type WolfState } from './PixelWolf';

/**
 * Coding workbench panel: a command log of file edits / bash calls with a
 * verify status. Opened/closed by App based on inferWorkPhase.
 */

interface WorkbenchPanelProps {
  toolCalls: ToolCall[];
  isRunning: boolean;
  activitySummary: ActivitySummary | null;
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

/** Renders file.edit's search/replace args — not a real file diff. */
function EditDiff({ search, replace }: { search: string; replace: string }) {
  const del = diffLines(search);
  const add = diffLines(replace);
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
      {del.lines.map((l, i) => row('−', 'var(--danger)', 'var(--danger-subtle)', l, i))}
      {del.truncated && row('−', 'var(--text-muted)', 'var(--danger-subtle)', '…', -1)}
      {add.lines.map((l, i) => row('+', 'var(--success)', 'var(--success-subtle)', l, i))}
      {add.truncated && row('+', 'var(--text-muted)', 'var(--success-subtle)', '…', -2)}
    </div>
  );
}

function CommandLine({ tc }: { tc: ToolCall }) {
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

  const search = name === 'file.edit' ? argString(tc.args, 'search') : null;
  const replace = name === 'file.edit' ? argString(tc.args, 'replace') ?? '' : null;

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
        <span style={{ color: 'var(--text-primary)', flex: 1 }}>{head}</span>
        {pending ? (
          <span className="spinner" style={{ width: 11, height: 11 }} />
        ) : (
          <span style={{ color: ok ? 'var(--success)' : 'var(--danger)', flexShrink: 0 }}>
            {ok ? '✓' : '✗'}
          </span>
        )}
      </div>
      {search !== null && replace !== null && (
        <EditDiff search={search} replace={replace} />
      )}
      {resultText && !(search !== null && ok) && (
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
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${s % 60}s`;
}

export function WorkbenchPanel({ toolCalls, isRunning, activitySummary, onClose }: WorkbenchPanelProps) {
  const phase = inferWorkPhase(toolCalls);
  const coding = toolCalls.filter(tc => CODING_TOOLS.has(normalizeToolName(tc.toolName)));

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
      width: 'clamp(340px, 40%, 600px)',
      flexShrink: 0,
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

      {/* Command log */}
      <div ref={logRef} style={{
        flex: 1,
        overflow: 'auto',
        padding: '14px',
        fontFamily: 'var(--font-mono)',
        fontSize: 12.5,
        lineHeight: 1.5,
      }}>
        {coding.length === 0 ? (
          <div style={{ color: 'var(--text-muted)' }}>Waiting for the first edit or command…</div>
        ) : (
          coding.map(tc => <CommandLine key={tc.id} tc={tc} />)
        )}
      </div>

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
        <span style={{ marginLeft: 'auto' }}>daemon · 127.0.0.1</span>
      </div>
    </aside>
  );
}

export default WorkbenchPanel;
