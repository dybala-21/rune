import type { FileChange, ToolCall } from '../types';
import { argString, normalizeToolName } from '../utils/tooling';

export function DiffText({ text }: { text: string }) {
  return <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, lineHeight: 1.7, overflowX: 'auto' }}>
    {text.split('\n').map((line, index) => {
      const header = line.startsWith('+++') || line.startsWith('---');
      const added = !header && line.startsWith('+');
      const removed = !header && line.startsWith('-');
      const hunk = line.startsWith('@@');
      return <div key={index} style={{ whiteSpace: 'pre', minWidth: 'max-content', padding: '0 12px',
        background: added ? 'var(--success-subtle)' : removed ? 'var(--danger-subtle)' : hunk ? 'var(--accent-subtle)' : undefined,
        color: added ? 'var(--success)' : removed ? 'var(--danger)' : hunk ? 'var(--accent)' : 'var(--text-secondary)',
      }}>{line || '\u00a0'}</div>;
    })}
  </div>;
}

export function FileChangesPane({ changes, toolCalls, historical }: {
  changes: FileChange[]; toolCalls: ToolCall[]; historical: boolean;
}) {
  const proposals = changes.length ? [] : toolCalls.filter(call =>
    ['file.write', 'file.edit', 'file.delete'].includes(normalizeToolName(call.toolName)));
  return <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: 12 }}>
    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
      {changes.length ? 'Saved changes from this task' : 'Requested changes · actual diff was not recorded'}
    </div>
    {changes.map(change => <section key={change.id} style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', marginBottom: 12 }}>
      <div style={{ padding: '9px 12px', background: 'var(--bg-secondary)', fontSize: 12, overflowWrap: 'anywhere' }}>
        <strong>{change.path}</strong> <span style={{ color: 'var(--text-muted)' }}>· {change.kind}</span>
      </div>
      {change.patch && <DiffText text={change.patch} />}
      {change.notice && <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12 }}>{change.notice}</div>}
    </section>)}
    {proposals.map(call => {
      const name = normalizeToolName(call.toolName);
      const search = argString(call.args, 'search') ?? '';
      const replacement = argString(call.args, name === 'file.write' ? 'content' : 'replace') ?? '';
      const preview = [...(search ? search.split('\n').map(line => '-' + line) : []),
        ...(replacement ? replacement.split('\n').map(line => '+' + line) : [])].join('\n');
      return <section key={call.id} style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', marginBottom: 12 }}>
        <div style={{ padding: '9px 12px', background: 'var(--bg-secondary)', fontSize: 12, overflowWrap: 'anywhere' }}>
          <strong>{argString(call.args, 'path', 'file_path') ?? call.toolName}</strong>
          <span style={{ color: 'var(--text-muted)' }}> · {call.success === false ? 'Failed request' : historical ? 'Recorded request' : call.result === undefined ? 'Proposed change' : 'Requested change'}</span>
        </div>
        {preview ? <DiffText text={preview.slice(0, 32000)} /> : <div style={{ padding: 12, fontSize: 12 }}>No text preview was recorded.</div>}
      </section>;
    })}
    {!changes.length && !proposals.length && <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
      No saved file changes for this task.{!historical && ' Use Workspace diff to inspect current Git changes.'}
    </div>}
  </div>;
}
