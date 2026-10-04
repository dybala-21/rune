import { useState } from 'react';
import { reconcileCronJob, type CronJobInfo, type RoutinePolicy } from '../api';

export function defaultRoutinePolicy(): RoutinePolicy {
  return { workspace: '', max_steps: 30, timeout_seconds: 120, token_budget: 50000,
    deadline: null, verification: [], input_paths: [], output_paths: [], notify: 'changes' };
}

const field: React.CSSProperties = { display: 'flex', flexDirection: 'column', gap: 5, fontSize: 12, color: 'var(--text-secondary)' };
const input: React.CSSProperties = { width: '100%', minWidth: 0, boxSizing: 'border-box', fontSize: 13, padding: 8, borderRadius: 6, border: '1px solid var(--border)', color: 'var(--text-primary)', background: 'var(--bg-secondary)' };

export function RoutineFields({ policy, onChange }: { policy: RoutinePolicy; onChange: (policy: RoutinePolicy) => void }) {
  const update = (patch: Partial<RoutinePolicy>) => onChange({ ...policy, ...patch });
  return <details>
    <summary style={{ cursor: 'pointer', fontSize: 12 }}>Scope and limits · {policy.timeout_seconds}s · {policy.token_budget.toLocaleString()} tokens</summary>
    <div style={{ display: 'grid', gap: 12, marginTop: 12 }}>
      <label style={field}>Working folder<input style={input} value={policy.workspace} onChange={e => update({ workspace: e.target.value })} placeholder="Rune's working folder if empty" /></label>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
        <label style={field}>Time limit (seconds)<input style={input} type="number" min="1" max="1800" value={policy.timeout_seconds} onChange={e => update({ timeout_seconds: Number(e.target.value) })} /></label>
        <label style={field}>Token limit<input style={input} type="number" min="1000" max="500000" value={policy.token_budget} onChange={e => update({ token_budget: Number(e.target.value) })} /></label>
        <label style={field}>Step limit<input style={input} type="number" min="1" max="200" value={policy.max_steps} onChange={e => update({ max_steps: Number(e.target.value) })} /></label>
      </div>
      <label style={field}>Stop after (date, time and time zone)<input style={input} value={policy.deadline ?? ''} onChange={e => update({ deadline: e.target.value || null })} placeholder="2026-12-31T18:00:00+09:00" /></label>
      <label style={field}>Input files to watch (one per line)<textarea style={input} value={(policy.input_paths ?? []).join('\n')} onChange={e => update({ input_paths: e.target.value.split('\n') })} placeholder="Paths relative to the working folder" /></label>
      <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: 11 }}>Skip completed work when these files and the outputs below are unchanged. Leave inputs empty for tasks that depend on time, email or live web data.</p>
      <label style={field}>Output files to check (one per line)<textarea style={input} value={(policy.output_paths ?? []).join('\n')} onChange={e => update({ output_paths: e.target.value.split('\n') })} placeholder="Missing or changed outputs trigger another run" /></label>
      <label style={field}>Verification commands (one per line)<textarea style={input} value={policy.verification.join('\n')} onChange={e => update({ verification: e.target.value.split('\n') })} placeholder="Optional checks for agent tasks; normal permissions apply" /></label>
      <label style={field}>Notify connected channel<select style={input} value={policy.notify} onChange={e => update({ notify: e.target.value as RoutinePolicy['notify'] })}>
        <option value="changes">When results change</option><option value="failures">When attention is needed</option><option value="always">After every run</option>
      </select></label>
      <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: 11 }}>Agent tasks use model tokens. Scheduling does not grant permission for protected actions. Every run keeps a result here.</p>
    </div>
  </details>;
}

export function RoutineHistory({ job, onRefresh }: { job: CronJobInfo; onRefresh: () => Promise<void> }) {
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const blocked = job.recentRuns?.find(run => run.blocked);
  const review = async () => {
    if (!blocked) return;
    setSaving(true); setError('');
    try {
      await reconcileCronJob(job.id, blocked.id, note);
      setNote('');
      await onRefresh();
    } catch (err) { setError(err instanceof Error ? err.message : 'Could not save review'); }
    finally { setSaving(false); }
  };
  return <section key={job.id} aria-label="Recent task runs" style={{ borderTop: '1px solid var(--border)', paddingTop: 12, fontSize: 12 }}>
    <strong>Recent runs</strong>
    {!job.recentRuns?.length && <p style={{ color: 'var(--text-muted)' }}>No runs yet.</p>}
    {job.recentRuns?.map(run => <details key={run.id} style={{ marginTop: 8 }}>
      <summary style={{ cursor: 'pointer' }}>{new Date(run.started_at * 1000).toLocaleString()} · {run.result?.status.split('_').join(' ') ?? 'Outcome pending'}</summary>
      <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontFamily: 'inherit', color: 'var(--text-secondary)' }}>{run.result?.output || run.result?.error || run.result?.note || 'The worker has not reported a final result.'}</pre>
    </details>)}
    {blocked && <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
      <p style={{ margin: 0 }}>Further runs are blocked until you check whether this run changed files or external systems.</p>
      {job.enabled ? <p style={{ margin: 0, color: 'var(--text-muted)' }}>Pause and save this task before recording your review.</p> : <>
        <label style={field}>What did you check?<textarea style={input} value={note} onChange={e => setNote(e.target.value)} /></label>
        <button style={input} disabled={saving || !note.trim()} onClick={review}>Record review and clear block</button>
        <span style={{ color: 'var(--text-muted)' }}>This leaves the task paused and does not mark its result verified.</span>
      </>}
    </div>}
    {error && <p role="alert" style={{ color: 'var(--danger)' }}>{error}</p>}
  </section>;
}
