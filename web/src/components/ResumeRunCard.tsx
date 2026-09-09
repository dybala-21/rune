import { useRef, useState } from 'react';
import { resumeRun } from '../api';
import type { RunSnapshot } from '../utils/runSnapshot';

export function ResumeRunCard({ run, connected, onResumed }: {
  run: RunSnapshot;
  connected: boolean;
  onResumed: (sessionId: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const sending = useRef(false);
  const supported = run.recoveryVersion === 1;
  const resume = async () => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError('');
    try {
      const result = await resumeRun(run.runId);
      onResumed(result.sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      sending.current = false;
      setBusy(false);
    }
  };
  return (
    <div style={{ padding: '10px 20px', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ flex: 1, color: 'var(--text-secondary)' }}>
          {supported ? 'Check saved work and continue the interrupted task.' : 'This older execution has no tool recovery record.'}
        </span>
        {supported && <button type="button" disabled={busy || !connected} onClick={resume}
          style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid var(--border)',
            background: 'var(--bg-surface)', color: 'var(--text-primary)', cursor: 'pointer' }}>
          {busy ? 'Checking…' : 'Resume task'}
        </button>}
      </div>
      {error && <div role="alert" style={{ marginTop: 6, color: 'var(--danger)' }}>{error}</div>}
    </div>
  );
}
