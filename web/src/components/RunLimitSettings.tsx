import { useEffect, useRef, useState } from 'react';
import { patchConfig, type RunLimits } from '../api';

export function parseRunLimits(requests: string, cost: string): RunLimits {
  const modelRequests = Number(requests);
  if (!requests.trim() || !Number.isSafeInteger(modelRequests) || modelRequests < 0) {
    throw new Error('Enter a whole number of model requests, zero or greater.');
  }
  const costUsd = cost.trim() ? Number(cost) : null;
  if (costUsd !== null && (!Number.isFinite(costUsd) || costUsd < 0)) {
    throw new Error('Enter a cost of zero or greater, or leave it blank.');
  }
  return { modelRequests, costUsd };
}

export function RunLimitSettings({ limits, onSaved }: {
  limits: RunLimits;
  onSaved: () => Promise<void>;
}) {
  const [requests, setRequests] = useState(String(limits.modelRequests));
  const [cost, setCost] = useState(limits.costUsd === null ? '' : String(limits.costUsd));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const dirty = useRef(false);

  useEffect(() => {
    if (!dirty.current) {
      setRequests(String(limits.modelRequests));
      setCost(limits.costUsd === null ? '' : String(limits.costUsd));
    }
  }, [limits.modelRequests, limits.costUsd]);

  const edit = (value: string, field: 'requests' | 'cost') => {
    dirty.current = true;
    setSaved(false);
    setError('');
    if (field === 'requests') setRequests(value);
    else setCost(value);
  };

  const save = async () => {
    setSaving(true);
    setError('');
    setSaved(false);
    try {
      await patchConfig({ runLimits: parseRunLimits(requests, cost) });
      dirty.current = false;
      await onSaved();
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save run limits.');
    } finally {
      setSaving(false);
    }
  };

  const inputStyle = { width: '100%', boxSizing: 'border-box' as const, padding: '8px 10px',
    marginTop: 5, borderRadius: 6, background: 'var(--bg-secondary)',
    border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', fontSize: 12 };

  return (
    <section aria-label="Run limits" style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-subtle)' }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Run limits</div>
      <p style={{ margin: '0 0 12px', fontSize: 11, lineHeight: 1.5, color: 'var(--text-muted)' }}>
        Applies to new runs, including scheduled work. Retries, delegated work and background learning share the limits.
      </p>
      <label htmlFor="run-request-limit" style={{ display: 'block', fontSize: 12, marginBottom: 12 }}>
        Model requests per run
        <input id="run-request-limit" type="number" min="0" step="1" value={requests} disabled={saving}
          onChange={event => edit(event.target.value, 'requests')} style={inputStyle} />
      </label>
      <label htmlFor="run-cost-limit" style={{ display: 'block', fontSize: 12 }}>
        Estimated token cost limit (USD)
        <input id="run-cost-limit" type="number" min="0" step="any" placeholder="No cost limit"
          value={cost} disabled={saving} onChange={event => edit(event.target.value, 'cost')} style={inputStyle} />
      </label>
      <p style={{ margin: '8px 0 12px', fontSize: 11, lineHeight: 1.5, color: 'var(--text-muted)' }}>
        Leave cost blank for no cost limit. Zero blocks model requests. Rune stops further requests when
        the recorded estimate reaches the limit or cost cannot be determined. An in-flight request can
        exceed it. This is not an account spending cap; embedding and tool fees are excluded.
      </p>
      {error && <p role="alert" style={{ fontSize: 11, color: 'var(--danger)' }}>{error}</p>}
      {saved && <p role="status" style={{ fontSize: 11, color: 'var(--success)' }}>Saved for new runs.</p>}
      <button type="button" onClick={save} disabled={saving}
        style={{ padding: '7px 12px', borderRadius: 6, border: '1px solid var(--border-subtle)',
          background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 12 }}>
        {saving ? 'Saving…' : 'Save limits'}
      </button>
    </section>
  );
}
