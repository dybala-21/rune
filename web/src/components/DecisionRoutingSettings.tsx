import { useEffect, useState } from 'react';
import { patchConfig, setEnvVar, type ConfigInfo } from '../api';

type Routing = NonNullable<ConfigInfo['decisionRouting']>;

export function DecisionRoutingSettings({ routing, onSaved }: {
  routing: Routing;
  onSaved: () => Promise<void>;
}) {
  const [backend, setBackend] = useState(routing.backend);
  const [key, setKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const { hasKey, keyScope } = routing;

  useEffect(() => { setBackend(routing.backend); }, [routing.backend]);

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      if (backend === 'jev' && key.trim()) {
        await setEnvVar('TYPESAFE_API_KEY', key.trim(), 'effective');
        setKey('');
      }
      await patchConfig({ decisionRouting: { backend } });
      await onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save routing settings.');
    } finally {
      setSaving(false);
    }
  };

  const status = {
    disabled: 'Using your current model.',
    ready: 'Jev enabled. Uncertain decisions use your current model.',
    unverified: 'Jev enabled. The connection will be checked on your next request.',
    local: 'Using your local model. Nothing is sent to Jev.',
    automatic_model: 'Automatic model selection uses the current connection to preserve your existing cost policy. Jev is not called in this mode.',
    missing_key: 'Jev key missing. Using your current model.',
    cooldown: 'Jev temporarily unavailable. Using your current model.',
    auth_error: 'TypeSafe denied access. Replace your key to retry. Using your current model.',
  }[routing.status];

  return (
    <section aria-label="Task routing" style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-subtle)' }}>
      <label htmlFor="decision-backend" style={{ display: 'block', fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
        Task routing
      </label>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 10px' }}>
        Choose how Rune identifies task needs and input or output files. All features work with your current connection.
      </p>
      <select id="decision-backend" value={backend} disabled={saving}
        onChange={(event) => { setBackend(event.target.value as Routing['backend']); setKey(''); setError(''); }}
        style={{ width: '100%', padding: '8px 10px', borderRadius: 6, background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', fontSize: 12 }}>
        <option value="connected">Use current connection</option>
        <option value="jev">Jev · optional accelerator</option>
      </select>
      {backend === 'jev' && (
        <div style={{ marginTop: 10 }}>
          <p style={{ fontSize: 11, lineHeight: 1.5, color: 'var(--text-secondary)', margin: '0 0 8px' }}>
            Experimental. Sends your request and a short excerpt of the previous request to TypeSafe for routing.
            Billed separately. Local models stay local.
          </p>
          <p style={{ fontSize: 11, lineHeight: 1.5, color: 'var(--text-muted)', margin: '0 0 8px' }}>
            {keyScope === 'process'
              ? 'The active key comes from the launch environment. Update it there and restart Rune.'
              : keyScope === 'project'
                ? 'The project key takes precedence. Replacing it updates this project only.'
                : 'Keys are saved in your user settings on this device.'}
          </p>
          <label htmlFor="decision-key" style={{ display: 'block', fontSize: 11, marginBottom: 4 }}>
            {hasKey ? 'Replace TypeSafe key (optional)' : 'TypeSafe API key'}
          </label>
          <input id="decision-key" type="password" autoComplete="off" spellCheck={false}
            value={key} onChange={(event) => setKey(event.target.value)} disabled={saving || keyScope === 'process'}
            placeholder={hasKey ? 'Key saved on this device' : 'Paste your key'}
            style={{ boxSizing: 'border-box', width: '100%', padding: '8px 10px', borderRadius: 6,
              border: '1px solid var(--border-subtle)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
          <a href="https://console.typesafe.ai" target="_blank" rel="noreferrer"
            style={{ display: 'inline-block', marginTop: 6, fontSize: 11, color: 'var(--accent)' }}>TypeSafe account</a>
        </div>
      )}
      <p aria-live="polite" style={{ fontSize: 11, lineHeight: 1.5, color: 'var(--text-muted)', margin: '10px 0' }}>{status}</p>
      {backend !== routing.backend && <p style={{ fontSize: 11, color: 'var(--text-secondary)', margin: '0 0 10px' }}>
        Unsaved change. Takes effect for new requests after saving.
      </p>}
      {error && <p role="alert" style={{ fontSize: 11, color: 'var(--danger)', lineHeight: 1.5 }}>{error}</p>}
      <button onClick={save} disabled={saving || (backend === 'jev' && !hasKey && !key.trim()) || (backend === routing.backend && !key.trim())}
        style={{ padding: '6px 12px', border: '1px solid var(--border-subtle)', borderRadius: 6,
          background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 11 }}>
        {saving ? 'Saving…' : 'Save'}
      </button>
    </section>
  );
}
