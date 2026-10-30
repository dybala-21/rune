import { useEffect, useId, useState } from 'react';
import { patchConfig, setEnvVar, type ConfigInfo } from '../api';
import './DecisionRoutingSettings.css';

type Routing = NonNullable<ConfigInfo['decisionRouting']>;
const labels = { connected: 'Off', jev: 'Jev', microsoft: 'Microsoft Decision' };
const keyNames = { jev: 'TYPESAFE_API_KEY', microsoft: 'AZURE_DECISION_API_KEY' };

export function parseMicrosoftDecision(endpoint: string, deployment: string) {
  let url: URL;
  try { url = new URL(endpoint.trim()); } catch { throw new Error('Enter your Azure resource endpoint.'); }
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.port
    || !['.services.ai.azure.com', '.cognitiveservices.azure.com', '.openai.azure.com'].some(suffix => url.hostname.endsWith(suffix))
    || !['', '/providers/microsoft/v1/systemone'].includes(url.pathname.replace(/\/$/, ''))) {
    throw new Error('Use the Azure resource’s HTTPS endpoint or its System One URL.');
  }
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(deployment.trim())) {
    throw new Error('Enter the deployment name shown in Azure Foundry.');
  }
  return { endpoint: url.origin, deployment: deployment.trim() };
}

export function decisionStatusText(routing: Routing): string {
  const name = labels[routing.backend];
  return {
    disabled: 'Off. Your current model handles task routing.',
    ready: `${name} is connected. Uncertain decisions use your current model.`,
    unverified: `${name} is selected. The connection will be checked on your next request.`,
    local: 'Paused for local models. Task routing stays local.',
    automatic_model: 'Paused during automatic model selection. Choose a specific chat model to use this connection.',
    missing_key: `${name} needs an API key. Using your current model.`,
    missing_config: 'Microsoft Decision needs an endpoint and deployment name. Using your current model.',
    cooldown: `${name} is temporarily unavailable. Using your current model.`,
    auth_error: `${name} denied access. Check its key and deployment permissions. Using your current model.`,
  }[routing.status];
}

export function DecisionRoutingSettings({ routing, onSaved }: {
  routing: Routing;
  onSaved: () => Promise<void>;
}) {
  const id = useId();
  const [backend, setBackend] = useState(routing.backend);
  const [endpoint, setEndpoint] = useState(routing.microsoft.endpoint);
  const [deployment, setDeployment] = useState(routing.microsoft.deployment);
  const [key, setKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const connection = backend === 'connected' ? null : routing.providers[backend];
  const microsoftChanged = backend === 'microsoft'
    && (endpoint.trim() !== routing.microsoft.endpoint || deployment.trim() !== routing.microsoft.deployment);
  const dirty = backend !== routing.backend || microsoftChanged || !!key.trim();
  const needsKey = connection && !connection.hasKey && !key.trim();
  const needsConfig = backend === 'microsoft' && (!endpoint.trim() || !deployment.trim());

  useEffect(() => {
    setBackend(routing.backend);
    setEndpoint(routing.microsoft.endpoint);
    setDeployment(routing.microsoft.deployment);
  }, [routing.backend, routing.microsoft.endpoint, routing.microsoft.deployment]);

  const save = async () => {
    setSaving(true);
    setError('');
    setNotice('');
    let keySaved = false;
    try {
      const microsoft = backend === 'microsoft' ? parseMicrosoftDecision(endpoint, deployment) : undefined;
      if (backend !== 'connected' && key.trim()) {
        await setEnvVar(keyNames[backend], key.trim(), 'effective');
        keySaved = true;
        setKey('');
      }
      await patchConfig({ decisionRouting: { backend, ...(microsoft ? { microsoft } : {}) } });
      await onSaved();
      setNotice('Saved. Applies to new requests.');
    } catch (err) {
      if (keySaved) {
        try { await onSaved(); } catch (refreshError) { console.error('Could not refresh decision settings', refreshError); }
      }
      setError(`${keySaved ? 'Key saved. ' : ''}${err instanceof Error ? err.message : 'Could not save settings.'}`);
    } finally { setSaving(false); }
  };

  return (
    <section className="decision-settings" aria-label="Decision model">
      <fieldset disabled={saving} aria-describedby={`${id}-help`}>
        <legend>Decision model</legend>
        <p id={`${id}-help`} className="decision-help">Choose how Rune routes tasks. Your chat model stays the same.</p>
        <div className="decision-options">
          {(['connected', 'jev', 'microsoft'] as const).map(value => (
            <label key={value} className={`decision-option${backend === value ? ' selected' : ''}`}>
              <input type="radio" name={`${id}-backend`} value={value} checked={backend === value}
                onChange={() => { setBackend(value); setKey(''); setError(''); setNotice(''); }} />
              <span>{value === 'microsoft' ? 'Microsoft' : labels[value]}</span>
            </label>
          ))}
        </div>
        {backend !== 'connected' && connection && <div className="decision-connection">
          <p className="decision-help">{backend === 'jev' ? 'Jev by TypeSafe' : 'Microsoft Decision on Azure'} receives task context for routing. Billed separately.</p>
          {backend === 'microsoft' && <>
            <label htmlFor={`${id}-endpoint`}>Azure endpoint</label>
            <input id={`${id}-endpoint`} type="url" value={endpoint} onChange={event => { setEndpoint(event.target.value); setError(''); }}
              autoComplete="off" spellCheck={false} placeholder="https://your-resource.services.ai.azure.com" />
            <label htmlFor={`${id}-deployment`}>Deployment name</label>
            <input id={`${id}-deployment`} value={deployment} onChange={event => { setDeployment(event.target.value); setError(''); }}
              autoComplete="off" spellCheck={false} placeholder="Name shown in Azure Foundry" aria-describedby={`${id}-deployment-help`} />
            <p id={`${id}-deployment-help`} className="decision-hint">Use your deployment name, which may differ from the model name.</p>
          </>}
          <label htmlFor={`${id}-key`}>API key <span className="decision-key-state">{connection.hasKey ? 'Saved' : 'Required'}</span></label>
          <input id={`${id}-key`} type="password" autoComplete="off" spellCheck={false} value={key}
            onChange={event => { setKey(event.target.value); setError(''); }} disabled={connection.keyScope === 'process'}
            placeholder={connection.hasKey ? 'Leave blank to keep the saved key' : 'Paste your API key'} />
          <p className="decision-hint">{connection.keyScope === 'process'
            ? 'Managed by your launch environment. Update it there and restart Rune.'
            : connection.keyScope === 'project' ? 'Saved for this project. Other connections keep their own keys.'
              : 'Saved on this device. Other connections keep their own keys.'}</p>
        </div>}
      </fieldset>
      <p className="decision-status" role="status">{decisionStatusText(routing)}</p>
      {error && <p className="decision-error" role="alert">{error}</p>}
      <div className="decision-save">
        <button type="button" onClick={save} disabled={saving || !dirty || !!needsKey || needsConfig}>{saving ? 'Saving…' : 'Save'}</button>
        <span aria-live="polite">{dirty ? needsKey ? 'Add a key to enable this model.' : needsConfig ? 'Add the Azure connection details.' : 'Unsaved changes' : notice}</span>
      </div>
    </section>
  );
}
