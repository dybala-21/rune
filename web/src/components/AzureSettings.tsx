import { useEffect, useRef, useState } from 'react';
import { patchConfig, setEnvVar, type AzurePricing, type ConfigInfo } from '../api';

type Connection = NonNullable<ConfigInfo['azure']>;
const resource = (url: string) => url.trim().replace(/\/$/, '').replace(/\/openai\/v1$/, '');
const pricesFor = (connection: Connection): AzurePricing['deployments'] =>
  connection.pricing?.endpoint === resource(connection.endpoint) ? connection.pricing.deployments : {};
const inputStyle = {
  width: '100%', boxSizing: 'border-box' as const, padding: '7px 9px', borderRadius: 6,
  border: '1px solid var(--border-subtle)', background: 'var(--bg-secondary)', color: 'var(--text-primary)',
};

export function AzureSettings({ connection, onSaved }: {
  connection: Connection; onSaved: () => Promise<void>;
}) {
  const [endpoint, setEndpoint] = useState(connection.endpoint);
  const [key, setKey] = useState('');
  const [rows, setRows] = useState(Object.entries(connection.deployments));
  const [pricing, setPricing] = useState(pricesFor(connection));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const dirty = useRef(false);
  const change = () => { dirty.current = true; setSaved(false); };

  useEffect(() => {
    if (!dirty.current) {
      setEndpoint(connection.endpoint);
      setRows(Object.entries(connection.deployments));
      setPricing(pricesFor(connection));
    }
  }, [connection]);

  const save = async () => {
    setSaving(true);
    setError('');
    setSaved(false);
    try {
      const url = new URL(endpoint.trim());
      if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash
          || !['', '/openai/v1'].includes(url.pathname.replace(/\/$/, ''))) {
        throw new Error('Enter the HTTPS endpoint from Azure Foundry.');
      }
      const deployments: Record<string, string> = {};
      for (const [name, model] of rows) {
        if (![name, model].every(value => /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(value.trim()))) {
          throw new Error('Enter a deployment name and model ID for each row.');
        }
        if (Object.prototype.hasOwnProperty.call(deployments, name.trim())) throw new Error('Deployment names must be unique.');
        deployments[name.trim()] = model.trim();
      }
      const selected = Object.fromEntries(Object.entries(pricing).filter(([name]) => name in deployments));
      await patchConfig({ azureDeployments: deployments,
        azurePricing: Object.keys(selected).length ? { endpoint: resource(endpoint), deployments: selected } : null });
      if (endpoint.trim() !== connection.endpoint) {
        await setEnvVar(connection.endpointKey, endpoint.trim(), 'effective');
      }
      if (key.trim()) {
        await setEnvVar(connection.keyName, key.trim(), 'effective');
        setKey('');
      }
      dirty.current = false;
      await onSaved();
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save Azure settings.');
    } finally { setSaving(false); }
  };

  return <details style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-subtle)', fontSize: 12 }}>
    <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Azure OpenAI</summary>
    <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
      Register deployments from your Azure resource. Saving settings does not create a deployment or call a model.
    </p>
    <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
      For cost estimates, select the deployment type shown in Foundry. Rates cover GPT-6 Astra, Sol and Luna.
      Leave it unknown if unsure; Rune will not assume a rate.
    </p>
    <fieldset disabled={saving} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
      <label htmlFor="azure-endpoint">Endpoint</label>
      <input id="azure-endpoint" value={endpoint} placeholder="https://your-resource.openai.azure.com/"
        disabled={connection.endpointScope === 'process'} style={{ ...inputStyle, margin: '5px 0 10px' }}
        onChange={event => { change(); setEndpoint(event.target.value); setPricing({}); }} />
      <label htmlFor="azure-key">{connection.hasKey ? 'Replace API key (optional)' : 'API key'}</label>
      <input id="azure-key" type="password" autoComplete="off" spellCheck={false} value={key}
        placeholder={connection.hasKey ? 'Key saved' : 'Paste your Azure key'}
        disabled={connection.keyScope === 'process'} style={{ ...inputStyle, margin: '5px 0 10px' }}
        onChange={event => { change(); setKey(event.target.value); }} />
      {(connection.keyScope === 'process' || connection.endpointScope === 'process') &&
        <p style={{ color: 'var(--text-muted)' }}>Launch environment values must be changed there, then Rune restarted.</p>}
      {rows.map(([name, model], index) => <div key={index} style={{ marginBottom: 12 }}>
        <label htmlFor={`azure-deployment-${index}`}>Deployment name</label>
        <input id={`azure-deployment-${index}`} value={name} style={{ ...inputStyle, margin: '4px 0 6px' }}
          onChange={event => { change(); setRows(rows.map((row, i) => i === index ? [event.target.value, model] : row)); }} />
        <label htmlFor={`azure-model-${index}`}>Model ID</label>
        <input id={`azure-model-${index}`} value={model} list="azure-model-suggestions" style={{ ...inputStyle, marginTop: 4 }}
          placeholder="gpt-6-sol"
          onChange={event => { change(); setRows(rows.map((row, i) => i === index ? [name, event.target.value] : row)); }} />
        <label htmlFor={`azure-pricing-${index}`} style={{ display: 'block', marginTop: 8 }}>Deployment pricing</label>
        <select id={`azure-pricing-${index}`} value={pricing[name.trim()] ?? ''} style={{ ...inputStyle, marginTop: 4 }}
          onChange={event => {
            change();
            const next = { ...pricing };
            if (event.target.value) next[name.trim()] = event.target.value as 'global' | 'us' | 'eu';
            else delete next[name.trim()];
            setPricing(next);
          }}>
          <option value="">Unknown / other deployment type</option>
          <option value="global">Global Standard</option>
          <option value="us">Data Zone Standard — US</option>
          <option value="eu">Data Zone Standard — EU</option>
        </select>
        <button type="button" onClick={() => { change(); setRows(rows.filter((_, i) => i !== index)); }}
          style={{ marginTop: 5 }} aria-label={`Remove deployment ${name || index + 1}`}>Remove</button>
      </div>)}
      <datalist id="azure-model-suggestions">
        {connection.suggestedModels.map(model => <option key={model} value={model} />)}
      </datalist>
      <button type="button" onClick={() => { change(); setRows([...rows, ['', '']]); }}>Add deployment</button>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Use the deployment name shown in Foundry. Model ID controls reasoning options. Changes apply to new runs.
      </p>
      <button type="button" onClick={save} disabled={!dirty.current || (!connection.hasKey && !key.trim())}>
        {saving ? 'Saving…' : 'Save'}
      </button>
    </fieldset>
    {error && <p role="alert" style={{ color: 'var(--danger)' }}>{error}</p>}
    {saved && <p role="status">Settings saved. Connection has not been tested.</p>}
  </details>;
}
