import { useEffect, useRef, useState } from 'react';
import { patchConfig, setEnvVar, type AzurePricing, type ConfigInfo } from '../api';
import { fetchAzureModels, testAzureConnection, type AzureCheckResult } from '../azureApi';
import './AzureSettings.css';

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
  const [checking, setChecking] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, AzureCheckResult>>({});
  const [catalog, setCatalog] = useState<string[]>([]);
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const revision = useRef(0);
  const lastConnection = useRef(connection);
  const dirty = useRef(false);
  const change = () => { dirty.current = true; setSaved(false); setChecks({}); revision.current += 1; };
  const changeConnection = () => { change(); setCatalog([]); setCatalogLoaded(false); setSelectedModel(''); };
  const busy = saving || checking !== null;

  useEffect(() => () => { revision.current += 1; }, []);

  useEffect(() => {
    const changed = JSON.stringify(lastConnection.current) !== JSON.stringify(connection);
    lastConnection.current = connection;
    if (!changed) return;
    setChecks({});
    setCatalog([]);
    setCatalogLoaded(false);
    setSelectedModel('');
    revision.current += 1;
    if (!dirty.current) {
      setEndpoint(connection.endpoint);
      setRows(Object.entries(connection.deployments));
      setPricing(pricesFor(connection));
    }
  }, [connection]);

  const check = async (id: string, deployment?: string, model?: string) => {
    const current = revision.current;
    setChecking(id);
    setChecks(previous => { const next = { ...previous }; delete next[id]; return next; });
    const draft = { endpoint: endpoint.trim(), apiKey: key.trim() || undefined };
    try {
      const result = id === 'catalog' ? await fetchAzureModels(draft) : await testAzureConnection(draft, deployment, model);
      if (current !== revision.current) return;
      setChecks(previous => ({ ...previous, [id]: result }));
      if (id === 'catalog') {
        setCatalog(result.ok ? result.models ?? [] : []);
        setCatalogLoaded(result.ok);
        setSelectedModel('');
      }
    } catch (err) {
      if (current === revision.current) {
        setChecks(previous => ({ ...previous, [id]: { ok: false, detail: err instanceof Error ? err.message : 'Could not check Azure.' } }));
        if (id === 'catalog') { setCatalog([]); setCatalogLoaded(false); }
      }
    } finally { setChecking(null); }
  };

  const checkStatus = (id: string) => checks[id] && <p role={checks[id].ok ? 'status' : 'alert'}
    style={{ color: checks[id].ok ? 'var(--text-muted)' : 'var(--danger)', lineHeight: 1.5 }}>
    {checks[id].detail}{checks[id].latencyMs !== undefined && ` (${checks[id].latencyMs} ms)`}
  </p>;

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

  return <details className="azure-settings" style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-subtle)', fontSize: 12 }}>
    <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Azure Foundry</summary>
    <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
      Connect one Azure resource, then register its OpenAI, Kimi, or DeepSeek deployments.
      Fetch the model catalog, then enter the deployment names from Foundry.
    </p>
    <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
      For cost estimates, select the deployment type shown in Foundry. Rates cover GPT-6 Astra, Sol and Luna.
      Leave it unknown if unsure; Rune will not assume a rate.
    </p>
    <fieldset disabled={busy} aria-busy={busy} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
      <label htmlFor="azure-endpoint">Endpoint</label>
      <input id="azure-endpoint" value={endpoint} placeholder="https://your-resource.services.ai.azure.com/openai/v1"
        disabled={connection.endpointScope === 'process'} style={{ ...inputStyle, margin: '5px 0 10px' }}
        onChange={event => { changeConnection(); setEndpoint(event.target.value); setPricing({}); }} />
      <label htmlFor="azure-key">{connection.hasKey ? 'Replace API key (optional)' : 'API key'}</label>
      <input id="azure-key" type="password" autoComplete="off" spellCheck={false} value={key}
        aria-describedby="azure-key-help"
        placeholder={connection.hasKey ? 'Key saved' : 'Paste your Azure key'}
        disabled={connection.keyScope === 'process'} style={{ ...inputStyle, margin: '5px 0 10px' }}
        onChange={event => { changeConnection(); setKey(event.target.value); }} />
      <p id="azure-key-help" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        All deployments below share this endpoint and API key. Save the key once; leave this field blank to keep it.
      </p>
      {(connection.keyScope === 'process' || connection.endpointScope === 'process') &&
        <p style={{ color: 'var(--text-muted)' }}>Launch environment values must be changed there, then Rune restarted.</p>}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button type="button" disabled={!endpoint.trim() || (!connection.hasKey && !key.trim())}
          onClick={() => check('catalog')}>
          {checking === 'catalog' ? 'Fetching…' : 'Fetch models'}
        </button>
        <button type="button" disabled={!endpoint.trim() || (!connection.hasKey && !key.trim())}
          onClick={() => check('connection')}>
          {checking === 'connection' ? 'Testing…' : 'Test connection'}
        </button>
      </div>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        These checks use the values above without saving and do not generate tokens.
        The catalog lists model offerings, not your deployed models. A valid key does not confirm a deployment exists.
      </p>
      {checkStatus('connection')}
      {checkStatus('catalog')}
      {catalogLoaded && <div style={{ marginBottom: 12 }}>
        <label htmlFor="azure-catalog">Model catalog ({catalog.length})</label>
        <select id="azure-catalog" value={selectedModel} style={{ ...inputStyle, margin: '5px 0' }}
          onChange={event => setSelectedModel(event.target.value)}>
          <option value="">{catalog.length ? 'Choose a model' : 'No models returned'}</option>
          {catalog.map(model => <option key={model} value={model}>{model}</option>)}
        </select>
        <button type="button" disabled={!selectedModel} onClick={() => {
          change(); setRows([...rows, ['', selectedModel]]); setSelectedModel('');
        }}>Add selected model</button>
      </div>}
      {rows.length > 0 && <p id="azure-deployment-test-help" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Test deployment sends one short request (up to 256 output tokens); Azure usage charges may apply.
      </p>}
      {rows.map(([name, model], index) => <div className="azure-deployment" key={index}>
        <label htmlFor={`azure-deployment-${index}`}>Deployment name</label>
        <input id={`azure-deployment-${index}`} value={name} placeholder="Exact deployment name from Foundry"
          style={{ ...inputStyle, margin: '4px 0 6px' }}
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
        <button type="button" style={{ margin: '5px 0 0 8px' }}
          disabled={!name.trim() || !model.trim() || !endpoint.trim() || (!connection.hasKey && !key.trim())}
          onClick={() => check(`deployment-${index}`, name.trim(), model.trim())}
          aria-describedby="azure-deployment-test-help"
          aria-label={`Test deployment ${name || index + 1}`}>
          {checking === `deployment-${index}` ? 'Testing…' : 'Test deployment'}
        </button>
        {checkStatus(`deployment-${index}`)}
      </div>)}
      <datalist id="azure-model-suggestions">
        {[...new Set([...connection.suggestedModels, ...catalog])].map(model => <option key={model} value={model} />)}
      </datalist>
      <button type="button" onClick={() => { change(); setRows([...rows, ['', '']]); }}>Add deployment</button>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Deployment name must match Foundry exactly. Model ID identifies the underlying model and its capabilities.
        After saving, choose the Azure deployment in the model picker. Changes apply to new runs.
      </p>
      <button type="button" onClick={save} disabled={!dirty.current || (!connection.hasKey && !key.trim())}>
        {saving ? 'Saving…' : 'Save'}
      </button>
    </fieldset>
    {error && <p role="alert" style={{ color: 'var(--danger)' }}>{error}</p>}
    {saved && <p role="status">Settings saved. Select the deployment in the model picker.</p>}
  </details>;
}
