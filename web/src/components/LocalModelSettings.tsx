import { useEffect, useRef, useState } from 'react';
import {
  patchConfig, setEnvVar, testEndpoint,
  type CompatibleEndpoint, type ConfigInfo, type EndpointTestResult,
} from '../api';

type State = NonNullable<ConfigInfo['localModels']>;
type Row = { name: string; baseUrl: string; models: string; toolMode: CompatibleEndpoint['toolMode']; free: string; key: string };

const inputStyle = {
  width: '100%', boxSizing: 'border-box' as const, padding: '7px 9px', borderRadius: 6,
  border: '1px solid var(--border-subtle)', background: 'var(--bg-secondary)', color: 'var(--text-primary)',
};
const field = { ...inputStyle, margin: '4px 0 8px' };
const muted = { color: 'var(--text-muted)', lineHeight: 1.5 };
const TOOLS: Record<EndpointTestResult['tools'], string> = {
  native: 'Tool calls work.', text: 'Reachable, but the model answered without a tool call.',
  unsupported: 'Tool calls are turned off on the server.', unknown: 'Tool support is unknown.',
};

const rowsFrom = (state: State): Row[] => Object.entries(state.endpoints).map(([name, entry]) => ({
  name, baseUrl: entry.baseUrl, models: entry.models.join('\n'), toolMode: entry.toolMode,
  free: entry.free == null ? '' : String(entry.free), key: '',
}));

export function LocalModelSettings({ state, onSaved }: { state: State; onSaved: () => Promise<void> }) {
  const [ollama, setOllama] = useState(state.ollama.baseUrl);
  const [rows, setRows] = useState(rowsFrom(state));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const [tests, setTests] = useState<Record<string, EndpointTestResult | 'running'>>({});
  const dirty = useRef(false);
  const change = () => { dirty.current = true; setSaved(false); };
  const edit = (index: number, patch: Partial<Row>) => { change(); setRows(rows.map((row, i) => i === index ? { ...row, ...patch } : row)); };

  useEffect(() => {
    if (!dirty.current) { setOllama(state.ollama.baseUrl); setRows(rowsFrom(state)); }
  }, [state]);

  const save = async () => {
    setSaving(true);
    setError('');
    setSaved(false);
    try {
      const endpoints: Record<string, CompatibleEndpoint> = {};
      for (const row of rows) {
        const name = row.name.trim();
        if (!/^[a-z0-9][a-z0-9-]{0,39}$/.test(name)) throw new Error('Use lowercase letters, digits and hyphens for endpoint names.');
        if (name in endpoints) throw new Error('Endpoint names must be unique.');
        const url = new URL(row.baseUrl.trim());
        if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Enter an http(s) base URL for each endpoint.');
        endpoints[name] = {
          baseUrl: row.baseUrl.trim(), toolMode: row.toolMode,
          models: row.models.split(/[\n,]/).map(model => model.trim()).filter(Boolean),
          ...(row.free ? { free: row.free === 'true' } : {}),
        };
      }
      await patchConfig({ endpoints });
      for (const row of rows) {
        const keyName = state.endpoints[row.name.trim()]?.keyName
          ?? `RUNE_ENDPOINT_${row.name.trim().toUpperCase().replace(/[^A-Z0-9]/g, '_')}_API_KEY`;
        if (row.key.trim()) await setEnvVar(keyName, row.key.trim(), 'effective');
      }
      if (ollama.trim() && ollama.trim() !== state.ollama.baseUrl) await setEnvVar(state.ollama.envKey, ollama.trim(), 'effective');
      dirty.current = false;
      setRows(rows.map(row => ({ ...row, key: '' })));
      await onSaved();
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save model servers.');
    } finally { setSaving(false); }
  };

  const runTest = async (name: string) => {
    setTests(current => ({ ...current, [name]: 'running' }));
    try {
      const result = await testEndpoint(name);
      setTests(current => ({ ...current, [name]: result }));
    } catch (err) {
      setTests(current => ({ ...current, [name]: { reachable: false, models: [], tools: 'unknown',
        detail: err instanceof Error ? err.message : 'The test failed.' } }));
    }
  };

  return <details style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-subtle)', fontSize: 12 }}>
    <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Open-weight model servers</summary>
    <p style={muted}>
      Connect Ollama on another machine, or any server with an OpenAI-compatible API: vLLM, SGLang, LM Studio,
      llama.cpp, or a hosted service. Models appear in the model picker as endpoint/model.
    </p>
    <fieldset disabled={saving} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
      <label htmlFor="ollama-base">Ollama server</label>
      <input id="ollama-base" value={ollama} placeholder="http://localhost:11434" style={field}
        disabled={state.ollama.scope === 'process'} onChange={event => { change(); setOllama(event.target.value); }} />
      {rows.map((row, index) => {
        const stored = state.endpoints[row.name.trim()];
        const test = tests[row.name.trim()];
        return <div key={index} style={{ margin: '10px 0 14px', paddingTop: 10, borderTop: '1px solid var(--border-subtle)' }}>
          <label htmlFor={`endpoint-name-${index}`}>Name</label>
          <input id={`endpoint-name-${index}`} value={row.name} placeholder="gpu" style={field}
            onChange={event => edit(index, { name: event.target.value })} />
          <label htmlFor={`endpoint-url-${index}`}>Base URL</label>
          <input id={`endpoint-url-${index}`} value={row.baseUrl} placeholder="http://192.168.0.20:8000/v1" style={field}
            onChange={event => edit(index, { baseUrl: event.target.value })} />
          <label htmlFor={`endpoint-key-${index}`}>{stored?.hasKey ? 'Replace API key (optional)' : 'API key (optional)'}</label>
          <input id={`endpoint-key-${index}`} type="password" autoComplete="off" spellCheck={false} value={row.key}
            placeholder={stored?.hasKey ? 'Key saved' : 'Leave empty if the server has no key'} style={field}
            disabled={stored?.keyScope === 'process'} onChange={event => edit(index, { key: event.target.value })} />
          <label htmlFor={`endpoint-models-${index}`}>Models (optional, one per line)</label>
          <textarea id={`endpoint-models-${index}`} value={row.models} rows={2} style={field}
            placeholder="Empty: use the models the server lists" onChange={event => edit(index, { models: event.target.value })} />
          <label htmlFor={`endpoint-tools-${index}`}>Tool mode</label>
          <select id={`endpoint-tools-${index}`} value={row.toolMode} style={field}
            onChange={event => edit(index, { toolMode: event.target.value as Row['toolMode'] })}>
            <option value="native">Native tool calls</option>
            <option value="guided">Guided JSON (for servers or models without tool calls)</option>
          </select>
          <label htmlFor={`endpoint-free-${index}`}>Billing</label>
          <select id={`endpoint-free-${index}`} value={row.free} style={field}
            onChange={event => edit(index, { free: event.target.value })}>
            <option value="">Automatic (private address = self-hosted)</option>
            <option value="true">Self-hosted, no per-token cost</option>
            <option value="false">Paid service, cost unknown</option>
          </select>
          {row.baseUrl.trim().startsWith('http:') && row.key.trim() &&
            <p style={muted}>This URL is not encrypted. The key will be sent in plain text.</p>}
          <button type="button" onClick={() => { change(); setRows(rows.filter((_, i) => i !== index)); }}
            aria-label={`Remove endpoint ${row.name || index + 1}`}>Remove</button>{' '}
          <button type="button" disabled={!stored || dirty.current || test === 'running'} onClick={() => runTest(row.name.trim())}
            title="Lists models and sends one short tool request">{test === 'running' ? 'Testing…' : 'Test'}</button>
          {test && test !== 'running' && <p role="status" style={{ ...muted, marginTop: 6 }}>
            {test.reachable ? `${test.models.length} model(s) listed. ${TOOLS[test.tools]}` : 'Not reachable.'} {test.detail}
          </p>}
        </div>;
      })}
      <button type="button" onClick={() => {
        change();
        setRows([...rows, { name: '', baseUrl: '', models: '', toolMode: 'native', free: '', key: '' }]);
      }}>Add endpoint</button>{' '}
      <button type="button" onClick={save} disabled={!dirty.current}>{saving ? 'Saving…' : 'Save'}</button>
    </fieldset>
    {error && <p role="alert" style={{ color: 'var(--danger)' }}>{error}</p>}
    {saved && <p role="status">Saved. Use Test to check the connection.</p>}
  </details>;
}
