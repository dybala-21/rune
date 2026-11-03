import { useMemo, useState } from 'react';
import type { ConfigInfo } from '../api';
import type { AzureCheckResult } from '../azureApi';
import type { AzureConnection } from './azureForm';
import { useAzureSettings } from './useAzureSettings';
import './AzureSettings.css';

function CheckStatus({ result }: { result?: AzureCheckResult }) {
  if (!result) return null;
  const neutral = result.state === 'cancelled' || result.state === 'incomplete';
  return <p className={`azure-check ${result.ok ? 'is-success' : neutral ? '' : 'is-error'}`}
    role={result.ok || neutral ? 'status' : 'alert'}>
    {result.detail}{result.latencyMs !== undefined && <span className="azure-timing"> · {result.latencyMs} ms</span>}
  </p>;
}

export function AzureSettings({ connection, activeModel, onSaved }: {
  connection: AzureConnection; activeModel?: ConfigInfo['activeModel'] | null; onSaved: () => Promise<void>;
}) {
  const form = useAzureSettings(connection, onSaved);
  const [open, setOpen] = useState(!connection.hasKey || !Object.keys(connection.deployments).length);
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState('');
  const matches = useMemo(() => (form.catalog ?? []).filter(model => model.toLowerCase().includes(query.trim().toLowerCase())), [form.catalog, query]);
  const picked = matches.includes(selected) ? selected : '';
  const unavailable = form.saving || form.conflict;
  const fieldError = (id: string) => form.errors[id] && <p id={`${id}-error`} tabIndex={-1} className="azure-field-error">{form.errors[id]}</p>;
  const fieldProps = (id: string) => ({ id, 'aria-invalid': !!form.errors[id],
    'aria-describedby': form.errors[id] ? `${id}-error` : undefined });

  return <details className="azure-settings" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Azure Foundry <span className="azure-summary-state">{connection.hasKey ? `${Object.keys(connection.deployments).length} deployments saved` : 'Set up connection'}</span></summary>
    <p>Connect your Azure resource, then add the deployments you want to use.</p>
    {form.conflict && <div className="azure-conflict" role="alert">
      <p>Saved settings may have changed. Reload before continuing; reloading discards this draft.</p>
      <button type="button" disabled={form.saving} onClick={form.reload}>Reload saved settings</button>
    </div>}

    <section aria-labelledby="azure-resource-heading">
      <h3 id="azure-resource-heading">1. Connect resource</h3>
      <label htmlFor="azure-endpoint">Resource endpoint</label>
      <input {...fieldProps('azure-endpoint')} value={form.endpoint} type="url" spellCheck={false}
        placeholder="https://your-resource.services.ai.azure.com/openai/v1"
        disabled={unavailable || form.baseline.endpointScope === 'process'}
        onChange={event => form.editEndpoint(event.target.value)} />
      {fieldError('azure-endpoint')}
      <label htmlFor="azure-key">{form.baseline.hasKey ? 'API key · saved' : 'API key'}</label>
      <input {...fieldProps('azure-key')} value={form.key} type="password" autoComplete="off" spellCheck={false}
        placeholder={form.baseline.hasKey ? 'Leave blank to keep the saved key' : 'Paste the resource API key'}
        disabled={unavailable || form.baseline.keyScope === 'process'}
        onChange={event => form.editKey(event.target.value)} />
      {fieldError('azure-key')}
      <p className="azure-help">One key is shared by all deployments below. Changing resources requires its key.</p>
      {(form.baseline.keyScope === 'process' || form.baseline.endpointScope === 'process') &&
        <p className="azure-help">Locked values come from the launch environment. Change them there and restart Rune.</p>}
      <div className="azure-actions">
        <button type="button" className="azure-primary" disabled={unavailable || form.pending.connection}
          onClick={() => form.check()}>Connect and fetch models</button>
        {form.pending.connection && <button type="button" onClick={() => form.stop('connection')}>Cancel check</button>}
      </div>
      {form.pending.connection && <p role="status">Checking the key and loading the catalog…</p>}
      <p className="azure-help">No tokens are generated. Changes are saved only when you choose Save.</p>
      <CheckStatus result={form.checks.connection} />
    </section>

    <section aria-labelledby="azure-deployments-heading">
      <h3 id="azure-deployments-heading">2. Add deployments</h3>
      <p className="azure-help">The catalog lists model offerings. Enter your deployment's exact name from Foundry to use it.</p>
      {form.catalog !== null && <div className="azure-catalog">
        <label htmlFor="azure-model-search">Search model catalog</label>
        <input id="azure-model-search" type="search" value={query} placeholder="Search by model name or version"
          disabled={unavailable} onChange={event => setQuery(event.target.value)} />
        <p className="azure-help" role="status">{matches.length} of {form.catalog.length} models</p>
        <label htmlFor="azure-catalog">Matching models</label>
        <select id="azure-catalog" size={5} value={picked} disabled={unavailable || !matches.length}
          onChange={event => setSelected(event.target.value)}>
          <option value="" disabled>{matches.length ? 'Choose a model' : 'No matching models'}</option>
          {matches.map(model => <option key={model} value={model}>{model}</option>)}
        </select>
        <button type="button" disabled={unavailable || !picked} onClick={() => { form.add(picked); setSelected(''); }}>
          Add selected model
        </button>
      </div>}
      {!form.rows.length && <p>No deployments registered yet. Choose a catalog model or add one manually.</p>}
      {form.rows.length > 0 && <p id="azure-test-help" className="azure-help">
        Test deployment sends one short request, up to 256 output tokens. Azure usage charges may apply.
      </p>}
      {form.baseline.deploymentTestBlocked && <p id="azure-test-blocked" className="azure-help">{form.baseline.deploymentTestBlocked}</p>}
      {form.rows.map((row, index) => {
        const canPrice = ['gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna'].includes(row.model.trim());
        const active = activeModel?.provider === 'azure' && activeModel.model === row.name
          && form.baseline.deployments[row.name] === row.model;
        return <section key={row.id} className="azure-deployment" aria-labelledby={`${row.id}-heading`}>
          <div className="azure-row-heading">
            <h4 id={`${row.id}-heading`}>Deployment {index + 1}</h4>
            {active && <span className="azure-badge">In use</span>}
          </div>
          <label htmlFor={`${row.id}-name`}>Deployment name</label>
          <input {...fieldProps(`${row.id}-name`)} value={row.name} disabled={unavailable}
            placeholder="Exact name from Foundry" onChange={event => form.editRow(row.id, { name: event.target.value })} />
          {fieldError(`${row.id}-name`)}
          <label htmlFor={`${row.id}-model`}>Underlying model</label>
          <input {...fieldProps(`${row.id}-model`)} value={row.model} disabled={unavailable} list="azure-model-suggestions"
            placeholder="e.g. Kimi-K2.7-Code" onChange={event => form.editRow(row.id, { model: event.target.value, pricing: '' })} />
          {fieldError(`${row.id}-model`)}
          <div className="azure-actions">
            <button type="button" disabled={unavailable || form.pending[row.id] || !!form.baseline.deploymentTestBlocked}
              aria-describedby={form.baseline.deploymentTestBlocked ? 'azure-test-blocked' : 'azure-test-help'}
              aria-label={`Test deployment ${row.name || index + 1}`} onClick={() => form.check(row)}>Test deployment</button>
            {form.pending[row.id] && <button type="button" onClick={() => form.stop(row.id)}>Stop waiting</button>}
          </div>
          {form.pending[row.id] && <p role="status">Waiting for this deployment's response… You can keep editing.</p>}
          <CheckStatus result={form.checks[row.id]} />
          <details className="azure-pricing">
            <summary>Cost estimates (optional)</summary>
            {canPrice ? <>
              <label htmlFor={`${row.id}-pricing`}>Deployment pricing</label>
              <select id={`${row.id}-pricing`} value={row.pricing} disabled={unavailable}
                onChange={event => form.editRow(row.id, { pricing: event.target.value })}>
                <option value="">Unknown / other deployment type</option>
                <option value="global">Global Standard</option>
                <option value="us">Data Zone Standard — US</option>
                <option value="eu">Data Zone Standard — EU</option>
              </select>
              <p className="azure-help">Match the deployment type in Foundry. Leave unknown if unsure.</p>
            </> : <p className="azure-help">Rune has no configured price for this model. Cost estimates remain unknown.</p>}
          </details>
          <div className="azure-actions azure-row-footer">
            <button type="button" className="azure-primary" disabled={unavailable}
              aria-label={`Save and use ${row.name || `deployment ${index + 1}`}`} onClick={() => form.save(row.name.trim())}>Save and use</button>
            <button type="button" className="azure-remove" disabled={unavailable}
              aria-label={`Remove deployment ${row.name || index + 1}`} onClick={() => form.remove(row.id)}>Remove</button>
          </div>
        </section>;
      })}
      <datalist id="azure-model-suggestions">
        {[...new Set([...connection.suggestedModels, ...(form.catalog ?? [])])].map(model => <option key={model} value={model} />)}
      </datalist>
      <button type="button" disabled={unavailable} onClick={() => form.add()}>Add deployment manually</button>
    </section>
    <div className="azure-save-bar">
      <p className="azure-help">Saving applies all changes in this form.</p>
      {form.failure && <p id="azure-save-error" tabIndex={-1} className="azure-field-error" role="alert">{form.failure}</p>}
      {form.message && <p role="status">{form.message}</p>}
      {form.saving && <p role="status">Saving Azure settings…</p>}
      <div className="azure-actions">
        <button type="button" disabled={unavailable || !form.dirty} onClick={() => form.save()}>Save settings</button>
        {form.dirty && <span className="azure-help">Unsaved changes</span>}
      </div>
    </div>
  </details>;
}
