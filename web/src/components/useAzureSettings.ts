import { useEffect, useRef, useState } from 'react';
import { ApiError, fetchConfig, type AzurePricing } from '../api';
import { fetchAzureModels, saveAzureSettings, testAzureConnection, type AzureCheckResult } from '../azureApi';
import { connectionErrors, focusError, resource, rowErrors, type AzureConnection, type AzureRow } from './azureForm';

export function useAzureSettings(connection: AzureConnection, onSaved: () => Promise<void>) {
  const nextId = useRef(0);
  const toRows = (value: AzureConnection) => Object.entries(value.deployments).map(([name, model]) => ({
    id: `azure-row-${nextId.current++}`, name, model, pricing: value.pricing?.endpoint === resource(value.endpoint)
      ? value.pricing.deployments[name] ?? '' : '',
  }));
  const [baseline, setBaseline] = useState(connection);
  const [endpoint, setEndpoint] = useState(connection.endpoint);
  const [key, setKey] = useState('');
  const [rows, setRows] = useState<AzureRow[]>(() => toRows(connection));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const [failure, setFailure] = useState('');
  const [conflict, setConflict] = useState(false);
  const [saving, setSaving] = useState(false);
  const [checks, setChecks] = useState<Record<string, AzureCheckResult>>({});
  const [catalog, setCatalog] = useState<string[] | null>(null);
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const requests = useRef(new Map<string, AbortController>());
  const draft = () => ({ endpoint: endpoint.trim(), apiKey: key.trim() || undefined, expectedRevision: baseline.revision });
  const deployments = Object.fromEntries(rows.map(row => [row.name.trim(), row.model.trim()]));
  const prices = Object.fromEntries(rows.filter(row => row.pricing).map(row => [row.name.trim(), row.pricing]));
  const originalPrices = baseline.pricing?.endpoint === resource(baseline.endpoint) ? baseline.pricing.deployments : {};
  const dirty = key !== '' || resource(endpoint) !== resource(baseline.endpoint)
    || JSON.stringify(deployments) !== JSON.stringify(baseline.deployments)
    || JSON.stringify(prices) !== JSON.stringify(originalPrices) || rows.some(row => !row.name || !row.model);
  const current = useRef({ dirty, baseline });
  current.current = { dirty, baseline };

  const clearResult = (id: string) => setChecks(previous => {
    const next = { ...previous }; delete next[id]; return next;
  });
  const stop = (id: string, notify = true) => {
    requests.current.get(id)?.abort(); requests.current.delete(id);
    setPending(previous => ({ ...previous, [id]: false }));
    if (notify) setChecks(previous => ({ ...previous, [id]: { ok: false, state: 'cancelled',
      detail: id === 'connection' ? 'Connection check cancelled.' : 'Stopped waiting. Azure may still charge for a request already submitted.' } }));
  };
  const invalidateConnection = () => {
    for (const id of requests.current.keys()) stop(id, false);
    setChecks({}); setCatalog(null);
  };
  useEffect(() => () => { for (const controller of requests.current.values()) controller.abort(); requests.current.clear(); }, []);
  useEffect(() => {
    if (connection.revision === current.current.baseline.revision) { setBaseline(connection); return; }
    if (connection.credentialRevision !== current.current.baseline.credentialRevision) invalidateConnection();
    if (current.current.dirty) { setConflict(true); return; }
    setBaseline(connection); setEndpoint(connection.endpoint); setRows(toRows(connection));
    setChecks({});
  }, [connection]); // Polls with an unchanged revision do not erase checks or drafts.

  const edit = (fieldIds: string[]) => {
    setMessage(''); setFailure('');
    setErrors(previous => Object.fromEntries(Object.entries(previous).filter(([id]) => !fieldIds.includes(id))));
  };
  const editEndpoint = (value: string) => {
    invalidateConnection(); edit(['azure-endpoint', 'azure-key']); setEndpoint(value);
    if (resource(value) !== resource(endpoint)) setRows(previous => previous.map(row => ({ ...row, pricing: '' })));
  };
  const editKey = (value: string) => { invalidateConnection(); edit(['azure-key']); setKey(value); };
  const editRow = (id: string, changes: Partial<AzureRow>) => {
    if (changes.name !== undefined || changes.model !== undefined) { stop(id, false); clearResult(id); }
    edit([`${id}-name`, `${id}-model`]);
    setRows(previous => previous.map(row => row.id === id ? { ...row, ...changes } : row));
  };
  const add = (model = '') => {
    const id = `azure-row-${nextId.current++}`;
    setRows(previous => [...previous, { id, name: '', model, pricing: '' }]); edit([]);
    requestAnimationFrame(() => document.getElementById(`${id}-name`)?.focus());
  };
  const remove = (id: string) => { stop(id, false); clearResult(id); edit([]); setRows(previous => previous.filter(row => row.id !== id)); };
  const validate = (selectedRows: AzureRow[]) => {
    const found = { ...connectionErrors(baseline, endpoint, key), ...rowErrors(selectedRows) };
    setErrors(found);
    if (Object.keys(found).length) { focusError(found); return false; }
    return true;
  };
  const check = async (row?: AzureRow) => {
    if (conflict || !validate(row ? [row] : [])) return;
    const id = row?.id ?? 'connection';
    stop(id, false); clearResult(id);
    const controller = new AbortController(); requests.current.set(id, controller);
    setPending(previous => ({ ...previous, [id]: true }));
    try {
      const result = row ? await testAzureConnection(draft(), row.name.trim(), row.model.trim(), controller.signal)
        : await fetchAzureModels(draft(), controller.signal);
      if (controller.signal.aborted || requests.current.get(id) !== controller) return;
      setChecks(previous => ({ ...previous, [id]: result }));
      if (!row) setCatalog(result.ok ? result.models ?? [] : null);
    } catch (err) {
      if (controller.signal.aborted) return;
      setChecks(previous => ({ ...previous, [id]: { ok: false, detail: err instanceof Error ? err.message : 'Could not check Azure.' } }));
      if (err instanceof ApiError && err.status === 409) setConflict(true);
    } finally {
      if (requests.current.get(id) === controller) { requests.current.delete(id); setPending(previous => ({ ...previous, [id]: false })); }
    }
  };
  const save = async (useDeployment?: string) => {
    if (conflict || !validate(rows)) return;
    setSaving(true); setFailure(''); setMessage('');
    for (const id of requests.current.keys()) stop(id, false);
    const pricing = Object.keys(prices).length ? { endpoint: resource(endpoint), deployments: prices } as AzurePricing : null;
    try {
      const result = await saveAzureSettings({ ...draft(), revision: baseline.revision, deployments, pricing, useDeployment });
      setBaseline({ ...baseline, endpoint: endpoint.trim(), hasKey: true, deployments, pricing,
        revision: result.revision, credentialRevision: result.credentialRevision });
      setKey('');
      setMessage(useDeployment ? `Saved. ${useDeployment} is selected for new runs.` : 'Azure settings saved.');
      window.dispatchEvent(new Event('rune:config-changed'));
      try { await onSaved(); } catch { setMessage('Saved. Refresh settings to see the latest selection.'); }
    } catch (err) {
      setFailure(err instanceof ApiError ? err.message : 'Could not confirm saving. Reload saved settings before retrying.');
      if (!(err instanceof ApiError) || err.status === 409) setConflict(true);
      requestAnimationFrame(() => document.getElementById('azure-save-error')?.focus());
    } finally { setSaving(false); }
  };
  const reload = async () => {
    setSaving(true); setFailure('');
    try {
      const latest = (await fetchConfig()).azure;
      if (!latest) throw new Error('Azure settings are unavailable.');
      invalidateConnection(); setBaseline(latest); setEndpoint(latest.endpoint); setKey('');
      setRows(toRows(latest)); setErrors({}); setMessage(''); setConflict(false);
      await onSaved();
    } catch (err) {
      setFailure(err instanceof Error ? err.message : 'Could not reload saved settings. Your draft was kept.');
    } finally { setSaving(false); }
  };
  return { baseline, endpoint, key, rows, errors, message, failure, conflict, saving, dirty, checks, catalog, pending,
    editEndpoint, editKey, editRow, add, remove, check, stop, save, reload };
}
