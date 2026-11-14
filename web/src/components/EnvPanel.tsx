import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchEnvVars,
  setEnvVar,
  unsetEnvVar,
  type EnvVarInfo,
  type EnvActivation,
} from '../api';
import { useFocusTrap } from '../hooks/useFocusTrap';
import './EnvPanel.css';

interface EnvPanelProps {
  onClose: () => void;
  initialMode?: 'providers' | 'all';
  initialKey?: string;
  onChooseModel?: (provider: string) => void;
  managed?: boolean;
}

type ScopeFilter = 'all' | 'user' | 'project' | 'process';
type ViewMode = 'list' | 'view' | 'edit' | 'create';
type SaveStatus = { key: string; action: 'saved' | 'removed'; activation: EnvActivation; provider?: string };
const UNKNOWN_ACTIVATION: EnvActivation = { state: 'unknown', reason: 'The engine did not report when this change takes effect.' };

const CATEGORY_CONFIG: Record<string, { label: string; order: number }> = {
  llm: { label: 'LLM Providers', order: 0 },
  search: { label: 'Search', order: 1 },
  logging: { label: 'Logging', order: 2 },
  telegram: { label: 'Telegram', order: 3 },
  discord: { label: 'Discord', order: 4 },
  slack: { label: 'Slack', order: 5 },
  mattermost: { label: 'Mattermost', order: 6 },
  line: { label: 'LINE', order: 7 },
  whatsapp: { label: 'WhatsApp', order: 8 },
  'google-chat': { label: 'Google Chat', order: 9 },
  other: { label: 'Other', order: 10 },
};

const PROVIDER_KEYS = [
  { key: 'OPENAI_API_KEY', name: 'OpenAI', provider: 'openai' },
  { key: 'ANTHROPIC_API_KEY', name: 'Anthropic', provider: 'anthropic' },
  { key: 'GEMINI_API_KEY', name: 'Google Gemini', provider: 'gemini' },
  { key: 'XAI_API_KEY', name: 'xAI', provider: 'xai' },
];

const providerForKey = (key: string) => PROVIDER_KEYS.find(provider => provider.key === key
  || (provider.key === 'GEMINI_API_KEY' && key === 'GOOGLE_API_KEY'));

const SCOPE_COLORS: Record<string, string> = {
  user: 'var(--accent)',
  project: 'var(--success)',
};

export function EnvPanel({ onClose, initialMode = 'all', initialKey, onChooseModel, managed = false }: EnvPanelProps) {
  const [mode, setMode] = useState(initialMode);
  const [variables, setVariables] = useState<EnvVarInfo[]>([]);
  const [paths, setPaths] = useState<{ user: string; project: string }>({ user: '', project: '' });
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<EnvVarInfo | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<SaveStatus | null>(null);

  const [formKey, setFormKey] = useState('');
  const [formValue, setFormValue] = useState('');
  const [formScope, setFormScope] = useState<'user' | 'project'>('user');

  const [deleteConfirm, setDeleteConfirm] = useState<EnvVarInfo | null>(null);
  const returnFocus = useRef(typeof document === 'undefined' ? null : document.activeElement as HTMLElement | null);
  const deleteTrigger = useRef<HTMLElement | null>(null);
  const listTrigger = useRef<HTMLElement | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const trapRef = useFocusTrap<HTMLDivElement>(!deleteConfirm, returnFocus.current);
  const confirmRef = useFocusTrap<HTMLDivElement>(Boolean(deleteConfirm), deleteTrigger.current);
  const editing = viewMode === 'edit' || viewMode === 'create';

  const loadVars = useCallback(async (): Promise<EnvVarInfo[] | null> => {
    setLoading(true);
    setLoadFailed(false);
    setError(null);
    try {
      const result = await fetchEnvVars();
      setVariables(result.variables);
      if (result.paths) setPaths(result.paths);
      return result.variables;
    } catch (err) {
      setLoadFailed(true);
      setError(err instanceof Error ? err.message : "Couldn't load environment variables");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    void loadVars().then(loaded => {
      if (!active || !loaded || !initialKey) return;
      const key = initialKey.trim().toUpperCase();
      const existing = loaded.find(variable => variable.key === key
        || (key === 'GEMINI_API_KEY' && variable.key === 'GOOGLE_API_KEY'));
      setFormKey(key);
      setFormValue('');
      setSelected(existing ?? null);
      setViewMode(existing ? 'view' : 'create');
    });
    return () => { active = false; };
  }, [loadVars, initialKey]);

  useEffect(() => {
    if (viewMode === 'view') detailRef.current?.querySelector<HTMLElement>('[data-env-title]')?.focus();
  }, [viewMode, selected]);

  const cancelEditing = () => {
    if (saving) return;
    setFormValue('');
    setViewMode(selected ? 'view' : 'list');
    if (!selected) requestAnimationFrame(() => (listTrigger.current?.isConnected ? listTrigger.current
      : trapRef.current?.querySelector<HTMLElement>('[aria-label="Close environment settings"]'))?.focus());
  };

  const showList = () => {
    if (saving) return;
    setFormValue('');
    setSelected(null);
    setViewMode('list');
    requestAnimationFrame(() => (listTrigger.current?.isConnected ? listTrigger.current
      : trapRef.current?.querySelector<HTMLElement>('[aria-label="Close environment settings"]'))?.focus());
  };

  const handleSelect = (v: EnvVarInfo) => {
    setSelected(v);
    setViewMode('view');
  };

  const handleStartCreate = (key = '') => {
    if (loading || loadFailed || saving) return;
    listTrigger.current = document.activeElement as HTMLElement | null;
    setError(null);
    setFormKey(key);
    setFormValue('');
    setFormScope('user');
    setSelected(null);
    setViewMode('create');
  };

  const handleStartEdit = () => {
    if (!selected || selected.scope === 'process' || saving) return;
    setError(null);
    setFormKey(selected.key);
    setFormValue('');
    setFormScope(managed ? 'user' : selected.scope);
    setViewMode('edit');
  };

  const handleSave = async () => {
    if (saving || !formKey.trim() || !formValue.trim()) return;
    const key = formKey.trim().toUpperCase();
    if (variables.some(variable => variable.key === key && variable.scope === 'process')) {
      setError('This value is inherited. Change it in the environment that starts Rune.');
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const result = await setEnvVar(key, formValue, managed ? 'user' : formScope);
      setSaveStatus({ key, action: 'saved', activation: result?.activation ?? UNKNOWN_ACTIVATION,
        provider: result?.provider ?? providerForKey(key)?.provider });
      setFormValue('');
      const updatedList = await loadVars();
      if (!updatedList) setVariables([]);
      const updated = updatedList?.find((v) => v.key === key);
      if (updated) {
        setSelected(updated);
        setViewMode('view');
      } else {
        setSelected(null);
        setViewMode('list');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't save");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (v: EnvVarInfo) => {
    if (saving || v.scope === 'process') return;
    setSaving(true);
    setError(null);
    try {
      const result = await unsetEnvVar(v.key, v.scope);
      setSaveStatus({ key: v.key, action: 'removed', activation: result?.activation ?? UNKNOWN_ACTIVATION });
      const updatedList = await loadVars();
      if (!updatedList) setVariables([]);
      setSelected(null);
      setViewMode('list');
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't delete");
    } finally {
      setSaving(false);
    }
  };

  // Filter and group
  const filtered = variables.filter((v) => {
    if (mode === 'providers' && v.category !== 'llm') return false;
    if (scopeFilter !== 'all' && v.scope !== scopeFilter) return false;
    if (search) {
      return `${v.key} ${providerForKey(v.key)?.name ?? ''}`.toLowerCase().includes(search.toLowerCase());
    }
    return true;
  });

  const grouped = new Map<string, EnvVarInfo[]>();
  for (const v of filtered) {
    const list = grouped.get(v.category) || [];
    list.push(v);
    grouped.set(v.category, list);
  }
  const sortedCategories = [...grouped.keys()].sort(
    (a, b) => (CATEGORY_CONFIG[a]?.order ?? 99) - (CATEGORY_CONFIG[b]?.order ?? 99),
  );

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !saving && !editing && !deleteConfirm) onClose(); }}
      onKeyDown={event => {
        if (event.key !== 'Escape' || event.defaultPrevented) return;
        event.preventDefault();
        event.stopPropagation();
        if (saving) return;
        if (deleteConfirm) setDeleteConfirm(null);
        else if (editing) cancelEditing();
        else onClose();
      }}
    >
      <div
        ref={trapRef}
        role="dialog"
        aria-modal="true"
        aria-label={mode === 'providers' ? 'API keys' : 'Environment variables'}
        inert={Boolean(deleteConfirm)}
        aria-hidden={deleteConfirm ? true : undefined}
        className="env-dialog fade-scale"
        style={{
          width: '90vw',
          maxWidth: 960,
          height: '80dvh',
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div className="env-header" style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '16px 20px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="10" cy="10" r="3" />
            <path d="M10 1v2M10 17v2M1 10h2M17 10h2M3.93 3.93l1.41 1.41M14.66 14.66l1.41 1.41M3.93 16.07l1.41-1.41M14.66 5.34l1.41-1.41" />
          </svg>
          <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>
            {mode === 'providers' ? 'API keys' : 'Environment variables'}
          </span>
          <div style={{ flex: 1 }} />
          <button className="env-mode-button" disabled={editing || saving} onClick={() => {
            setMode(mode === 'providers' ? 'all' : 'providers');
            setScopeFilter('all');
            setSearch('');
            setSelected(null);
            setViewMode('list');
          }}>{mode === 'providers' ? 'All environment variables' : 'Provider keys'}</button>
          <button
            onClick={() => handleStartCreate()}
            disabled={loading || loadFailed || saving || editing}
            style={{
              padding: '6px 14px',
              background: 'var(--accent)',
              color: 'var(--text-on-accent)',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            {mode === 'providers' ? 'Add key' : 'Add variable'}
          </button>
          <button
            onClick={onClose}
            disabled={saving}
            aria-label="Close environment settings"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: 18,
              cursor: 'pointer',
              padding: '2px 6px',
              lineHeight: 1,
            }}
          >
            {'\u00D7'}
          </button>
        </div>

        {saveStatus && <div className="env-save-status" data-activation={saveStatus.activation.state}>
          <div role="status" aria-live="polite" aria-atomic="true">
            <strong>{saveStatus.key} {saveStatus.action}.</strong>{' '}
            {saveStatus.activation.state === 'next_run' ? 'Applies to new runs.'
              : saveStatus.activation.state === 'restart_required'
                ? `${managed ? 'Restart the workspace engine' : 'Restart Rune’s engine'} to apply this change.`
                : 'Activation timing is unknown.'}
            {saveStatus.activation.reason && <p>{saveStatus.activation.reason}</p>}
          </div>
          {saveStatus.action === 'saved' && saveStatus.provider && onChooseModel && <button type="button"
            disabled={saving || editing} onClick={() => onChooseModel(saveStatus.provider!)}>Choose model</button>}
        </div>}

        {/* Error banner */}
        {error && (
          <div role="alert" style={{
            padding: '8px 20px',
            background: 'rgba(239,68,68,0.1)',
            borderBottom: '1px solid var(--danger)',
            color: 'var(--danger)',
            fontSize: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            <span style={{ flex: 1 }}>{error}</span>
            {loadFailed && <button className="env-mode-button" onClick={() => void loadVars()}>Reload</button>}
            {!loadFailed && <button
              onClick={() => setError(null)}
              aria-label="Dismiss environment error"
              style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', fontSize: 14 }}
            >
              {'\u00D7'}
            </button>}
          </div>
        )}

        {mode === 'providers' && <p className="env-intro">Choose a provider to add or update its key. {managed
          ? 'Keys are saved in your hosted workspace.' : 'Keys are saved on your connected Rune engine.'}</p>}
        <div className="env-layout" data-view={viewMode}>
          <div className="env-list" style={{
            flexShrink: 0,
            borderRight: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            {mode === 'providers' && <div className="env-provider-shortcuts" aria-label="Provider keys">
              {PROVIDER_KEYS.map(provider => {
                const existing = variables.find(variable => variable.key === provider.key
                  || (provider.key === 'GEMINI_API_KEY' && variable.key === 'GOOGLE_API_KEY'));
                return <button key={provider.key} disabled={loading || loadFailed || saving || editing} onClick={event => {
                  listTrigger.current = event.currentTarget;
                  if (existing) handleSelect(existing);
                  else handleStartCreate(provider.key);
                }}><strong>{provider.name}</strong><span>{existing
                  ? existing.scope === 'process' ? 'Inherited key' : 'Saved key' : 'Add key'}</span></button>;
              })}
            </div>}
            <div style={{ padding: '12px 12px 8px' }}>
              <input
                type="text"
                placeholder={mode === 'providers' ? 'Search providers or keys' : 'Search variables'}
                aria-label="Search environment variables"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{
                  width: '100%',
                  padding: '7px 10px',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  color: 'var(--text-primary)',
                  fontSize: 12,
                  outline: 'none',
                  fontFamily: 'var(--font-sans)',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* Scope filter */}
            <div style={{ display: 'flex', gap: 2, padding: '0 12px 8px' }}>
              {(['all', 'user', ...(managed ? [] : ['project']), ...(variables.some(variable => variable.scope === 'process') ? ['process'] : [])] as ScopeFilter[]).map((s) => (
                <button
                  key={s}
                  onClick={() => setScopeFilter(s)}
                  style={{
                    flex: 1,
                    padding: '4px 0',
                    background: scopeFilter === s ? 'var(--bg-tertiary)' : 'transparent',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    color: scopeFilter === s ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontSize: 10,
                    fontWeight: 500,
                    cursor: 'pointer',
                    textTransform: 'capitalize',
                  }}
                >
                  {s === 'process' ? 'Inherited' : s}
                </button>
              ))}
            </div>

            {/* Variable list grouped by category */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 8px' }}>
              {loading ? (
                <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  <div>Loading...</div>
                </div>
              ) : filtered.length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  {loadFailed ? 'The saved variables could not be loaded.' : 'No variables found'}
                </div>
              ) : (
                sortedCategories.map((cat) => (
                  <div key={cat}>
                    <div style={{
                      fontSize: 9,
                      fontWeight: 600,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      padding: '10px 12px 4px',
                    }}>
                      {CATEGORY_CONFIG[cat]?.label ?? cat}
                    </div>
                    {grouped.get(cat)!.map((v) => (
                      <button
                        key={`${v.key}-${v.scope}`}
                        disabled={saving || editing}
                        onClick={event => { listTrigger.current = event.currentTarget; handleSelect(v); }}
                        style={{
                          display: 'block',
                          width: '100%',
                          textAlign: 'left',
                          padding: '8px 12px',
                          marginBottom: 1,
                          background: selected?.key === v.key && selected?.scope === v.scope
                            ? 'var(--bg-tertiary)' : 'transparent',
                          border: 'none',
                          borderRadius: 'var(--radius-md)',
                          cursor: 'pointer',
                          borderLeft: selected?.key === v.key && selected?.scope === v.scope
                            ? '2px solid var(--accent)' : '2px solid transparent',
                          transition: 'background 0.1s',
                        }}
                        onMouseEnter={(e) => {
                          if (!(selected?.key === v.key && selected?.scope === v.scope))
                            e.currentTarget.style.background = 'var(--bg-hover)';
                        }}
                        onMouseLeave={(e) => {
                          if (!(selected?.key === v.key && selected?.scope === v.scope))
                            e.currentTarget.style.background = 'transparent';
                        }}
                      >
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                        }}>
                          {v.isSecret && (
                            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="var(--text-muted)" strokeWidth="1.2">
                              <rect x="1.5" y="5" width="7" height="4" rx="0.5" />
                              <path d="M3 5V3.5a2 2 0 014 0V5" />
                            </svg>
                          )}
                          <span style={{
                            fontSize: 11,
                            fontWeight: 500,
                            color: 'var(--text-primary)',
                            fontFamily: 'var(--font-mono)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            flex: 1,
                          }}>
                            {mode === 'providers' ? providerForKey(v.key)?.name ?? v.key : v.key}
                          </span>
                          <Badge label={v.scope === 'process' ? 'Inherited' : v.scope} color={SCOPE_COLORS[v.scope] ?? 'var(--text-muted)'} />
                        </div>
                        <div style={{
                          fontSize: 10,
                          color: 'var(--text-muted)',
                          fontFamily: 'var(--font-mono)',
                          marginTop: 2,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {v.maskedValue}
                        </div>
                      </button>
                    ))}
                  </div>
                ))
              )}
            </div>
          </div>

          <div ref={detailRef} className="env-detail">
            {viewMode !== 'list' && <button className="env-back" disabled={saving} onClick={showList}>← Back to {mode === 'providers' ? 'keys' : 'variables'}</button>}
            {viewMode === 'list' && !selected && (
              <EnvEmptyState providers={mode === 'providers'} disabled={loading || loadFailed} onCreateNew={() => handleStartCreate()} />
            )}

            {viewMode === 'view' && selected && (
              <EnvDetailView
                envVar={selected}
                paths={paths}
                managed={managed}
                onEdit={handleStartEdit}
                onDelete={() => {
                  deleteTrigger.current = document.activeElement as HTMLElement | null;
                  setError(null);
                  setDeleteConfirm(selected);
                }}
              />
            )}

            {(viewMode === 'edit' || viewMode === 'create') && (
              <EnvEditor
                mode={viewMode}
                envKey={formKey}
                value={formValue}
                scope={formScope}
                paths={paths}
                managed={managed}
                saving={saving}
                onKeyChange={setFormKey}
                onValueChange={setFormValue}
                onScopeChange={setFormScope}
                onSave={handleSave}
                onCancel={cancelEditing}
              />
            )}
          </div>
        </div>

      </div>
      {deleteConfirm && <div className="env-delete-overlay" onClick={event => {
        if (event.target === event.currentTarget && !saving) setDeleteConfirm(null);
      }}>
        <div ref={confirmRef} className="env-delete-dialog fade-scale" role="alertdialog" aria-modal="true"
          aria-labelledby="env-delete-title" aria-describedby="env-delete-description">
          <h2 id="env-delete-title">Remove variable?</h2>
          <p id="env-delete-description">Remove <strong>{deleteConfirm.key}</strong> from {deleteConfirm.scope} settings?</p>
          {error && <p role="alert" className="env-delete-error">{error}</p>}
          <div className="env-delete-actions">
            <button disabled={saving} onClick={() => setDeleteConfirm(null)}>Cancel</button>
            <button className="env-remove" disabled={saving} onClick={() => handleDelete(deleteConfirm)}>
              {saving ? 'Removing…' : 'Remove'}
            </button>
          </div>
        </div>
      </div>}
    </div>
  );
}

// ── Sub-components ──

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 6px',
      background: `color-mix(in srgb, ${color} 15%, transparent)`,
      color,
      borderRadius: 'var(--radius-sm)',
      fontSize: 9,
      fontWeight: 500,
      textTransform: 'capitalize',
      letterSpacing: '0.3px',
      lineHeight: '16px',
    }}>
      {label}
    </span>
  );
}

function EnvEmptyState({ onCreateNew, providers, disabled }: { onCreateNew: () => void; providers: boolean; disabled: boolean }) {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 12,
      color: 'var(--text-muted)',
    }}>
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.2" opacity={0.4}>
        <circle cx="20" cy="20" r="8" />
        <path d="M20 4v4M20 32v4M4 20h4M32 20h4M9.17 9.17l2.83 2.83M27.99 27.99l2.83 2.83M9.17 30.83l2.83-2.83M27.99 12.01l2.83-2.83" />
      </svg>
      <div style={{ fontSize: 13 }}>{providers ? 'Choose a provider to get started' : 'Select a variable to view details'}</div>
      <button
        onClick={onCreateNew}
        disabled={disabled}
        style={{
          padding: '6px 16px',
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-secondary)',
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        {providers ? 'Add another provider key' : 'Add a variable'}
      </button>
    </div>
  );
}

function EnvDetailView({
  envVar,
  paths,
  managed,
  onEdit,
  onDelete,
}: {
  envVar: EnvVarInfo;
  paths: { user: string; project: string };
  managed: boolean;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const inherited = envVar.scope === 'process';
  const filePath = inherited ? null : envVar.scope === 'user' ? paths.user : paths.project;

  return (
    <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '12px 20px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <span data-env-title tabIndex={-1} style={{
          overflowWrap: 'anywhere',
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
          flex: 1,
        }}>
          {envVar.key}
        </span>
        {!inherited && <><button
          onClick={onEdit}
          style={{
            padding: '5px 12px',
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--text-secondary)',
            fontSize: 11,
            cursor: 'pointer',
          }}
        >
          Edit
        </button>
        <button
          onClick={onDelete}
          style={{
            padding: '5px 12px',
            background: 'transparent',
            border: '1px solid var(--danger)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--danger)',
            fontSize: 11,
            cursor: 'pointer',
          }}
        >
          Remove
        </button></>}
      </div>

      {/* Detail */}
      <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <DetailRow label="Value">
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            {envVar.isSecret && (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="var(--text-muted)" strokeWidth="1.2">
                <rect x="2" y="6" width="8" height="4.5" rx="0.5" />
                <path d="M3.5 6V4.5a2.5 2.5 0 015 0V6" />
              </svg>
            )}
            <code style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
              color: 'var(--text-primary)',
              background: 'var(--bg-secondary)',
              padding: '4px 8px',
              borderRadius: 'var(--radius-sm)',
            }}>
              {envVar.maskedValue}
            </code>
          </div>
        </DetailRow>

        <DetailRow label="Scope">
          <Badge label={inherited ? 'Inherited' : envVar.scope} color={SCOPE_COLORS[envVar.scope] ?? 'var(--text-muted)'} />
        </DetailRow>

        <DetailRow label="Category">
          <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>
            {CATEGORY_CONFIG[envVar.category]?.label ?? envVar.category}
          </span>
        </DetailRow>

        <DetailRow label={inherited ? 'Source' : 'File'}>
          <code style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-secondary)',
          }}>
            {inherited ? managed ? 'Hosted server environment (read-only)' : 'Rune process environment (read-only)' : filePath}
          </code>
        </DetailRow>

        {(envVar.isSecret || inherited) && (
          <div style={{
            padding: '10px 14px',
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontSize: 11,
            color: 'var(--text-muted)',
            lineHeight: 1.5,
          }}>
            {envVar.isSecret && 'Secret values are never displayed in full.'}
            {inherited ? ' Change it in the environment that starts Rune.' : ' To update, choose Edit and enter the new value.'}
          </div>
        )}
      </div>
    </div>
  );
}

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 10,
        fontWeight: 600,
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: 4,
      }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function EnvEditor({
  mode,
  envKey,
  value,
  scope,
  paths,
  managed,
  saving,
  onKeyChange,
  onValueChange,
  onScopeChange,
  onSave,
  onCancel,
}: {
  mode: 'edit' | 'create';
  envKey: string;
  value: string;
  scope: 'user' | 'project';
  paths: { user: string; project: string };
  managed: boolean;
  saving: boolean;
  onKeyChange: (v: string) => void;
  onValueChange: (v: string) => void;
  onScopeChange: (v: 'user' | 'project') => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const [showValue, setShowValue] = useState(false);
  const scopes = managed ? (['user'] as const) : (['user', 'project'] as const);
  const isValid = envKey.trim() && (mode === 'edit' ? value.trim() : value.trim() && envKey.trim());

  return (
    <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '12px 20px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
          {mode === 'create' ? 'Add Variable' : `Edit: ${envKey}`}
        </span>
        <button
          onClick={onCancel}
          style={{
            padding: '5px 12px',
            background: 'var(--bg-tertiary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            color: 'var(--text-secondary)',
            fontSize: 11,
            cursor: 'pointer',
          }}
        >
          Cancel
        </button>
        <button
          onClick={onSave}
          disabled={!isValid || saving}
          style={{
            padding: '5px 14px',
            background: isValid && !saving ? 'var(--accent)' : 'var(--bg-tertiary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            color: isValid && !saving ? 'white' : 'var(--text-muted)',
            fontSize: 11,
            fontWeight: 600,
            cursor: isValid && !saving ? 'pointer' : 'default',
          }}
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
      </div>

      {/* Form */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px 20px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 520 }}>
          {/* Key */}
          {mode === 'create' && (
            <FormField label="Key" hint="e.g. OPENAI_API_KEY">
              <input
                type="text"
                aria-label="Variable name"
                disabled={saving}
                value={envKey}
                onChange={(e) => onKeyChange(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, '_'))}
                placeholder="MY_API_KEY"
                style={inputStyle}
                autoFocus={!envKey}
              />
            </FormField>
          )}

          {/* Value */}
          <FormField label="Value" hint={mode === 'edit' ? 'Enter the new value' : undefined}>
            <div style={{ position: 'relative' }}>
              <input
                type={showValue ? 'text' : 'password'}
                aria-label="Value"
                autoComplete="new-password"
                disabled={saving}
                value={value}
                onChange={(e) => onValueChange(e.target.value)}
                placeholder={mode === 'edit' ? 'Enter the replacement value' : 'Paste the value'}
                style={{ ...inputStyle, paddingRight: 36 }}
                autoFocus={mode === 'edit' || Boolean(envKey)}
              />
              <button
                type="button"
                onClick={() => setShowValue(!showValue)}
                style={{
                  position: 'absolute',
                  right: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: 2,
                  fontSize: 11,
                }}
                title={showValue ? 'Hide' : 'Show'}
              >
                {showValue ? (
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.2">
                    <path d="M1 1l12 12M5.6 5.6a2 2 0 002.8 2.8M2.5 4.7C1.6 5.5 1 6.5 1 7s2 4 6 4c.8 0 1.5-.1 2.2-.4M11.5 9.3c.9-.8 1.5-1.8 1.5-2.3s-2-4-6-4c-.8 0-1.5.1-2.2.4" />
                  </svg>
                ) : (
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.2">
                    <path d="M1 7s2-4 6-4 6 4 6 4-2 4-6 4-6-4-6-4z" />
                    <circle cx="7" cy="7" r="2" />
                  </svg>
                )}
              </button>
            </div>
          </FormField>

          {/* Scope */}
          <FormField label="Scope" hint={managed ? 'This workspace engine uses user settings.' : 'Where to save'}>
            <div style={{ display: 'flex', gap: 8 }}>
              {scopes.map((s) => (
                <button
                  key={s}
                  onClick={() => onScopeChange(s)}
                  disabled={saving}
                  style={{
                    flex: 1,
                    padding: '8px',
                    background: scope === s ? 'var(--accent-subtle)' : 'var(--bg-secondary)',
                    border: `1px solid ${scope === s ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius-md)',
                    color: scope === s ? 'var(--accent)' : 'var(--text-secondary)',
                    fontSize: 12,
                    fontWeight: 500,
                    cursor: 'pointer',
                    textTransform: 'capitalize',
                  }}
                >
                  {s}
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                    {s === 'user' ? paths.user : paths.project}
                  </div>
                </button>
              ))}
            </div>
          </FormField>
        </div>
      </div>
    </div>
  );
}

function FormField({ label, hint, children }: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
        <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{label}</label>
        {hint && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{hint}</span>}
      </div>
      {children}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  color: 'var(--text-primary)',
  fontSize: 13,
  fontFamily: 'var(--font-mono)',
  outline: 'none',
  boxSizing: 'border-box',
};
