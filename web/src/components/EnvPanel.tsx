import { useCallback, useEffect, useState } from 'react';
import {
  fetchEnvVars,
  setEnvVar,
  unsetEnvVar,
  type EnvVarInfo,
} from '../api';

interface EnvPanelProps {
  onClose: () => void;
}

type ScopeFilter = 'all' | 'user' | 'project';
type ViewMode = 'list' | 'view' | 'edit' | 'create';

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

const SCOPE_COLORS: Record<string, string> = {
  user: 'var(--accent)',
  project: 'var(--success)',
};

export function EnvPanel({ onClose }: EnvPanelProps) {
  const [variables, setVariables] = useState<EnvVarInfo[]>([]);
  const [paths, setPaths] = useState<{ user: string; project: string }>({ user: '', project: '' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<EnvVarInfo | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Form state
  const [formKey, setFormKey] = useState('');
  const [formValue, setFormValue] = useState('');
  const [formScope, setFormScope] = useState<'user' | 'project'>('user');

  const [deleteConfirm, setDeleteConfirm] = useState<EnvVarInfo | null>(null);

  const loadVars = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchEnvVars();
      setVariables(result.variables);
      if (result.paths) setPaths(result.paths);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load env vars');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadVars(); }, [loadVars]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (viewMode === 'edit' || viewMode === 'create') {
          setViewMode(selected ? 'view' : 'list');
        } else {
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, viewMode, selected]);

  const handleSelect = (v: EnvVarInfo) => {
    setSelected(v);
    setViewMode('view');
  };

  const handleStartCreate = () => {
    setFormKey('');
    setFormValue('');
    setFormScope('user');
    setSelected(null);
    setViewMode('create');
  };

  const handleStartEdit = () => {
    if (!selected) return;
    setFormKey(selected.key);
    setFormValue('');
    setFormScope(selected.scope);
    setViewMode('edit');
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const key = formKey.trim().toUpperCase();
      await setEnvVar(key, formValue, formScope);
      setHasChanges(true);
      await loadVars();
      const updated = (await fetchEnvVars()).variables.find((v) => v.key === key);
      if (updated) {
        setSelected(updated);
        setViewMode('view');
      } else {
        setViewMode('list');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (v: EnvVarInfo) => {
    setSaving(true);
    try {
      await unsetEnvVar(v.key, v.scope);
      setHasChanges(true);
      await loadVars();
      setSelected(null);
      setViewMode('list');
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete');
    } finally {
      setSaving(false);
    }
  };

  // Filter and group
  const filtered = variables.filter((v) => {
    if (scopeFilter !== 'all' && v.scope !== scopeFilter) return false;
    if (search) {
      return v.key.toLowerCase().includes(search.toLowerCase());
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
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="fade-scale"
        style={{
          width: '90vw',
          maxWidth: 960,
          height: '80vh',
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
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
            Environment Variables
          </span>
          <div style={{ flex: 1 }} />
          <button
            onClick={handleStartCreate}
            style={{
              padding: '6px 14px',
              background: 'var(--accent)',
              color: 'white',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            + Add Variable
          </button>
          <button
            onClick={onClose}
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

        {/* Restart warning */}
        {hasChanges && (
          <div style={{
            padding: '8px 20px',
            background: 'rgba(234,179,8,0.1)',
            borderBottom: '1px solid var(--warning)',
            color: 'var(--warning)',
            fontSize: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3">
              <path d="M7 1L1 13h12L7 1z" />
              <path d="M7 5v3M7 10v.5" />
            </svg>
            <span>Changes require a daemon restart to take effect.</span>
          </div>
        )}

        {/* Error banner */}
        {error && (
          <div style={{
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
            <button
              onClick={() => setError(null)}
              style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', fontSize: 14 }}
            >
              {'\u00D7'}
            </button>
          </div>
        )}

        {/* Content */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Left: Variable list */}
          <div style={{
            width: 280,
            flexShrink: 0,
            borderRight: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            {/* Search */}
            <div style={{ padding: '12px 12px 8px' }}>
              <input
                type="text"
                placeholder="Search variables..."
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
              {(['all', 'user', 'project'] as ScopeFilter[]).map((s) => (
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
                  {s}
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
                  No variables found
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
                        onClick={() => handleSelect(v)}
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
                            {v.key}
                          </span>
                          <Badge label={v.scope} color={SCOPE_COLORS[v.scope] ?? 'var(--text-muted)'} />
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

          {/* Right: Detail/Editor */}
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {viewMode === 'list' && !selected && (
              <EnvEmptyState onCreateNew={handleStartCreate} />
            )}

            {viewMode === 'view' && selected && (
              <EnvDetailView
                envVar={selected}
                paths={paths}
                onEdit={handleStartEdit}
                onDelete={() => setDeleteConfirm(selected)}
              />
            )}

            {(viewMode === 'edit' || viewMode === 'create') && (
              <EnvEditor
                mode={viewMode}
                envKey={formKey}
                value={formValue}
                scope={formScope}
                paths={paths}
                saving={saving}
                onKeyChange={setFormKey}
                onValueChange={setFormValue}
                onScopeChange={setFormScope}
                onSave={handleSave}
                onCancel={() => setViewMode(selected ? 'view' : 'list')}
              />
            )}
          </div>
        </div>

        {/* Delete confirmation */}
        {deleteConfirm && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10,
          }}>
            <div
              className="fade-scale"
              style={{
                padding: '24px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-lg)',
                boxShadow: 'var(--shadow-lg)',
                maxWidth: 360,
                width: '90%',
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
                Remove Variable
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
                Remove <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{deleteConfirm.key}</strong> from{' '}
                <strong>{deleteConfirm.scope}</strong> scope?
              </div>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button
                  onClick={() => setDeleteConfirm(null)}
                  style={{
                    padding: '6px 14px',
                    background: 'var(--bg-tertiary)',
                    border: 'none',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-secondary)',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleDelete(deleteConfirm)}
                  disabled={saving}
                  style={{
                    padding: '6px 14px',
                    background: 'var(--danger)',
                    border: 'none',
                    borderRadius: 'var(--radius-md)',
                    color: 'white',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    opacity: saving ? 0.6 : 1,
                  }}
                >
                  {saving ? 'Removing...' : 'Remove'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
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

function EnvEmptyState({ onCreateNew }: { onCreateNew: () => void }) {
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
      <div style={{ fontSize: 13 }}>Select a variable to view details</div>
      <button
        onClick={onCreateNew}
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
        or add a new one
      </button>
    </div>
  );
}

function EnvDetailView({
  envVar,
  paths,
  onEdit,
  onDelete,
}: {
  envVar: EnvVarInfo;
  paths: { user: string; project: string };
  onEdit: () => void;
  onDelete: () => void;
}) {
  const filePath = envVar.scope === 'user' ? paths.user : paths.project;

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
        <span style={{
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
          flex: 1,
        }}>
          {envVar.key}
        </span>
        <button
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
        </button>
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
          <Badge label={envVar.scope} color={SCOPE_COLORS[envVar.scope] ?? 'var(--text-muted)'} />
        </DetailRow>

        <DetailRow label="Category">
          <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>
            {CATEGORY_CONFIG[envVar.category]?.label ?? envVar.category}
          </span>
        </DetailRow>

        <DetailRow label="File">
          <code style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-secondary)',
          }}>
            {filePath}
          </code>
        </DetailRow>

        {envVar.isSecret && (
          <div style={{
            padding: '10px 14px',
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontSize: 11,
            color: 'var(--text-muted)',
            lineHeight: 1.5,
          }}>
            This is a secret value. The full value cannot be viewed through the web UI.
            To update, click Edit and enter the new value.
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
  saving: boolean;
  onKeyChange: (v: string) => void;
  onValueChange: (v: string) => void;
  onScopeChange: (v: 'user' | 'project') => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const [showValue, setShowValue] = useState(false);
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
                value={envKey}
                onChange={(e) => onKeyChange(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, '_'))}
                placeholder="MY_API_KEY"
                style={inputStyle}
                autoFocus
              />
            </FormField>
          )}

          {/* Value */}
          <FormField label="Value" hint={mode === 'edit' ? 'Enter the new value' : undefined}>
            <div style={{ position: 'relative' }}>
              <input
                type={showValue ? 'text' : 'password'}
                value={value}
                onChange={(e) => onValueChange(e.target.value)}
                placeholder={mode === 'edit' ? 'Enter new value...' : 'sk-...'}
                style={{ ...inputStyle, paddingRight: 36 }}
                autoFocus={mode === 'edit'}
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
          <FormField label="Scope" hint="Where to save">
            <div style={{ display: 'flex', gap: 8 }}>
              {(['user', 'project'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => onScopeChange(s)}
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
