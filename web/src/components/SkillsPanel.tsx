import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchSkills,
  fetchSkill,
  createSkill,
  updateSkill,
  deleteSkill,
  type SkillInfo,
  type SkillDetail,
} from '../api';

interface SkillsPanelProps {
  onClose: () => void;
  initialSkillName?: string;
}

type ScopeFilter = 'all' | 'user' | 'project' | 'builtin';
type ViewMode = 'list' | 'view' | 'edit' | 'create';

const SCOPE_COLORS: Record<string, string> = {
  user: 'var(--accent)',
  project: 'var(--success)',
  builtin: 'var(--text-muted)',
};

const LIFECYCLE_COLORS: Record<string, string> = {
  active: 'var(--success)',
  candidate: 'var(--warning)',
  shadow: 'var(--text-muted)',
  retired: 'var(--danger)',
};

export function SkillsPanel({ onClose, initialSkillName }: SkillsPanelProps) {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('all');
  const [search, setSearch] = useState('');
  const [selectedSkill, setSelectedSkill] = useState<SkillDetail | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [saving, setSaving] = useState(false);

  // Paths from server
  const [projectPath, setProjectPath] = useState('');
  const [userPath, setUserPath] = useState('');

  // Edit/Create form state
  const [formName, setFormName] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [formBody, setFormBody] = useState('');
  const [formScope, setFormScope] = useState<'user' | 'project'>('user');
  const [formProjectPath, setFormProjectPath] = useState('');

  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const panelRef = useRef<HTMLDivElement>(null);

  const loadSkills = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchSkills();
      setSkills(result.skills);
      if (result.projectPath) setProjectPath(result.projectPath);
      if (result.userPath) setUserPath(result.userPath);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load skills');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSkills().then(() => {
      if (initialSkillName) {
        handleSelectSkill(initialSkillName);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadSkills]);

  // Escape key to close
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (viewMode === 'edit' || viewMode === 'create') {
          setViewMode(selectedSkill ? 'view' : 'list');
        } else {
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, viewMode, selectedSkill]);

  const handleSelectSkill = async (name: string) => {
    try {
      const detail = await fetchSkill(name);
      setSelectedSkill(detail);
      setViewMode('view');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load skill');
    }
  };

  const handleStartEdit = () => {
    if (!selectedSkill) return;
    setFormName(selectedSkill.name);
    setFormDescription(selectedSkill.description);
    setFormBody(selectedSkill.body);
    setViewMode('edit');
  };

  const handleStartCreate = () => {
    setFormName('');
    setFormDescription('');
    setFormBody('');
    setFormScope('user');
    setFormProjectPath(projectPath);
    setSelectedSkill(null);
    setViewMode('create');
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      if (viewMode === 'create') {
        await createSkill({
          name: formName.trim(),
          description: formDescription.trim(),
          body: formBody,
          scope: formScope,
          ...(formScope === 'project' && formProjectPath.trim() ? { projectPath: formProjectPath.trim() } : {}),
        });
      } else {
        await updateSkill({
          name: formName,
          description: formDescription.trim(),
          body: formBody,
        });
      }
      await loadSkills();
      if (viewMode === 'create') {
        const detail = await fetchSkill(formName.trim());
        setSelectedSkill(detail);
      } else {
        const detail = await fetchSkill(formName);
        setSelectedSkill(detail);
      }
      setViewMode('view');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (name: string) => {
    setSaving(true);
    try {
      await deleteSkill(name);
      await loadSkills();
      setSelectedSkill(null);
      setViewMode('list');
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete');
    } finally {
      setSaving(false);
    }
  };

  const filtered = skills.filter((s) => {
    if (scopeFilter !== 'all' && s.scope !== scopeFilter) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        s.name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        (s.tags ?? []).some((t) => t.toLowerCase().includes(q))
      );
    }
    return true;
  });

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
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
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
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '16px 20px',
            borderBottom: '1px solid var(--border)',
            flexShrink: 0,
          }}
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 4h5l2 2h5a1 1 0 011 1v8a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1z" />
            <path d="M8 11l2 2 4-4" />
          </svg>
          <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>
            Skills Manager
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
            + New Skill
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

        {/* Error banner */}
        {error && (
          <div
            style={{
              padding: '8px 20px',
              background: 'rgba(239,68,68,0.1)',
              borderBottom: '1px solid var(--danger)',
              color: 'var(--danger)',
              fontSize: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
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
          {/* Left: Skill list */}
          <div
            style={{
              width: 280,
              flexShrink: 0,
              borderRight: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Search */}
            <div style={{ padding: '12px 12px 8px' }}>
              <input
                type="text"
                placeholder="Search skills..."
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

            {/* Scope filter tabs */}
            <div
              style={{
                display: 'flex',
                gap: 2,
                padding: '0 12px 8px',
              }}
            >
              {(['all', 'user', 'project', 'builtin'] as ScopeFilter[]).map((scope) => (
                <button
                  key={scope}
                  onClick={() => setScopeFilter(scope)}
                  style={{
                    flex: 1,
                    padding: '4px 0',
                    background: scopeFilter === scope ? 'var(--bg-tertiary)' : 'transparent',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    color: scopeFilter === scope ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontSize: 10,
                    fontWeight: 500,
                    cursor: 'pointer',
                    textTransform: 'capitalize',
                  }}
                >
                  {scope}
                </button>
              ))}
            </div>

            {/* Skills list */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 8px' }}>
              {loading ? (
                <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  <span className="spinner" style={{ width: 16, height: 16, marginBottom: 8 }} />
                  <div>Loading skills...</div>
                </div>
              ) : filtered.length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  No skills found
                </div>
              ) : (
                filtered.map((skill) => (
                  <button
                    key={skill.name}
                    onClick={() => handleSelectSkill(skill.name)}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '10px 12px',
                      marginBottom: 2,
                      background:
                        selectedSkill?.name === skill.name ? 'var(--bg-tertiary)' : 'transparent',
                      border: 'none',
                      borderRadius: 'var(--radius-md)',
                      cursor: 'pointer',
                      borderLeft:
                        selectedSkill?.name === skill.name
                          ? '2px solid var(--accent)'
                          : '2px solid transparent',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={(e) => {
                      if (selectedSkill?.name !== skill.name)
                        e.currentTarget.style.background = 'var(--bg-hover)';
                    }}
                    onMouseLeave={(e) => {
                      if (selectedSkill?.name !== skill.name)
                        e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        marginBottom: 3,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 13,
                          fontWeight: 500,
                          color: 'var(--text-primary)',
                          fontFamily: 'var(--font-mono)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          flex: 1,
                        }}
                      >
                        {skill.name}
                      </span>
                      <Badge label={skill.scope} color={SCOPE_COLORS[skill.scope] ?? 'var(--text-muted)'} />
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: 'var(--text-muted)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {skill.description.split('\n')[0].slice(0, 80)}
                    </div>
                    <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                      <Badge
                        label={skill.lifecycle}
                        color={LIFECYCLE_COLORS[skill.lifecycle] ?? 'var(--text-muted)'}
                      />
                      {skill.category && (
                        <Badge label={skill.category} color="var(--text-secondary)" />
                      )}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Right: Detail/Editor */}
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {viewMode === 'list' && !selectedSkill && (
              <EmptyState onCreateNew={handleStartCreate} />
            )}

            {viewMode === 'view' && selectedSkill && (
              <SkillDetailView
                skill={selectedSkill}
                onEdit={handleStartEdit}
                onDelete={() => setDeleteConfirm(selectedSkill.name)}
              />
            )}

            {(viewMode === 'edit' || viewMode === 'create') && (
              <SkillEditor
                mode={viewMode}
                name={formName}
                description={formDescription}
                body={formBody}
                scope={formScope}
                projectPath={formProjectPath}
                userPath={userPath}
                saving={saving}
                onNameChange={setFormName}
                onDescriptionChange={setFormDescription}
                onBodyChange={setFormBody}
                onScopeChange={setFormScope}
                onProjectPathChange={setFormProjectPath}
                onSave={handleSave}
                onCancel={() => setViewMode(selectedSkill ? 'view' : 'list')}
              />
            )}
          </div>
        </div>

        {/* Delete confirmation */}
        {deleteConfirm && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(0,0,0,0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 10,
            }}
          >
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
                Delete Skill
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
                Are you sure you want to delete{' '}
                <strong style={{ color: 'var(--text-primary)' }}>{deleteConfirm}</strong>?
                This cannot be undone.
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
                  {saving ? 'Deleting...' : 'Delete'}
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
    <span
      style={{
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
      }}
    >
      {label}
    </span>
  );
}

function EmptyState({ onCreateNew }: { onCreateNew: () => void }) {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        color: 'var(--text-muted)',
      }}
    >
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.2" opacity={0.4}>
        <path d="M8 8h10l4 4h10a2 2 0 012 2v16a2 2 0 01-2 2H8a2 2 0 01-2-2V10a2 2 0 012-2z" />
        <path d="M16 22l4 4 8-8" />
      </svg>
      <div style={{ fontSize: 13 }}>Select a skill to view details</div>
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
        or create a new one
      </button>
    </div>
  );
}

function SkillDetailView({
  skill,
  onEdit,
  onDelete,
}: {
  skill: SkillDetail;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '12px 20px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontSize: 15,
            fontWeight: 600,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            flex: 1,
          }}
        >
          {skill.name}
        </span>
        {skill.scope !== 'builtin' && (
          <>
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
              Delete
            </button>
          </>
        )}
      </div>

      {/* Metadata */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-subtle)', flexShrink: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
          {skill.description}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          <MetaItem label="Scope" value={skill.scope} />
          <MetaItem label="Lifecycle" value={skill.lifecycle} />
          {skill.author && <MetaItem label="Author" value={skill.author} />}
          {skill.version && <MetaItem label="Version" value={skill.version} />}
          {skill.category && <MetaItem label="Category" value={skill.category} />}
        </div>
        {skill.tags && skill.tags.length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
            {skill.tags.map((t) => (
              <Badge key={t} label={t} color="var(--text-secondary)" />
            ))}
          </div>
        )}
      </div>

      {/* Body (Markdown source) */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px 20px' }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: 8,
          }}
        >
          SKILL.md Body
        </div>
        <pre
          style={{
            fontSize: 12,
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
            background: 'var(--bg-secondary)',
            padding: 16,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.6,
            margin: 0,
          }}
        >
          {skill.body}
        </pre>
      </div>
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        fontSize: 11,
      }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{label}:</span>
      <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function SkillEditor({
  mode,
  name,
  description,
  body,
  scope,
  projectPath,
  userPath,
  saving,
  onNameChange,
  onDescriptionChange,
  onBodyChange,
  onScopeChange,
  onProjectPathChange,
  onSave,
  onCancel,
}: {
  mode: 'edit' | 'create';
  name: string;
  description: string;
  body: string;
  scope: 'user' | 'project';
  projectPath: string;
  userPath: string;
  saving: boolean;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onBodyChange: (v: string) => void;
  onScopeChange: (v: 'user' | 'project') => void;
  onProjectPathChange: (v: string) => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const isValid = name.trim() && description.trim() && body.trim();

  return (
    <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '12px 20px',
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>
          {mode === 'create' ? 'Create New Skill' : `Edit: ${name}`}
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 640 }}>
          {/* Name */}
          {mode === 'create' && (
            <FormField label="Name" hint="kebab-case (e.g. my-skill)">
              <input
                type="text"
                value={name}
                onChange={(e) => onNameChange(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
                placeholder="my-skill-name"
                style={inputStyle}
              />
            </FormField>
          )}

          {/* Scope (create only) */}
          {mode === 'create' && (
            <FormField label="Scope" hint="Where to save this skill">
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
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                      {s === 'user' ? userPath || '~/.rune/skills/' : projectPath || '.rune/skills/'}
                    </div>
                  </button>
                ))}
              </div>
              {scope === 'project' && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                    Project skills path (change to save to a different project):
                  </div>
                  <input
                    type="text"
                    value={projectPath}
                    onChange={(e) => onProjectPathChange(e.target.value)}
                    placeholder="/path/to/project/.rune/skills"
                    style={{
                      ...inputStyle,
                      fontFamily: 'var(--font-mono)',
                      fontSize: 11,
                    }}
                  />
                </div>
              )}
            </FormField>
          )}

          {/* Description */}
          <FormField label="Description" hint="What the skill does + when to trigger">
            <textarea
              value={description}
              onChange={(e) => onDescriptionChange(e.target.value)}
              placeholder='Downloads organizer. Activates on "organize downloads", "clean up files"...'
              rows={3}
              style={{ ...inputStyle, resize: 'vertical', minHeight: 60 }}
            />
          </FormField>

          {/* Body */}
          <FormField label="Body (Markdown)" hint="Skill instructions — steps, examples, warnings">
            <textarea
              value={body}
              onChange={(e) => onBodyChange(e.target.value)}
              placeholder={'# My Skill\n\n## Steps\n1. First step\n2. Second step\n\n## Examples\n...'}
              rows={14}
              style={{
                ...inputStyle,
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                resize: 'vertical',
                minHeight: 200,
                lineHeight: 1.6,
              }}
            />
          </FormField>
        </div>
      </div>
    </div>
  );
}

function FormField({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
        <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{label}</label>
        {hint && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{hint}</span>
        )}
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
  fontFamily: 'var(--font-sans)',
  outline: 'none',
  boxSizing: 'border-box',
};
