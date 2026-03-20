import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchCronJobs,
  createCronJob,
  updateCronJob,
  deleteCronJob,
  type CronJobInfo,
} from '../api';

interface CronPanelProps {
  onClose: () => void;
}

type FormMode = 'create' | 'edit';

function formatDateTime(iso?: string): string {
  if (!iso) return 'never';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'invalid';
  return d.toLocaleString();
}

export function CronPanel({ onClose }: CronPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [jobs, setJobs] = useState<CronJobInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [heartbeatActive, setHeartbeatActive] = useState(true);

  const [mode, setMode] = useState<FormMode>('create');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [name, setName] = useState('');
  const [schedule, setSchedule] = useState('');
  const [command, setCommand] = useState('');
  const [enabled, setEnabled] = useState(true);
  const [maxRuns, setMaxRuns] = useState('');

  const loadJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchCronJobs();
      const sorted = result.jobs
        .slice()
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
      setJobs(sorted);
      setHeartbeatActive(result.heartbeatActive);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cron jobs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const resetForm = () => {
    setMode('create');
    setSelectedId(null);
    setName('');
    setSchedule('');
    setCommand('');
    setEnabled(true);
    setMaxRuns('');
    setError(null);
  };

  const selectJob = (job: CronJobInfo) => {
    setMode('edit');
    setSelectedId(job.id);
    setName(job.name);
    setSchedule(job.schedule);
    setCommand(job.command);
    setEnabled(job.enabled);
    setMaxRuns(job.maxRuns !== undefined ? String(job.maxRuns) : '');
    setError(null);
  };

  const parseMaxRuns = (): number | undefined => {
    const trimmed = maxRuns.trim();
    if (!trimmed) return undefined;
    const parsed = Number.parseInt(trimmed, 10);
    if (!Number.isFinite(parsed) || parsed < 1) {
      throw new Error('Max runs must be a positive integer');
    }
    return parsed;
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const maxRunsValue = parseMaxRuns();
      if (mode === 'create') {
        await createCronJob({
          name: name.trim(),
          schedule: schedule.trim(),
          command: command.trim(),
          enabled,
          ...(maxRunsValue !== undefined ? { maxRuns: maxRunsValue } : {}),
        });
      } else {
        if (!selectedId) throw new Error('No selected job');
        await updateCronJob({
          jobId: selectedId,
          name: name.trim(),
          schedule: schedule.trim(),
          command: command.trim(),
          enabled,
          ...(maxRunsValue !== undefined ? { maxRuns: maxRunsValue } : {}),
        });
      }
      await loadJobs();
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save cron job');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    const ok = window.confirm(`Delete cron job "${selectedId}"?`);
    if (!ok) return;

    setSaving(true);
    setError(null);
    try {
      await deleteCronJob(selectedId);
      await loadJobs();
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete cron job');
    } finally {
      setSaving(false);
    }
  };

  const selectedJob = selectedId
    ? jobs.find((job) => job.id === selectedId) ?? null
    : null;

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
          width: '92vw',
          maxWidth: 1080,
          height: '82vh',
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
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
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round">
            <circle cx="9" cy="9" r="6.5" />
            <path d="M9 5.2V9l2.8 1.8" />
          </svg>
          <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-primary)' }}>
            Cron Jobs
          </span>
          <span style={{
            fontSize: 11,
            color: heartbeatActive ? 'var(--success)' : 'var(--danger)',
            background: heartbeatActive ? 'var(--success-subtle)' : 'var(--danger-subtle)',
            border: `1px solid ${heartbeatActive ? 'var(--success)' : 'var(--danger)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '2px 8px',
          }}>
            {heartbeatActive ? 'Heartbeat active' : 'Heartbeat inactive'}
          </span>
          <div style={{ flex: 1 }} />
          <button
            onClick={loadJobs}
            disabled={loading}
            style={secondaryButtonStyle}
          >
            Refresh
          </button>
          <button
            onClick={resetForm}
            style={primaryButtonStyle}
          >
            + New Job
          </button>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: 18,
              cursor: 'pointer',
              lineHeight: 1,
              padding: '4px 8px',
            }}
          >
            ×
          </button>
        </div>

        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <div style={{
            width: 360,
            borderRight: '1px solid var(--border)',
            overflowY: 'auto',
            minHeight: 0,
          }}>
            {loading ? (
              <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Loading...</div>
            ) : jobs.length === 0 ? (
              <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                No cron jobs yet.
              </div>
            ) : (
              jobs.map((job) => {
                const selected = job.id === selectedId;
                return (
                  <button
                    key={job.id}
                    onClick={() => selectJob(job)}
                    style={{
                      width: '100%',
                      border: 'none',
                      borderBottom: '1px solid var(--border-subtle)',
                      padding: '12px 14px',
                      textAlign: 'left',
                      background: selected ? 'var(--bg-tertiary)' : 'transparent',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: job.enabled ? 'var(--success)' : 'var(--text-muted)',
                      }} />
                      <span style={{
                        fontSize: 12,
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {job.name}
                      </span>
                    </div>
                    <div style={{
                      marginTop: 4,
                      fontSize: 10,
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-mono)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>
                      {job.schedule}
                    </div>
                    <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-muted)' }}>
                      runs {job.runCount}
                      {job.maxRuns ? ` / ${job.maxRuns}` : ''}
                      {' · '}
                      last {formatDateTime(job.lastRunAt)}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          <div style={{
            flex: 1,
            minWidth: 0,
            overflowY: 'auto',
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              {mode === 'create' ? 'Create Cron Job' : `Edit Cron Job (${selectedId})`}
            </div>

            {error && (
              <div style={{
                border: '1px solid var(--danger)',
                background: 'var(--danger-subtle)',
                color: 'var(--danger)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 12px',
                fontSize: 12,
              }}>
                {error}
              </div>
            )}

            <LabeledField label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="GeekNews 1-minute briefing"
                style={inputStyle}
              />
            </LabeledField>

            <LabeledField label="Schedule">
              <input
                value={schedule}
                onChange={(e) => setSchedule(e.target.value)}
                placeholder='*/1 * * * * or "every morning at 9"'
                style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }}
              />
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                Presets: <code>@daily</code>, <code>@hourly</code>, <code>@weekdays</code>, <code>@morning</code>
              </div>
            </LabeledField>

            <LabeledField label="Command">
              <textarea
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                placeholder="Check https://news.hada.io/new and generate a briefing for new posts"
                style={{
                  ...inputStyle,
                  minHeight: 160,
                  resize: 'vertical',
                  lineHeight: 1.5,
                  fontFamily: 'var(--font-sans)',
                }}
              />
            </LabeledField>

            <div style={{ display: 'flex', gap: 12 }}>
              <LabeledField label="Max Runs (optional)" style={{ flex: 1 }}>
                <input
                  value={maxRuns}
                  onChange={(e) => setMaxRuns(e.target.value)}
                  placeholder="Unlimited if empty"
                  style={inputStyle}
                />
              </LabeledField>
              <LabeledField label="Enabled" style={{ width: 180 }}>
                <label style={{
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  color: 'var(--text-secondary)',
                  fontSize: 12,
                }}>
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={(e) => setEnabled(e.target.checked)}
                  />
                  {enabled ? 'enabled' : 'disabled'}
                </label>
              </LabeledField>
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button
                onClick={handleSave}
                disabled={saving}
                style={primaryButtonStyle}
              >
                {saving ? 'Saving...' : mode === 'create' ? 'Create' : 'Save Changes'}
              </button>
              {mode === 'edit' && (
                <button
                  onClick={handleDelete}
                  disabled={saving}
                  style={dangerButtonStyle}
                >
                  Delete
                </button>
              )}
              {mode === 'edit' && selectedJob && (
                <button
                  onClick={resetForm}
                  disabled={saving}
                  style={secondaryButtonStyle}
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LabeledField({
  label,
  children,
  style,
}: {
  label: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div style={style}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  boxSizing: 'border-box',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  background: 'var(--bg-secondary)',
  color: 'var(--text-primary)',
  fontSize: 13,
  padding: '8px 10px',
};

const primaryButtonStyle: React.CSSProperties = {
  height: 34,
  padding: '0 14px',
  borderRadius: 'var(--radius-md)',
  border: '1px solid var(--accent)',
  background: 'var(--accent)',
  color: 'white',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
};

const secondaryButtonStyle: React.CSSProperties = {
  height: 34,
  padding: '0 14px',
  borderRadius: 'var(--radius-md)',
  border: '1px solid var(--border)',
  background: 'var(--bg-secondary)',
  color: 'var(--text-primary)',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
};

const dangerButtonStyle: React.CSSProperties = {
  height: 34,
  padding: '0 14px',
  borderRadius: 'var(--radius-md)',
  border: '1px solid var(--danger)',
  background: 'var(--danger-subtle)',
  color: 'var(--danger)',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
};
