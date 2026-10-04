import { useCallback, useEffect, useState } from 'react';
import {
  fetchCronJobs,
  createCronJob,
  updateCronJob,
  deleteCronJob,
  type CronJobInfo,
  type RoutinePolicy,
} from '../api';
import { RoutineFields, RoutineHistory, defaultRoutinePolicy } from './RoutineFields';
import { useFocusTrap } from '../hooks/useFocusTrap';

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
  const panelRef = useFocusTrap<HTMLDivElement>();
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
  const [taskType, setTaskType] = useState<'goal' | 'command'>('goal');
  const [policy, setPolicy] = useState<RoutinePolicy>(defaultRoutinePolicy);
  const [notifyChannel, setNotifyChannel] = useState('');
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
      setError(err instanceof Error ? err.message : "Couldn't load cron jobs");
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
    setTaskType('goal');
    setPolicy(defaultRoutinePolicy());
    setNotifyChannel('');
    setEnabled(true);
    setMaxRuns('');
    setError(null);
  };

  const selectJob = (job: CronJobInfo) => {
    setMode('edit');
    setSelectedId(job.id);
    setName(job.name);
    setSchedule(job.schedule);
    setCommand(job.goal || job.command);
    setTaskType(job.goal ? 'goal' : 'command');
    setPolicy(job.policy || defaultRoutinePolicy());
    setNotifyChannel(job.notifyChannel || '');
    setEnabled(job.enabled);
    setMaxRuns(job.maxRuns != null ? String(job.maxRuns) : '');
    setError(null);
  };

  const parseMaxRuns = (): number | undefined => {
    const trimmed = maxRuns.trim();
    if (!trimmed) return undefined;
    const parsed = Number(trimmed);
    if (!Number.isSafeInteger(parsed) || parsed < 1) {
      throw new Error('Max runs must be a positive integer');
    }
    return parsed;
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const maxRunsValue = parseMaxRuns();
      for (const [name, value, min, max] of [
        ['Time limit', policy.timeout_seconds, 1, 1800],
        ['Token limit', policy.token_budget, 1000, 500000],
        ['Step limit', policy.max_steps, 1, 200],
      ] as const) {
        if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${name} must be an integer from ${min} to ${max}`);
      }
      if (mode === 'create') {
        await createCronJob({
          name: name.trim(),
          schedule: schedule.trim(),
          command: taskType === 'command' ? command.trim() : '',
          goal: taskType === 'goal' ? command.trim() : '',
          policy,
          notifyChannel,
          enabled,
          ...(maxRunsValue !== undefined ? { maxRuns: maxRunsValue } : {}),
        });
      } else {
        if (!selectedId) throw new Error('No selected job');
        await updateCronJob({
          jobId: selectedId,
          name: name.trim(),
          schedule: schedule.trim(),
          command: taskType === 'command' ? command.trim() : '',
          goal: taskType === 'goal' ? command.trim() : '',
          policy,
          notifyChannel,
          enabled,
          maxRuns: maxRunsValue ?? null,
        });
      }
      await loadJobs();
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't save the cron job");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    // Use the job name so the confirmation identifies what will be deleted.
    const name = jobs.find((job) => job.id === selectedId)?.name || selectedId;
    const ok = window.confirm(`Delete the scheduled job "${name}"?`);
    if (!ok) return;

    setSaving(true);
    setError(null);
    try {
      await deleteCronJob(selectedId);
      await loadJobs();
      resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't delete the cron job");
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
        role="dialog"
        aria-modal="true"
        aria-label="Scheduled jobs"
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
            Scheduled tasks
          </span>
          <span style={{
            fontSize: 11,
            color: heartbeatActive ? 'var(--success)' : 'var(--text-muted)',
            background: heartbeatActive ? 'var(--success-subtle)' : 'var(--bg-secondary)',
            border: `1px solid ${heartbeatActive ? 'var(--success)' : 'var(--border)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '2px 8px',
          }}>
            {heartbeatActive ? 'Scheduler connected' : 'Runs while Rune is running'}
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
            + New task
          </button>
          <button
            onClick={onClose}
            aria-label="Close"
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

        <div className="routine-layout" style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <div className="routine-list" style={{
            width: 'clamp(180px, 28vw, 300px)',
            flexShrink: 0,
            borderRight: '1px solid var(--border)',
            overflowY: 'auto',
            minHeight: 0,
          }}>
            {loading ? (
              <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Loading...</div>
            ) : jobs.length === 0 ? (
              <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                No scheduled tasks yet.
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
              {mode === 'create' ? 'New scheduled task' : 'Edit scheduled task'}
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
                aria-label="Task name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="GeekNews 1-minute briefing"
                style={inputStyle}
              />
            </LabeledField>

            <LabeledField label="Schedule">
              <input
                aria-label="Schedule"
                value={schedule}
                onChange={(e) => setSchedule(e.target.value)}
                placeholder='0 9 * * 1-5'
                style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }}
              />
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                Five cron fields in Rune’s local time zone. Example: 0 9 * * 1-5 runs weekdays at 9 AM.
              </div>
            </LabeledField>

            <LabeledField label="Task type">
              <select aria-label="Task type" value={taskType} onChange={e => setTaskType(e.target.value as 'goal' | 'command')} style={inputStyle}>
                <option value="goal">Agent task</option><option value="command">Shell command</option>
              </select>
            </LabeledField>
            <LabeledField label={taskType === 'goal' ? 'What should Rune do?' : 'Shell command'}>
              <textarea
                aria-label={taskType === 'goal' ? 'Agent task' : 'Shell command'}
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

            <RoutineFields policy={policy} onChange={setPolicy} />
            <LabeledField label="Notification channel (optional)">
              <input aria-label="Notification channel" value={notifyChannel} onChange={e => setNotifyChannel(e.target.value)} placeholder="None; results stay here" style={inputStyle} />
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>Use a connected channel name, such as telegram or slack, to send results to its configured recipient.</div>
            </LabeledField>
            {selectedJob && <RoutineHistory key={selectedJob.id} job={selectedJob} onRefresh={loadJobs} />}
            <div style={{ display: 'flex', gap: 12 }}>
              <LabeledField label="Max Runs (optional)" style={{ flex: 1 }}>
                <input
                  aria-label="Maximum runs"
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
                  {enabled ? 'Enabled' : 'Paused'}
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
