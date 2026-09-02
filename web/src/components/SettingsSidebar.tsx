import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchSkills,
  fetchEnvVars,
  fetchConfig,
  patchConfig,
  fetchChannels,
  fetchCronJobs,
  restartChannel,
  type SkillInfo,
  type EnvVarInfo,
  type ConfigInfo,
  type ChannelInfo,
} from '../api';
import { toast } from '../utils/toast';

interface SettingsSidebarProps {
  onOpenSkillPanel: (selectedName?: string) => void;
  onOpenEnvPanel: () => void;
  onOpenCronPanel: () => void;
  onOpenMcpPanel: () => void;
  onOpenMarkdownPanel?: () => void;
}

const CHANNEL_STATUS_COLORS: Record<string, string> = {
  connected: 'var(--success)',
  connecting: 'var(--warning)',
  disconnected: 'var(--text-muted)',
  error: 'var(--danger)',
};

// Matches RolloutManager._VALID_MODES; 'auto' was offered here and rejected by the API.
const MEMORY_POLICY_MODES = ['legacy', 'shadow', 'balanced', 'strict'] as const;
const MEMORY_PRESET_VALUES = {
  speed: {
    policyMode: 'shadow',
    semanticLimit: '2',
    semanticMinScore: '0.5',
    uncertainSemanticLimit: '3',
    uncertainSemanticMinScore: '0.4',
    maxEpisodes: '3',
    contextMaxChars: '3000',
  },
  balanced: {
    policyMode: 'balanced',
    semanticLimit: '5',
    semanticMinScore: '0.3',
    uncertainSemanticLimit: '8',
    uncertainSemanticMinScore: '0.2',
    maxEpisodes: '10',
    contextMaxChars: '8000',
  },
  accuracy: {
    policyMode: 'strict',
    semanticLimit: '10',
    semanticMinScore: '0.15',
    uncertainSemanticLimit: '15',
    uncertainSemanticMinScore: '0.1',
    maxEpisodes: '20',
    contextMaxChars: '16000',
  },
} as const;
const MEMORY_PRESET_ORDER = ['speed', 'balanced', 'accuracy'] as const;
const SAFETY_PRESET_ORDER = ['conservative', 'balanced', 'developer'] as const;
const SAFETY_PRESET_HINTS: Record<typeof SAFETY_PRESET_ORDER[number], string> = {
  conservative: 'strict lock',
  balanced: 'auto rollout',
  developer: 'balanced lock',
};

type MemoryPolicyMode = typeof MEMORY_POLICY_MODES[number];
type MemoryPreset = typeof MEMORY_PRESET_ORDER[number];
type MemoryDraftPreset = MemoryPreset | 'custom';
type SafetyPreset = typeof SAFETY_PRESET_ORDER[number];

interface MemoryTuningDraft {
  preset: MemoryDraftPreset;
  policyMode: MemoryPolicyMode;
  semanticLimit: string;
  semanticMinScore: string;
  uncertainSemanticLimit: string;
  uncertainSemanticMinScore: string;
  maxEpisodes: string;
  contextMaxChars: string;
}

function toMemoryTuningDraft(config: ConfigInfo): MemoryTuningDraft {
  return {
    preset: config.memoryTuning.preset ?? 'custom',
    policyMode: config.memoryTuning.policyMode,
    semanticLimit: String(config.memoryTuning.semanticLimit),
    semanticMinScore: String(config.memoryTuning.semanticMinScore),
    uncertainSemanticLimit: String(config.memoryTuning.uncertainSemanticLimit),
    uncertainSemanticMinScore: String(config.memoryTuning.uncertainSemanticMinScore),
    maxEpisodes: String(config.memoryTuning.maxEpisodes),
    contextMaxChars: String(config.memoryTuning.contextMaxChars),
  };
}

export function SettingsSidebar({ onOpenSkillPanel, onOpenEnvPanel, onOpenCronPanel, onOpenMcpPanel, onOpenMarkdownPanel }: SettingsSidebarProps) {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [envVars, setEnvVars] = useState<EnvVarInfo[]>([]);
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [cronCount, setCronCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);
  const [restarting, setRestarting] = useState<string | null>(null);
  const [memoryDraft, setMemoryDraft] = useState<MemoryTuningDraft | null>(null);
  // The 30s refresh must not overwrite fields the user is still filling in.
  const memoryDirtyRef = useRef(false);
  // Wrap the setter so every field edit flags the draft; load() then leaves it alone.
  const editMemoryDraft: typeof setMemoryDraft = (v) => { memoryDirtyRef.current = true; setMemoryDraft(v); };
  const [savingMemory, setSavingMemory] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [selectedSafetyPreset, setSelectedSafetyPreset] = useState<SafetyPreset | null>(null);

  const load = useCallback(async () => {
    try {
      const [sk, ev, cfg, ch, cron] = await Promise.allSettled([
        fetchSkills(),
        fetchEnvVars(),
        fetchConfig(),
        fetchChannels(),
        fetchCronJobs(),
      ]);
      if (sk.status === 'fulfilled') setSkills(sk.value.skills);
      if (ev.status === 'fulfilled') setEnvVars(ev.value.variables);
      if (cfg.status === 'fulfilled') {
        setConfig(cfg.value);
        if (!memoryDirtyRef.current) setMemoryDraft(toMemoryTuningDraft(cfg.value));
        setSelectedSafetyPreset(cfg.value.safetyTuning.preset);
      }
      if (ch.status === 'fulfilled') setChannels(ch.value.channels);
      if (cron.status === 'fulfilled') setCronCount(cron.value.jobs.length);
    } catch {
      // keep empty
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(load, 30000);
    return () => clearInterval(timer);
  }, [load]);

  const handleToggleProactive = async () => {
    if (!config) return;
    setToggling(true);
    try {
      await patchConfig({ proactiveEnabled: !config.proactiveEnabled });
      const updated = await fetchConfig();
      setConfig(updated);
      setMemoryDraft(toMemoryTuningDraft(updated));
      setSelectedSafetyPreset(updated.safetyTuning.preset);
    } catch {
      toast.error("Couldn't change the proactive setting");
    } finally {
      setToggling(false);
    }
  };

  const handleToggleAdvisor = async () => {
    if (!config) return;
    setToggling(true);
    try {
      await patchConfig({ advisorEnabled: !config.advisorEnabled });
      const updated = await fetchConfig();
      setConfig(updated);
      setMemoryDraft(toMemoryTuningDraft(updated));
      setSelectedSafetyPreset(updated.safetyTuning.preset);
    } catch {
      toast.error("Couldn't change the advisor setting");
    } finally {
      setToggling(false);
    }
  };

  const handleSaveMemoryTuning = async () => {
    if (!memoryDraft) return;
    setSavingMemory(true);
    setMemoryError(null);
    try {
      const semanticLimit = Number.parseInt(memoryDraft.semanticLimit, 10);
      const semanticMinScore = Number(memoryDraft.semanticMinScore);
      const uncertainSemanticLimit = Number.parseInt(memoryDraft.uncertainSemanticLimit, 10);
      const uncertainSemanticMinScore = Number(memoryDraft.uncertainSemanticMinScore);
      const maxEpisodes = Number.parseInt(memoryDraft.maxEpisodes, 10);
      const contextMaxChars = Number.parseInt(memoryDraft.contextMaxChars, 10);

      if (!Number.isInteger(semanticLimit)) throw new Error('Semantic limit must be a whole number');
      if (!Number.isFinite(semanticMinScore)) throw new Error('Semantic min score must be a number');
      if (!Number.isInteger(uncertainSemanticLimit)) throw new Error('Uncertain semantic limit must be a whole number');
      if (!Number.isFinite(uncertainSemanticMinScore)) throw new Error('Uncertain semantic min score must be a number');
      if (!Number.isInteger(maxEpisodes)) throw new Error('Max episodes must be a whole number');
      if (!Number.isInteger(contextMaxChars)) throw new Error('Context max chars must be a whole number');

      await patchConfig({
        memoryTuning: {
          scope: 'project',
          ...(memoryDraft.preset !== 'custom' ? { preset: memoryDraft.preset } : {}),
          policyMode: memoryDraft.policyMode,
          semanticLimit,
          semanticMinScore,
          uncertainSemanticLimit,
          uncertainSemanticMinScore,
          maxEpisodes,
          contextMaxChars,
        },
      });
      const updated = await fetchConfig();
      setConfig(updated);
      memoryDirtyRef.current = false;   // saved — the refresh may take over again
      setMemoryDraft(toMemoryTuningDraft(updated));
      setSelectedSafetyPreset(updated.safetyTuning.preset);
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : "Couldn't save memory tuning");
    } finally {
      setSavingMemory(false);
    }
  };

  const applyPreset = (preset: MemoryPreset) => {
    editMemoryDraft((prev) => {
      if (!prev) return prev;
      const values = MEMORY_PRESET_VALUES[preset];
      return {
        ...prev,
        preset,
        policyMode: values.policyMode,
        semanticLimit: values.semanticLimit,
        semanticMinScore: values.semanticMinScore,
        uncertainSemanticLimit: values.uncertainSemanticLimit,
        uncertainSemanticMinScore: values.uncertainSemanticMinScore,
        maxEpisodes: values.maxEpisodes,
        contextMaxChars: values.contextMaxChars,
      };
    });
  };

  const handleRestart = async (name: string) => {
    setRestarting(name);
    try {
      await restartChannel(name);
      await new Promise((r) => setTimeout(r, 1000));
      const ch = await fetchChannels();
      setChannels(ch.channels);
    } catch {
      // ignore
    } finally {
      setRestarting(null);
    }
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--bg-primary)',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-muted)',
        fontSize: 12,
      }}>
        Loading...
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg-primary)',
    }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>

        {/* Proactive toggle */}
        {config && (
          <div style={{ padding: '6px 14px', marginBottom: 4 }}>
            <button
              onClick={handleToggleProactive}
              disabled={toggling}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                width: '100%',
                padding: '10px 12px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                cursor: toggling ? 'wait' : 'pointer',
                textAlign: 'left',
                opacity: toggling ? 0.6 : 1,
                transition: 'opacity 0.15s',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke={config.proactiveEnabled ? 'var(--accent)' : 'var(--text-muted)'} strokeWidth="1.3" strokeLinecap="round">
                <path d="M7 1.5v1M7 11.5v1M1.5 7h1M11.5 7h1M3.2 3.2l.7.7M10.1 10.1l.7.7M3.2 10.8l.7-.7M10.1 3.9l.7-.7" />
                <circle cx="7" cy="7" r="2.5" />
              </svg>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Proactive
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>
                  Autonomous suggestions
                </div>
              </div>
              <div style={{
                width: 32,
                height: 18,
                borderRadius: 9,
                background: config.proactiveEnabled ? 'var(--accent)' : 'var(--bg-tertiary)',
                position: 'relative',
                transition: 'background 0.2s',
                flexShrink: 0,
              }}>
                <div style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: 'white',
                  position: 'absolute',
                  top: 2,
                  left: config.proactiveEnabled ? 16 : 2,
                  transition: 'left 0.2s',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                }} />
              </div>
            </button>
          </div>
        )}

        {/* Advisor toggle */}
        {config && (
          <div style={{ padding: '6px 14px', marginBottom: 4 }}>
            <button
              onClick={handleToggleAdvisor}
              disabled={toggling}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                width: '100%',
                padding: '10px 12px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                cursor: toggling ? 'wait' : 'pointer',
                textAlign: 'left',
                opacity: toggling ? 0.6 : 1,
                transition: 'opacity 0.15s',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke={config.advisorEnabled ? 'var(--accent)' : 'var(--text-muted)'} strokeWidth="1.3" strokeLinecap="round">
                <path d="M7 1.5 L12 4.5 L12 9.5 L7 12.5 L2 9.5 L2 4.5 Z" />
                <path d="M7 5 L7 9 M5 7 L9 7" />
              </svg>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Advisor
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>
                  Stronger-model guidance at key moments
                </div>
              </div>
              <div style={{
                width: 32,
                height: 18,
                borderRadius: 9,
                background: config.advisorEnabled ? 'var(--accent)' : 'var(--bg-tertiary)',
                position: 'relative',
                transition: 'background 0.2s',
                flexShrink: 0,
              }}>
                <div style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: 'white',
                  position: 'absolute',
                  top: 2,
                  left: config.advisorEnabled ? 16 : 2,
                  transition: 'left 0.2s',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                }} />
              </div>
            </button>
          </div>
        )}

        {config && memoryDraft && (
          <div style={{ padding: '6px 14px', marginBottom: 6 }}>
            <div style={{
              padding: '10px 12px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                Memory Tuning
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2, marginBottom: 10 }}>
                Scope: project (.rune/.env)
              </div>

              <div style={{ display: 'grid', gap: 8 }}>
                <div style={{ display: 'grid', gap: 6 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Preset</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {MEMORY_PRESET_ORDER.map((preset) => {
                      const active = memoryDraft.preset === preset;
                      return (
                        <button
                          key={preset}
                          onClick={() => applyPreset(preset)}
                          style={{
                            flex: 1,
                            padding: '4px 6px',
                            borderRadius: 'var(--radius-sm)',
                            border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                            background: active ? 'var(--accent-subtle)' : 'var(--bg-tertiary)',
                            color: active ? 'var(--accent)' : 'var(--text-secondary)',
                            fontSize: 10,
                            fontWeight: 600,
                            cursor: 'pointer',
                            textTransform: 'capitalize',
                          }}
                        >
                          {preset}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <label style={{ display: 'grid', gap: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Policy Mode</span>
                  <select
                    value={memoryDraft.policyMode}
                    onChange={(e) => {
                      const next = e.target.value as MemoryPolicyMode;
                      editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', policyMode: next } : prev));
                    }}
                    style={memoryInputStyle}
                  >
                    {MEMORY_POLICY_MODES.map((mode) => (
                      <option key={mode} value={mode}>{mode}</option>
                    ))}
                  </select>
                </label>

                <MemoryField
                  label="Semantic Limit (1~20)"
                  value={memoryDraft.semanticLimit}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', semanticLimit: value } : prev))}
                />
                <MemoryField
                  label="Semantic Min Score (0~1)"
                  value={memoryDraft.semanticMinScore}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', semanticMinScore: value } : prev))}
                />
                <MemoryField
                  label="Uncertain Semantic Limit (1~20)"
                  value={memoryDraft.uncertainSemanticLimit}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', uncertainSemanticLimit: value } : prev))}
                />
                <MemoryField
                  label="Uncertain Semantic Min Score (0~1)"
                  value={memoryDraft.uncertainSemanticMinScore}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', uncertainSemanticMinScore: value } : prev))}
                />
                <MemoryField
                  label="Max Episodes (1~50)"
                  value={memoryDraft.maxEpisodes}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', maxEpisodes: value } : prev))}
                />
                <MemoryField
                  label="Context Max Chars (1000~32000)"
                  value={memoryDraft.contextMaxChars}
                  onChange={(value) => editMemoryDraft((prev) => (prev ? { ...prev, preset: 'custom', contextMaxChars: value } : prev))}
                />
              </div>

              {memoryError && (
                <div style={{ marginTop: 8, fontSize: 10, color: 'var(--danger)' }}>
                  {memoryError}
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
                <button
                  onClick={handleSaveMemoryTuning}
                  disabled={savingMemory}
                  style={{
                    padding: '4px 10px',
                    background: savingMemory ? 'var(--bg-tertiary)' : 'var(--accent)',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    color: savingMemory ? 'var(--text-muted)' : 'white',
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: savingMemory ? 'wait' : 'pointer',
                  }}
                >
                  {savingMemory ? 'Saving...' : 'Save Memory Tuning'}
                </button>
              </div>
            </div>
          </div>
        )}

        {config && (
          <div style={{ padding: '6px 14px', marginBottom: 6 }}>
            <div style={{
              padding: '10px 12px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                Safety Tuning
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2, marginBottom: 10 }}>
                Current mode: {config.safetyTuning.rolloutMode}
              </div>

              <div style={{ display: 'grid', gap: 6 }}>
                {SAFETY_PRESET_ORDER.map((preset) => {
                  const active = selectedSafetyPreset === preset;
                  return (
                    <button
                      key={preset}
                      disabled
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 8,
                        padding: '6px 8px',
                        borderRadius: 'var(--radius-sm)',
                        border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                        background: active ? 'var(--accent-subtle)' : 'var(--bg-tertiary)',
                        color: active ? 'var(--accent)' : 'var(--text-secondary)',
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'not-allowed',
                        textTransform: 'capitalize',
                        opacity: 0.5,
                      }}
                    >
                      <span>{preset}</span>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
                        {SAFETY_PRESET_HINTS[preset]}
                      </span>
                    </button>
                  );
                })}
              </div>

              <div style={{ marginTop: 8, fontSize: 9, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Read-only. The shell gate builds its own policy and does not read
                this preset yet, so switching it here would change nothing.
              </div>
            </div>
          </div>
        )}

        {/* Channels */}
        {channels.length > 0 && (
          <div style={{ marginBottom: 4 }}>
            <div style={{
              padding: '8px 14px 4px',
              fontSize: 10,
              fontWeight: 600,
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}>
              Channels
            </div>
            {channels.map((ch) => (
              <div
                key={ch.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '7px 14px',
                }}
              >
                <span style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: CHANNEL_STATUS_COLORS[ch.status] ?? 'var(--text-muted)',
                  flexShrink: 0,
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 500,
                    color: 'var(--text-primary)',
                    textTransform: 'capitalize',
                  }}>
                    {ch.name}
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                    {ch.status} · {ch.sessionCount} session{ch.sessionCount !== 1 ? 's' : ''}
                  </div>
                </div>
                {ch.status === 'error' || ch.status === 'disconnected' ? (
                  <button
                    onClick={() => handleRestart(ch.name)}
                    disabled={restarting === ch.name}
                    style={{
                      padding: '2px 8px',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      color: 'var(--text-secondary)',
                      fontSize: 9,
                      cursor: restarting === ch.name ? 'wait' : 'pointer',
                      opacity: restarting === ch.name ? 0.6 : 1,
                    }}
                  >
                    {restarting === ch.name ? '...' : 'Restart'}
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        )}

        {/* Divider */}
        {(config || channels.length > 0) && (
          <div style={{
            height: 1,
            background: 'var(--border)',
            margin: '4px 14px 8px',
          }} />
        )}

        {/* Skills section */}
        <SettingsSection
          icon={
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2.5 2.5h3.5l1.25 1.25H11a.75.75 0 01.75.75v6a.75.75 0 01-.75.75H2.5a.75.75 0 01-.75-.75v-7.25a.75.75 0 01.75-.75z" />
              <path d="M5.5 7.5l1.25 1.25L9.5 6" />
            </svg>
          }
          title="Skills"
          subtitle={`${skills.length} registered`}
          onClick={() => onOpenSkillPanel()}
        >
          {skills.slice(0, 5).map((s) => (
            <SettingsItem
              key={s.name}
              label={s.name}
              detail={s.scope}
              detailColor={s.scope === 'user' ? 'var(--accent)' : s.scope === 'project' ? 'var(--success)' : 'var(--text-muted)'}
              onClick={() => onOpenSkillPanel(s.name)}
              mono
            />
          ))}
          {skills.length > 5 && (
            <button
              onClick={() => onOpenSkillPanel()}
              style={{
                display: 'block',
                width: '100%',
                padding: '4px 14px 4px 36px',
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: 10,
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              +{skills.length - 5} more...
            </button>
          )}
        </SettingsSection>

        {/* Environment section */}
        <SettingsSection
          icon={
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3">
              <rect x="2" y="6.5" width="10" height="5.5" rx="1" />
              <path d="M4 6.5V4.5a3 3 0 016 0v2" />
            </svg>
          }
          title="Environment"
          subtitle={`${envVars.length} variables`}
          onClick={onOpenEnvPanel}
        >
          {envVars.slice(0, 5).map((v) => (
            <SettingsItem
              key={`${v.key}-${v.scope}`}
              label={v.key}
              detail={v.maskedValue.length > 16 ? v.maskedValue.slice(0, 16) + '...' : v.maskedValue}
              onClick={onOpenEnvPanel}
              mono
            />
          ))}
          {envVars.length > 5 && (
            <button
              onClick={onOpenEnvPanel}
              style={{
                display: 'block',
                width: '100%',
                padding: '4px 14px 4px 36px',
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: 10,
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              +{envVars.length - 5} more...
            </button>
          )}
        </SettingsSection>

        {/* Automation section */}
        <SettingsSection
          icon={
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
              <circle cx="7" cy="7" r="5" />
              <path d="M7 4.5V7l1.7 1" />
            </svg>
          }
          title="Automation"
          subtitle={`${cronCount} jobs`}
          onClick={onOpenCronPanel}
        >
          <SettingsItem
            label="Cron Jobs"
            detail={cronCount > 0 ? String(cronCount) : 'none'}
            onClick={onOpenCronPanel}
          />
        </SettingsSection>

        {/* MCP Section */}
        <SettingsSection
          icon={
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1Z" stroke="currentColor" strokeWidth="1.2"/>
              <circle cx="8" cy="8" r="2" stroke="currentColor" strokeWidth="1.2"/>
            </svg>
          }
          title="MCP Servers"
          subtitle="External tools"
          onClick={onOpenMcpPanel}
        >
          <SettingsItem
            label="Manage Servers"
            detail="configure"
            onClick={onOpenMcpPanel}
          />
        </SettingsSection>

        {/* Config Files (Markdown Editor) */}
        {onOpenMarkdownPanel && (
          <SettingsSection
            icon={
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M3 2h7l3 3v9a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2"/>
                <path d="M10 2v3h3" stroke="currentColor" strokeWidth="1.2"/>
                <path d="M5 8h6M5 11h4" stroke="currentColor" strokeWidth="1.2"/>
              </svg>
            }
            title="Config Files"
            subtitle="Heartbeat, Memory, Profile"
            onClick={onOpenMarkdownPanel}
          >
            <SettingsItem
              label="Edit Files"
              detail="markdown"
              onClick={onOpenMarkdownPanel}
            />
          </SettingsSection>
        )}

      </div>
    </div>
  );
}

// ── Sub-components ──

function SettingsSection({
  icon,
  title,
  subtitle,
  onClick,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  onClick: () => void;
  children?: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 4 }}>
      {/* Section header */}
      <button
        onClick={onClick}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          padding: '10px 14px',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          transition: 'background 0.1s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-hover)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
      >
        <span style={{ color: 'var(--text-muted)', display: 'flex', flexShrink: 0 }}>{icon}</span>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', flex: 1 }}>{title}</span>
        {subtitle && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{subtitle}</span>
        )}
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="var(--text-muted)" strokeWidth="1.3" strokeLinecap="round">
          <path d="M3.5 2L6.5 5L3.5 8" />
        </svg>
      </button>
      {/* Section items */}
      {children}
    </div>
  );
}

function SettingsItem({
  label,
  detail,
  detailColor,
  onClick,
  mono,
}: {
  label: string;
  detail?: string;
  detailColor?: string;
  onClick: () => void;
  mono?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        width: '100%',
        padding: '5px 14px 5px 36px',
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'background 0.1s',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-hover)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
    >
      <span style={{
        fontSize: 11,
        color: 'var(--text-secondary)',
        flex: 1,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-sans)',
      }}>
        {label}
      </span>
      {detail && (
        <span style={{
          fontSize: 9,
          color: detailColor ?? 'var(--text-muted)',
          flexShrink: 0,
          textTransform: 'capitalize',
        }}>
          {detail}
        </span>
      )}
    </button>
  );
}

function MemoryField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label style={{ display: 'grid', gap: 4 }}>
      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={memoryInputStyle}
      />
    </label>
  );
}

const memoryInputStyle: React.CSSProperties = {
  width: '100%',
  padding: '5px 8px',
  borderRadius: 'var(--radius-sm)',
  border: '1px solid var(--border)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-primary)',
  fontSize: 11,
  fontFamily: 'var(--font-mono)',
};
