import { useEffect, useMemo, useState } from 'react';
import { useAgent } from './hooks/useAgent';
import { useSessionHistory } from './hooks/useSessionHistory';
import { ChatPanel } from './components/ChatPanel';
import { SessionSidebar } from './components/SessionSidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import { StatusBar } from './components/StatusBar';
import { InputArea } from './components/InputArea';
import { SkillsPanel } from './components/SkillsPanel';
import { EnvPanel } from './components/EnvPanel';
import { CronPanel } from './components/CronPanel';
import { MCPPanel } from './components/MCPPanel';
import { MarkdownPanel } from './components/MarkdownPanel';
import { WorkbenchPanel } from './components/WorkbenchPanel';
import { CommandK, type Command } from './components/CommandK';
import { normalizeToolName, inferWorkPhase } from './utils/tooling';
import { fetchConfig, fetchSessions, type ConfigInfo, type SessionInfo } from './api';

type SidebarTab = 'chats' | 'settings';

export function App() {
  const agent = useAgent();
  const history = useSessionHistory();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('chats');
  const [skillsPanelOpen, setSkillsPanelOpen] = useState(false);
  const [skillsPanelInitial, setSkillsPanelInitial] = useState<string | undefined>();
  const [envPanelOpen, setEnvPanelOpen] = useState(false);
  const [cronPanelOpen, setCronPanelOpen] = useState(false);
  const [mcpPanelOpen, setMcpPanelOpen] = useState(false);
  const [markdownPanelOpen, setMarkdownPanelOpen] = useState(false);
  const [configInfo, setConfigInfo] = useState<ConfigInfo | null>(null);
  const [workbenchOpen, setWorkbenchOpen] = useState(false);
  const [workbenchDismissed, setWorkbenchDismissed] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteSessions, setPaletteSessions] = useState<SessionInfo[]>([]);

  const isViewingHistory = history.viewingSessionId !== null;

  // Derive current tool activity for StatusBar
  const currentActivity = useMemo(() => {
    if (agent.state !== 'running' || agent.toolCalls.length === 0) return null;
    const pending = [...agent.toolCalls].reverse().find(tc => tc.result === undefined);
    if (!pending) return null;
    const name = normalizeToolName(pending.toolName);
    // Short human-readable labels
    if (name === 'file.read') return 'Reading files';
    if (name === 'file.edit') return 'Editing code';
    if (name === 'file.write') return 'Writing files';
    if (name === 'bash') return 'Running command';
    if (name.startsWith('browser.')) return 'Browsing';
    if (name === 'web.search') return 'Searching web';
    if (name === 'web.fetch') return 'Fetching page';
    if (name.startsWith('code.')) return 'Analyzing code';
    if (name === 'think') return 'Thinking';
    return name;
  }, [agent.state, agent.toolCalls]);
  const displayMessages = isViewingHistory ? (history.historyState?.messages ?? []) : agent.messages;
  const displayToolCalls = isViewingHistory ? (history.historyState?.toolCalls ?? []) : agent.toolCalls;
  const displayThinkingBlocks = isViewingHistory ? (history.historyState?.thinkingBlocks ?? []) : agent.thinkingBlocks;
  const displayActivitySummary = isViewingHistory ? (history.historyState?.activitySummary ?? null) : agent.activitySummary;
  const displayDelegateEvents = isViewingHistory ? (history.historyState?.delegateEvents ?? []) : agent.delegateEvents;
  const displayCompactionEvents = isViewingHistory ? (history.historyState?.compactionEvents ?? []) : agent.compactionEvents;

  useEffect(() => {
    let cancelled = false;

    const loadConfig = async () => {
      try {
        const next = await fetchConfig();
        if (!cancelled) {
          setConfigInfo(next);
        }
      } catch (err) {
        // Keep the status bar stable if config loading fails.
        console.debug('config poll failed', err);
      }
    };

    void loadConfig();
    const timer = window.setInterval(loadConfig, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // Workbench auto-open honors a per-run dismissal; a new run (empty toolCalls)
  // clears it so the panel can open again.
  useEffect(() => {
    if (isViewingHistory) {
      setWorkbenchOpen(false);
      return;
    }
    if (agent.toolCalls.length === 0) {
      setWorkbenchOpen(false);
      setWorkbenchDismissed(false);
      return;
    }
    if (!workbenchDismissed && inferWorkPhase(agent.toolCalls) !== 'analyzing') {
      setWorkbenchOpen(true);
    }
  }, [agent.toolCalls, isViewingHistory, workbenchDismissed]);

  // ⌘K opens the palette; ⌘J toggles the workbench.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      const key = e.key.toLowerCase();
      if (key === 'k') {
        e.preventDefault();
        setPaletteOpen(o => !o);
      } else if (key === 'j') {
        e.preventDefault();
        if (isViewingHistory) return;
        setWorkbenchOpen(open => {
          // Closing counts as a dismissal so auto-open doesn't fight the user.
          setWorkbenchDismissed(open);
          return !open;
        });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isViewingHistory]);

  // Load recent sessions when the palette opens, for quick jump-to-session.
  useEffect(() => {
    if (!paletteOpen) return;
    let cancelled = false;
    fetchSessions({ limit: 8 })
      .then(r => { if (!cancelled) setPaletteSessions(r.sessions); })
      .catch(err => { console.debug('palette session fetch failed', err); });
    return () => { cancelled = true; };
  }, [paletteOpen]);

  const handleSelectSession = (sessionId: string | null) => {
    history.loadSession(sessionId);
  };

  const handleNewChat = () => {
    history.loadSession(null);
    agent.resetLiveConversation();
  };

  const handleOpenSkillPanel = (selectedName?: string) => {
    setSkillsPanelInitial(selectedName);
    setSkillsPanelOpen(true);
  };

  const paletteCommands: Command[] = [
    { id: 'new', label: 'New chat', run: handleNewChat },
  ];
  if (!isViewingHistory) {
    paletteCommands.push({
      id: 'workbench',
      label: workbenchOpen ? 'Hide workbench' : 'Show workbench',
      hint: '⌘J',
      run: () => { setWorkbenchDismissed(workbenchOpen); setWorkbenchOpen(o => !o); },
    });
  } else {
    paletteCommands.push({ id: 'live', label: 'Back to live', run: () => handleSelectSession(null) });
  }
  paletteCommands.push(
    { id: 'skills', label: 'Open Skills', run: () => handleOpenSkillPanel() },
    { id: 'env', label: 'Open Environment variables', run: () => setEnvPanelOpen(true) },
    { id: 'cron', label: 'Open Scheduled tasks', run: () => setCronPanelOpen(true) },
    { id: 'mcp', label: 'Open MCP servers', run: () => setMcpPanelOpen(true) },
    { id: 'memory', label: 'Open Memory files', run: () => setMarkdownPanelOpen(true) },
  );
  for (const s of paletteSessions) {
    if (s.id === history.viewingSessionId) continue;
    paletteCommands.push({
      id: `sess-${s.id}`,
      label: `Go to: ${s.title || 'Untitled'}`,
      hint: `${s.turnCount} turns`,
      run: () => handleSelectSession(s.id),
    });
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: 'var(--bg-primary)',
    }}>
      <StatusBar
        connected={agent.connected}
        state={agent.state}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        tokenUsage={!isViewingHistory ? agent.tokenUsage : undefined}
        currentStepInfo={!isViewingHistory ? agent.currentStepInfo : undefined}
        currentActivity={currentActivity}
        activeModel={configInfo?.activeModel ?? null}
        lastRunSuccess={!isViewingHistory ? (agent.activitySummary?.success ?? null) : null}
        onOpenPalette={() => setPaletteOpen(true)}
      />

      {!agent.connected && (
        <div className="fade-in" style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '7px 16px',
          background: 'var(--danger-subtle)',
          borderBottom: '1px solid var(--border)',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--danger)',
        }}>
          <span className="spinner" style={{ width: 11, height: 11, borderTopColor: 'var(--danger)' }} />
          <span>Engine unreachable — reconnecting…</span>
          <span style={{ color: 'var(--text-muted)', marginLeft: 'auto' }}>
            start it with: rune daemon start
          </span>
        </div>
      )}

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sidebar with slide animation */}
        <div style={{
          width: sidebarOpen ? 260 : 0,
          flexShrink: 0,
          overflow: 'hidden',
          transition: 'width 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          borderRight: sidebarOpen ? '1px solid var(--border)' : 'none',
        }}>
          <div style={{ width: 260, height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Tab navigation */}
            <div style={{
              display: 'flex',
              borderBottom: '1px solid var(--border)',
              flexShrink: 0,
            }}>
              <SidebarTabButton
                active={sidebarTab === 'chats'}
                onClick={() => setSidebarTab('chats')}
                icon={
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2 3a1 1 0 011-1h8a1 1 0 011 1v6a1 1 0 01-1 1H5l-2 2V10H3a1 1 0 01-1-1V3z" />
                  </svg>
                }
                label="Chats"
              />
              <SidebarTabButton
                active={sidebarTab === 'settings'}
                onClick={() => setSidebarTab('settings')}
                icon={
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="7" cy="7" r="2" />
                    <path d="M7 1.5v1.5M7 11v1.5M1.5 7H3M11 7h1.5M3.2 3.2l1 1M9.8 9.8l1 1M3.2 10.8l1-1M9.8 4.2l1-1" />
                  </svg>
                }
                label="Settings"
              />
            </div>

            {/* Tab content */}
            <div style={{ flex: 1, overflow: 'hidden' }}>
              {sidebarTab === 'chats' && (
                <SessionSidebar
                  currentSessionId={history.viewingSessionId}
                  onSelectSession={handleSelectSession}
                  onNewChat={handleNewChat}
                />
              )}
              {sidebarTab === 'settings' && (
                <SettingsSidebar
                  onOpenSkillPanel={handleOpenSkillPanel}
                  onOpenEnvPanel={() => setEnvPanelOpen(true)}
                  onOpenCronPanel={() => setCronPanelOpen(true)}
                  onOpenMcpPanel={() => setMcpPanelOpen(true)}
                  onOpenMarkdownPanel={() => setMarkdownPanelOpen(true)}
                />
              )}
            </div>
          </div>
        </div>

        {/* Main content area */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
          position: 'relative',
        }}>
          {/* History banner */}
          {isViewingHistory && (
            <div className="fade-in" style={{
              padding: '8px 20px',
              background: 'var(--bg-surface)',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              fontSize: 13,
              color: 'var(--text-secondary)',
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--warning)',
                flexShrink: 0,
              }} />
              <span style={{ flex: 1 }}>
                Viewing past session
                {history.loading && ' ...'}
              </span>
              <button
                onClick={() => handleSelectSession(null)}
                style={{
                  padding: '5px 14px',
                  background: 'var(--accent)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                Back to live
              </button>
            </div>
          )}

          {!isViewingHistory && agent.savedDraft.available && (
            <div className="fade-in" style={{
              padding: '10px 20px',
              background: 'var(--bg-surface)',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              flexWrap: 'wrap',
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--accent)',
                flexShrink: 0,
              }} />
              <span style={{ fontSize: 13, color: 'var(--text-secondary)', flex: 1, minWidth: 220 }}>
                A previous live draft is available
                {' · '}
                {agent.savedDraft.messageCount} messages
                {' · '}
                {agent.savedDraft.toolCallCount} tools
              </span>
              <button
                onClick={agent.restoreSavedDraft}
                style={{
                  padding: '5px 12px',
                  background: 'var(--accent)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                Restore
              </button>
              <button
                onClick={agent.discardSavedDraft}
                style={{
                  padding: '5px 12px',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  fontWeight: 500,
                }}
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Chat area + workbench */}
          <div style={{
            flex: 1,
            overflow: 'hidden',
            position: 'relative',
            display: 'flex',
            flexDirection: 'row',
          }}>
            <div style={{
              flex: 1,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
            }}>
              <ChatPanel
                conversationKey={history.viewingSessionId ?? 'live'}
                messages={displayMessages}
                toolCalls={displayToolCalls}
                thinkingBlocks={displayThinkingBlocks}
                isRunning={!isViewingHistory && agent.state === 'running'}
                activitySummary={displayActivitySummary}
                delegateEvents={displayDelegateEvents}
                compactionEvents={displayCompactionEvents}
                currentStepInfo={isViewingHistory ? null : agent.currentStepInfo}
                pendingQuestion={isViewingHistory ? null : agent.pendingQuestion}
                onRespondQuestion={agent.respondQuestion}
                pendingApproval={isViewingHistory ? null : agent.pendingApproval}
                onRespondApproval={agent.respondApproval}
                onSuggest={
                  !isViewingHistory && agent.connected && agent.state === 'idle'
                    ? agent.sendMessage
                    : undefined
                }
              />

              {!isViewingHistory && !workbenchOpen && agent.toolCalls.length > 0 &&
                inferWorkPhase(agent.toolCalls) !== 'analyzing' && (
                <button
                  onClick={() => { setWorkbenchDismissed(false); setWorkbenchOpen(true); }}
                  title="Show the coding workbench (⌘J)"
                  style={{
                    position: 'absolute',
                    top: 12,
                    right: 16,
                    zIndex: 15,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 7,
                    padding: '5px 11px',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--accent-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-secondary)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Workbench {'›'}
                </button>
              )}

              {/* Input area */}
              {isViewingHistory ? (
                <div style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  padding: '12px 20px',
                  textAlign: 'center',
                  zIndex: 10,
                }}>
                  <div className="glass" style={{
                    display: 'inline-block',
                    padding: '8px 20px',
                    borderRadius: 'var(--radius-lg)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-muted)',
                    fontSize: 13,
                  }}>
                    Read-only: viewing session history
                  </div>
                </div>
              ) : (
                <InputArea
                  onSend={agent.sendMessage}
                  onAbort={agent.abort}
                  isRunning={agent.state !== 'idle'}
                  disabled={!agent.connected}
                />
              )}
            </div>

            {!isViewingHistory && workbenchOpen && (
              <WorkbenchPanel
                toolCalls={agent.toolCalls}
                isRunning={agent.state === 'running'}
                activitySummary={agent.activitySummary}
                connected={agent.connected}
                onClose={() => { setWorkbenchOpen(false); setWorkbenchDismissed(true); }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Panel overlays */}
      {skillsPanelOpen && (
        <SkillsPanel
          onClose={() => setSkillsPanelOpen(false)}
          initialSkillName={skillsPanelInitial}
        />
      )}
      {envPanelOpen && (
        <EnvPanel onClose={() => setEnvPanelOpen(false)} />
      )}
      {cronPanelOpen && (
        <CronPanel onClose={() => setCronPanelOpen(false)} />
      )}
      {mcpPanelOpen && (
        <MCPPanel onClose={() => setMcpPanelOpen(false)} />
      )}
      {markdownPanelOpen && (
        <MarkdownPanel onClose={() => setMarkdownPanelOpen(false)} />
      )}

      <CommandK
        open={paletteOpen}
        commands={paletteCommands}
        onClose={() => setPaletteOpen(false)}
      />
    </div>
  );
}

function SidebarTabButton({ active, onClick, icon, label }: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        padding: '10px 0',
        background: 'transparent',
        border: 'none',
        borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
        color: active ? 'var(--text-primary)' : 'var(--text-muted)',
        fontSize: 11,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'color 0.15s, border-color 0.15s',
      }}
    >
      {icon}
      {label}
    </button>
  );
}
