import { ConversationWorkbenches } from './components/ConversationWorkbenches';
import { useWorkbenchState } from './hooks/useWorkbenchState';
import { toast } from './utils/toast';
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAgent } from './hooks/useAgent';
import { useSessionHistory } from './hooks/useSessionHistory';
import { ChatPanel } from './components/ChatPanel';
import { SessionSidebar } from './components/SessionSidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import { SettingsDialog } from './components/SettingsDialog';
import { StatusBar } from './components/StatusBar';
import { InputArea, type InputAreaHandle } from './components/InputArea';
import { ModelControls } from './components/ModelControls';
import { useFocusTrap } from './hooks/useFocusTrap';
// Load settings panels on demand to keep the initial bundle small.
const SkillsPanel = lazy(() => import('./components/SkillsPanel').then(m => ({ default: m.SkillsPanel })));
const EnvPanel = lazy(() => import('./components/EnvPanel').then(m => ({ default: m.EnvPanel })));
const CronPanel = lazy(() => import('./components/CronPanel').then(m => ({ default: m.CronPanel })));
const MCPPanel = lazy(() => import('./components/MCPPanel').then(m => ({ default: m.MCPPanel })));
const MarkdownPanel = lazy(() => import('./components/MarkdownPanel').then(m => ({ default: m.MarkdownPanel })));
const MemoryPanel = lazy(() => import('./components/MemoryPanel').then(m => ({ default: m.MemoryPanel })));
import { ResumeRunCard } from './components/ResumeRunCard';
import { CommandK, type Command } from './components/CommandK';
import { WorkspaceChip } from './components/WorkspaceChip';
import { InlineWorkspacePicker } from './components/InlineWorkspacePicker';
import { Toaster } from './components/Toaster';
import { normalizeToolName, isCodingToolName, computeRunVerdict } from './utils/tooling';
import { desktopAttention, shouldOpenWorkbench } from './utils/workbench';
import { fetchConfig, fetchSessions, getLiveSessionId, type ConfigInfo, type ModelConnectionAction, type SessionInfo } from './api';
import { watchPresence } from './utils/presence';
import { ConnectionBanner } from './components/ConnectionBanner';
import { SuggestionsDialog } from './components/SuggestionsDialog';
import { getModelSetupConnection } from './utils/modelConnection';
import './styles/conversation.css';

// Ask for a workspace when file or command tools need one.
function isWorkspaceTool(name: string): boolean {
  return isCodingToolName(name) || name === 'code.analyze';
}

export function App() {
  const agent = useAgent();
  useEffect(() => watchPresence(), []);
  const history = useSessionHistory();
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 820px)').matches);
  const [workbenchOverlay, setWorkbenchOverlay] = useState(() => window.matchMedia('(max-width: 960px)').matches);
  const [sidebarOpen, setSidebarOpen] = useState(() => !window.matchMedia('(max-width: 820px)').matches);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [newChatConfirmOpen, setNewChatConfirmOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState<'models' | undefined>();
  const [modelMenuRequest, setModelMenuRequest] = useState<{ provider: string; id: number }>();
  const [currentWorkspacePath, setCurrentWorkspacePath] = useState<string | null>(null);
  const [skillsPanelOpen, setSkillsPanelOpen] = useState(false);
  const [skillsPanelInitial, setSkillsPanelInitial] = useState<string | undefined>();
  const [envPanelOpen, setEnvPanelOpen] = useState(false);
  const [envPanelMode, setEnvPanelMode] = useState<'providers' | 'all'>('all');
  const [envInitialKey, setEnvInitialKey] = useState<string>();
  const openEnvironment = (mode: 'providers' | 'all', key?: string | null) => { setEnvPanelMode(mode); setEnvInitialKey(key ?? undefined); setEnvPanelOpen(true); };
  const [cronPanelOpen, setCronPanelOpen] = useState(false);
  const [mcpPanelOpen, setMcpPanelOpen] = useState(false);
  const [markdownPanelOpen, setMarkdownPanelOpen] = useState(false);
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);
  const [configInfo, setConfigInfo] = useState<ConfigInfo | null>(null);
  const [pendingModel, setPendingModel] = useState<{ provider: string; model: string }>();
  useEffect(() => { setPendingModel(undefined); }, [configInfo]);
  const liveWorkbench = useWorkbenchState(getLiveSessionId());
  const historyWorkbench = useWorkbenchState(history.viewingSessionId);
  const viewedWorkbench = history.viewingSessionId ? historyWorkbench : liveWorkbench;
  const { open: workbenchOpen, setOpen: setWorkbenchOpen, tab: workbenchTab,
    expanded: computerExpanded, dismissed: workbenchDismissed, setDismissed: setWorkbenchDismissed,
    setComputerRequest, setTab: setWorkbenchTab } = viewedWorkbench;
  const inputAreaRef = useRef<InputAreaHandle>(null);
  const [chatDragOver, setChatDragOver] = useState(false);
  const chatDragDepth = useRef(0);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteSessions, setPaletteSessions] = useState<SessionInfo[]>([]);
  const sidebarTriggerRef = useRef<HTMLButtonElement>(null);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const settingsReturnFocus = useRef<HTMLElement | null>(null);
  const suggestionsReturnFocus = useRef<HTMLElement | null>(null);
  const newChatReturnFocus = useRef<HTMLElement | null>(null);
  const openSuggestions = () => {
    suggestionsReturnFocus.current = paletteOpen ? sidebarTriggerRef.current : document.activeElement as HTMLElement | null;
    setSuggestionsOpen(true);
  };
  const panelOpen = skillsPanelOpen || envPanelOpen || cronPanelOpen || mcpPanelOpen || markdownPanelOpen || memoryPanelOpen;
  const overlayOpen = settingsOpen || panelOpen || paletteOpen || suggestionsOpen || newChatConfirmOpen;
  const sidebarRef = useFocusTrap<HTMLDivElement>(compact && sidebarOpen && !overlayOpen, sidebarTriggerRef.current);
  const openSettings = () => {
    setSettingsSection(undefined);
    settingsReturnFocus.current = paletteOpen ? settingsTriggerRef.current : document.activeElement as HTMLElement | null;
    setSettingsOpen(true);
  };

  const chooseProviderModel = (provider: string) => {
    setEnvPanelOpen(false);
    setSettingsOpen(false);
    setModelMenuRequest(previous => ({ provider, id: (previous?.id ?? 0) + 1 }));
  };
  const configureModel = (action: ModelConnectionAction) => {
    if (action.target === 'environment') {
      openEnvironment('providers', action.key);
    } else if (action.target === 'models') {
      chooseProviderModel('');
    } else {
      openSettings();
      setSettingsSection('models');
    }
  };
  useEffect(() => { setCurrentWorkspacePath(null); }, [liveWorkbench.sessionId]);

  const isViewingHistory = history.viewingSessionId !== null;
  const openComputer = useCallback(() => {
    if (isViewingHistory) return;
    setComputerRequest(value => value + 1);
    setWorkbenchTab('computer');
    setWorkbenchDismissed(false);
    setWorkbenchOpen(true);
  }, [isViewingHistory, setComputerRequest, setWorkbenchTab, setWorkbenchDismissed, setWorkbenchOpen]);
  const desktopWait = desktopAttention(agent.toolCalls, !isViewingHistory && agent.state === 'running');
  const attentionLabel = desktopWait === 'connection' ? 'Waiting for app access' : desktopWait === 'action' ? 'Action needs review' : undefined;

  useEffect(() => {
    const media = window.matchMedia('(max-width: 820px)');
    const workbenchMedia = window.matchMedia('(max-width: 960px)');
    const resize = () => { setCompact(media.matches); if (media.matches) setSidebarOpen(false); };
    const resizeWorkbench = () => setWorkbenchOverlay(workbenchMedia.matches);
    media.addEventListener('change', resize);
    workbenchMedia.addEventListener('change', resizeWorkbench);
    return () => { media.removeEventListener('change', resize); workbenchMedia.removeEventListener('change', resizeWorkbench); };
  }, []);

  const touchedWorkspace = useMemo(
    () => agent.toolCalls.some(tc => isWorkspaceTool(normalizeToolName(tc.toolName))),
    [agent.toolCalls],
  );

  const currentActivity = useMemo(() => {
    if (agent.state !== 'running' || agent.toolCalls.length === 0) return null;
    const pending = [...agent.toolCalls].reverse().find(tc => tc.result === undefined);
    if (!pending) return null;
    const name = normalizeToolName(pending.toolName);
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
  const suggestions = agent.messages.flatMap(message => message.suggestion ? [message.suggestion] : []);
  const pendingSuggestions = suggestions.filter(item => !item.state || ['pending', 'unconfirmed'].includes(item.state.response)).length;
  const modelSetup = getModelSetupConnection(configInfo && { ...configInfo, activeModel: pendingModel ? { ...pendingModel, source: 'active' } : configInfo.activeModel });
  const newChatBlockedReason = agent.state !== 'idle' ? 'Finish or stop the current task before starting a new chat.'
    : agent.loadingConversation ? 'Wait for the conversation to open.'
    : agent.savedDraft.available ? 'Restore or discard the saved conversation before starting a new chat.' : undefined;
  const taskTitle = displayMessages.find(message => message.role === 'user')?.content?.trim().split('\n')[0];
  const liveChatMessages = agent.messages.filter(message => !message.suggestion);
  const unconfirmedDelivery = liveChatMessages.some(message => message.role === 'user' && ['pending', 'unknown', 'rejected'].includes(message.delivery ?? ''));
  const liveTitle = agent.messages.find(message => message.role === 'user')?.content?.trim().split('\n')[0];
  const displayToolCalls = isViewingHistory ? (history.historyState?.toolCalls ?? []) : agent.toolCalls;
  const displayThinkingBlocks = isViewingHistory ? (history.historyState?.thinkingBlocks ?? []) : agent.thinkingBlocks;
  const displayActivitySummary = isViewingHistory ? (history.historyState?.activitySummary ?? null) : agent.activitySummary;
  const displayDelegateEvents = isViewingHistory ? (history.historyState?.delegateEvents ?? []) : agent.delegateEvents;
  const displayCompactionEvents = isViewingHistory ? (history.historyState?.compactionEvents ?? []) : agent.compactionEvents;
  const displayTrust = isViewingHistory ? (history.historyState?.trust ?? null) : agent.lastTrust;
  const displayFileChanges = isViewingHistory ? (history.historyState?.run?.fileChanges ?? []) : agent.fileChanges;
  const viewedSessionRef = useRef('');
  viewedSessionRef.current = history.viewingSessionId ?? getLiveSessionId();
  const interruptedRun = isViewingHistory ? history.historyState?.run : agent.interruptedRun;

  // Use the same verdict in chat, the workbench, and the status bar.
  const runVerdict = computeRunVerdict(displayTrust, displayActivitySummary);

  useEffect(() => {
    let cancelled = false;
    let version = 0;

    const loadConfig = async () => {
      const requestedVersion = ++version;
      try {
        const next = await fetchConfig();
        if (!cancelled && requestedVersion === version) {
          setConfigInfo(next);
        }
      } catch (err) {
        // Keep the status bar stable if config loading fails.
        console.debug('config poll failed', err);
      }
    };

    void loadConfig();
    window.addEventListener('rune:config-changed', loadConfig);
    const timer = window.setInterval(loadConfig, 30000);
    return () => {
      cancelled = true;
      window.removeEventListener('rune:config-changed', loadConfig);
      window.clearInterval(timer);
    };
  }, []);

  const hasWorkbenchContent = displayToolCalls.length > 0 || displayFileChanges.length > 0;
  useEffect(() => {
    if (workbenchOverlay || !hasWorkbenchContent || workbenchOpen || workbenchDismissed) return;
    // Open the panel once tool results give it something to show.
    if (shouldOpenWorkbench(displayToolCalls, displayFileChanges)) {
      setWorkbenchOpen(true);
    }
  }, [workbenchOverlay, displayToolCalls, displayFileChanges, hasWorkbenchContent, workbenchOpen, workbenchDismissed, setWorkbenchOpen]);

  // Let focused controls handle Escape before applying app shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k' && (!overlayOpen || paletteOpen)) {
        e.preventDefault();
        setPaletteOpen(o => !o);
        return;
      }
      if (overlayOpen) return;
      if (e.key === 'Escape' && !isViewingHistory && agent.state === 'running') {
        e.preventDefault();
        agent.abort();
        return;
      }
      if (!(e.metaKey || e.ctrlKey)) return;
      const key = e.key.toLowerCase();
      if (key === 'b' && e.shiftKey && !isViewingHistory) {
        e.preventDefault();
        openComputer();
      } else if (key === 'j') {
        e.preventDefault();
        setWorkbenchDismissed(workbenchOpen);
        setWorkbenchOpen(!workbenchOpen);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isViewingHistory, agent.state, agent.abort, openComputer, workbenchOpen, setWorkbenchOpen, setWorkbenchDismissed, overlayOpen, paletteOpen, settingsOpen, panelOpen]);

  // Prevent stray file drops from navigating away from the chat.
  useEffect(() => {
    const prevent = (e: DragEvent) => { e.preventDefault(); };
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', prevent);
    return () => {
      window.removeEventListener('dragover', prevent);
      window.removeEventListener('drop', prevent);
    };
  }, []);

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
    agent.cancelConversationLoad();
    history.loadSession(sessionId === getLiveSessionId() ? null : sessionId);
    if (window.matchMedia('(max-width: 820px)').matches) setSidebarOpen(false);
  };

  const startNewChat = () => {
    if (newChatBlockedReason) { toast.info(newChatBlockedReason); return; }
    if (!agent.resetLiveConversation()) return;
    setNewChatConfirmOpen(false);
    history.loadSession(null);
    if (compact) setSidebarOpen(false);
    // Restore focus after the closing dialog/sidebar releases its focus trap.
    requestAnimationFrame(() => inputAreaRef.current?.focus());
  };
  const handleNewChat = () => {
    if (newChatBlockedReason) { toast.info(newChatBlockedReason); return; }
    if (unconfirmedDelivery || (!liveChatMessages.length && inputAreaRef.current?.hasDraft())) {
      newChatReturnFocus.current = paletteOpen ? sidebarTriggerRef.current : document.activeElement as HTMLElement | null;
      setNewChatConfirmOpen(true);
    } else {
      startNewChat();
    }
  };

  const continueHistory = async (text?: string, attachments?: Parameters<typeof agent.sendMessage>[1]) => {
    const id = history.viewingSessionId;
    if (!id) return false;
    const loaded = await agent.loadConversation(id, text !== undefined);
    if (!loaded || viewedSessionRef.current !== id) return false;
    history.loadSession(null);
    return text === undefined ? true : agent.sendMessage(text, attachments);
  };


  const handleOpenSkillPanel = (selectedName?: string) => {
    setSkillsPanelInitial(selectedName);
    setSkillsPanelOpen(true);
  };

  const paletteCommands: Command[] = [
    { id: 'new', label: 'New chat', run: handleNewChat },
    { id: 'suggestions', label: 'Open Suggestions', run: openSuggestions },
    { id: 'settings', label: 'Open Settings', run: openSettings },
  ];
  if (!isViewingHistory) {
    paletteCommands.push({ id: 'computer', label: 'Open Computer', hint: '⌘⇧B', run: openComputer });
    paletteCommands.push({
      id: 'workbench',
      label: workbenchOpen ? 'Hide workbench' : 'Show workbench',
      hint: '⌘J',
      run: () => { setWorkbenchDismissed(workbenchOpen); setWorkbenchOpen(o => !o); },
    });
  } else {
    paletteCommands.push({ id: 'live', label: 'Back to current task', run: () => handleSelectSession(null) });
  }
  paletteCommands.push(
    { id: 'skills', label: 'Open Skills', run: () => handleOpenSkillPanel() },
    { id: 'env', label: 'Open Environment variables', run: () => openEnvironment('all') },
    { id: 'cron', label: 'Open Scheduled tasks', run: () => setCronPanelOpen(true) },
    { id: 'mcp', label: 'Open MCP servers', run: () => setMcpPanelOpen(true) },
    { id: 'memory', label: 'Open Memory', run: () => setMemoryPanelOpen(true) },
    { id: 'memory-files', label: 'Open Memory files', run: () => setMarkdownPanelOpen(true) },
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
    <div className="app-root" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100dvh',
      background: 'var(--bg-primary)',
    }}>
      <StatusBar
        inert={overlayOpen}
        workspaceLabel={configInfo ? configInfo.executionEnvironment?.managed ? 'Hosted workspace' : 'Connected engine' : 'Connecting'}
        onOpenSettings={openSettings}
        sidebarTriggerRef={sidebarTriggerRef}
        settingsTriggerRef={settingsTriggerRef}
        connected={agent.connected}
        state={agent.state}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        tokenUsage={!isViewingHistory ? agent.tokenUsage : undefined}
        currentStepInfo={!isViewingHistory ? agent.currentStepInfo : undefined}
        currentActivity={currentActivity}
        attentionLabel={attentionLabel}
        activeModel={configInfo?.activeModel ?? null}
        approvalMode={configInfo?.approvalMode}
        lastRunSuccess={runVerdict}
        onOpenPalette={() => setPaletteOpen(true)}
        trailing={!isViewingHistory ? <WorkspaceChip key={liveWorkbench.sessionId} onWorkspaceChange={setCurrentWorkspacePath} /> : null}
        workbenchOpen={workbenchOpen}
        onToggleWorkbench={() => {
          setWorkbenchDismissed(workbenchOpen);
          setWorkbenchOpen(o => !o);
        }}
      />

      {!agent.connected && <ConnectionBanner />}

      <div className="app-body" data-sidebar={sidebarOpen} inert={overlayOpen} style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {sidebarOpen && <button type="button" className="sidebar-backdrop" aria-label="Hide chats" tabIndex={-1}
          onClick={() => setSidebarOpen(false)} />}
        <div ref={sidebarRef} className="app-sidebar"
          role={compact && sidebarOpen ? 'dialog' : undefined} aria-modal={compact && sidebarOpen || undefined}
          aria-label="Chats" inert={!sidebarOpen} aria-hidden={!sidebarOpen}
          onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setSidebarOpen(false); } }} style={{
          width: sidebarOpen ? 'var(--sidebar-width)' : 0,
          flexShrink: 0,
          overflow: 'hidden',
          transition: 'width 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          borderRight: sidebarOpen ? '1px solid var(--border)' : 'none',
        }}>
          <div style={{ width: 'var(--sidebar-width)', height: '100%', display: 'flex', flexDirection: 'column' }}>
            {compact && <button type="button" className="sidebar-close" onClick={() => setSidebarOpen(false)}>Close chats <span aria-hidden="true">×</span></button>}
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <SessionSidebar
                currentSessionId={history.viewingSessionId}
                onSelectSession={handleSelectSession}
                onNewChat={handleNewChat}
                liveTitle={liveTitle}
                liveState={agent.state}
                retainLive={!unconfirmedDelivery}
                liveMessageCount={liveChatMessages.length}
                liveUpdatedAt={liveChatMessages[liveChatMessages.length - 1]?.timestamp}
                newChatDisabled={!!newChatBlockedReason}
                newChatDisabledReason={newChatBlockedReason}
                suggestionCount={pendingSuggestions}
                onOpenSuggestions={openSuggestions}
                onOpenSettings={openSettings}
                onOpenSchedule={() => setCronPanelOpen(true)}
                taskSelectionDisabled={agent.state !== 'idle' || agent.loadingConversation}
                onSelectTask={id => {
                  inputAreaRef.current?.setDraft(`Continue saved task ${id}.`);
                  if (compact) setSidebarOpen(false);
                }}
              />
            </div>
          </div>
        </div>

        <div className="task-space" style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
          position: 'relative',
        }}>
          {isViewingHistory && <header className="conversation-heading">
            <h1 title={taskTitle}>{history.loading ? 'Opening chat…' : taskTitle || 'Saved chat'}</h1>
            {['queued', 'running', 'waiting_input', 'waiting_approval'].includes(history.historyState?.run?.status ?? '') && <button type="button"
              disabled={!agent.connected || history.loading || agent.loadingConversation || agent.state !== 'idle' || agent.savedDraft.available}
              onClick={() => { void continueHistory().catch(error => toast.error(error instanceof Error ? error.message : 'Could not open this task.')); }}>
              {agent.loadingConversation ? 'Opening task…' : 'View running task'}
            </button>}
            <button type="button" onClick={() => handleSelectSession(null)}>
              {agent.state !== 'idle' ? 'Return to running task' : 'Back to current chat'}
            </button>
          </header>}

          {interruptedRun?.status === 'interrupted' && interruptedRun.sessionId === viewedSessionRef.current && (
            <ResumeRunCard key={interruptedRun.runId} run={interruptedRun} connected={agent.connected}
              onResumed={sessionId => {
                if (viewedSessionRef.current !== interruptedRun.sessionId) return;
                agent.followResumedRun(sessionId);
                handleSelectSession(null);
              }} />
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
              <span style={{ fontSize: 13, color: 'var(--text-secondary)', flex: 1, minWidth: 150 }}>
                Saved conversation
                {' · '}
                {agent.savedDraft.messageCount} {agent.savedDraft.messageCount === 1 ? 'message' : 'messages'}
                {agent.savedDraft.toolCallCount > 0 && ` · ${agent.savedDraft.toolCallCount} tool ${agent.savedDraft.toolCallCount === 1 ? 'call' : 'calls'}`}
              </span>
              <div className="notice-actions">
                <button
                  onClick={agent.restoreSavedDraft}
                  style={{
                    padding: '5px 12px',
                    background: 'var(--accent)',
                    color: 'var(--text-on-accent)',
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
                  Discard
                </button>
              </div>
            </div>
          )}

          {/* Keep the composer available beside every workbench view. */}
          <div
            className="chat-workbench-grid"
            data-bench={workbenchOpen ? 'open' : 'closed'}
            data-workbench-tab={workbenchTab}
            data-computer={workbenchTab === 'computer'}
            data-expanded={computerExpanded}
            style={{
              flex: 1,
              overflow: 'hidden',
              position: 'relative',
              display: 'grid',
            }}
          >
            <div
              className="chat-column"
              style={{
                minWidth: 0,
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
              }}
              onDragEnter={(e) => {
                if (isViewingHistory) return;
                if (!Array.from(e.dataTransfer?.types ?? []).includes('Files')) return;
                e.preventDefault();
                chatDragDepth.current += 1;
                setChatDragOver(true);
              }}
              onDragOver={(e) => {
                if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault();
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                chatDragDepth.current = Math.max(0, chatDragDepth.current - 1);
                if (chatDragDepth.current === 0) setChatDragOver(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                chatDragDepth.current = 0;
                setChatDragOver(false);
                if (!isViewingHistory && e.dataTransfer?.files.length) {
                  inputAreaRef.current?.addFiles(e.dataTransfer.files);
                }
              }}
            >
              {chatDragOver && (
                <div style={{
                  position: 'absolute', inset: 12, zIndex: 30,
                  border: '2px dashed var(--accent)', borderRadius: 'var(--radius-lg)',
                  background: 'var(--accent-subtle, rgba(125,211,232,0.08))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  pointerEvents: 'none', color: 'var(--accent)', fontSize: 15, fontWeight: 600,
                }}>
                  Drop files to attach
                </div>
              )}
              {isViewingHistory && !displayMessages.length ? (
                <div role="status" style={{ padding: 24, color: 'var(--text-muted)' }}>
                  {history.loading ? 'Loading conversation…' : 'No saved messages in this conversation.'}
                </div>
              ) : <ChatPanel
                config={configInfo}
                connected={agent.connected}
                onOpenSettings={openSettings}
                conversationKey={history.viewingSessionId ?? getLiveSessionId()}
                messages={displayMessages}
                toolCalls={displayToolCalls}
                thinkingBlocks={displayThinkingBlocks}
                isRunning={!isViewingHistory && agent.state === 'running'}
                onRegenerate={
                  !isViewingHistory && agent.connected && agent.state === 'idle' && !modelSetup
                    ? agent.regenerate
                    : undefined
                }
                onEditResend={
                  !isViewingHistory && agent.connected && agent.state === 'idle' && !modelSetup
                    ? agent.sendMessage
                    : undefined
                }
                activitySummary={displayActivitySummary}
                delegateEvents={displayDelegateEvents}
                compactionEvents={displayCompactionEvents}
                currentStepInfo={isViewingHistory ? null : agent.currentStepInfo}
                pendingQuestion={isViewingHistory || agent.state === 'stopping' ? null : agent.pendingQuestion}
                onRespondQuestion={agent.respondQuestion}
                pendingApproval={isViewingHistory || agent.state === 'stopping' ? null : agent.pendingApproval}
                onRespondApproval={agent.respondApproval}
                onSuggest={
                  !isViewingHistory && agent.connected && agent.state === 'idle' && !modelSetup
                    ? agent.sendMessage
                    : undefined
                }
                onDraft={!isViewingHistory ? text => inputAreaRef.current?.setDraft(text) : undefined}
                streamFooter={
                  /* Prompt for a workspace when coding starts without one. */
                  !isViewingHistory && touchedWorkspace ? <InlineWorkspacePicker /> : null
                }
              />}

            </div>
            <div className="chat-composer">
              <InputArea
                settings={<ModelControls config={configInfo} openRequest={modelMenuRequest} onConfigure={configureModel}
                  onModelChange={setPendingModel} />}
                conversationKey={history.viewingSessionId ?? getLiveSessionId()}
                ref={inputAreaRef}
                onSend={isViewingHistory ? continueHistory : agent.sendMessage}
                onAbort={agent.abort}
                isRunning={!isViewingHistory && agent.state !== 'idle'}
                isStopping={agent.state === 'stopping' || agent.state === 'submitting'}
                attentionLabel={agent.state === 'submitting' ? 'Sending…' : agent.state === 'stopping' ? 'Stopping…' : attentionLabel}
                sendBlockedReason={isViewingHistory && agent.state !== 'idle' ? 'Your current task is still running. Return to it to view or stop it.' : agent.loadingConversation ? 'Opening conversation…' : modelSetup?.reason || (modelSetup ? 'Set up this model before sending.' : undefined)}
                disabled={!agent.connected}
              />
            </div>

            <ConversationWorkbenches workspacePath={currentWorkspacePath} agent={agent} live={liveWorkbench} history={historyWorkbench} historyState={history.historyState} />
          </div>
        </div>
      </div>

      {newChatConfirmOpen && <NewChatConfirmation returnFocus={newChatReturnFocus.current}
        blockedReason={newChatBlockedReason} onCancel={() => setNewChatConfirmOpen(false)} onConfirm={startNewChat} />}
      {suggestionsOpen && <SuggestionsDialog returnFocus={suggestionsReturnFocus.current} suggestions={suggestions} onClose={() => setSuggestionsOpen(false)} />}

      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} covered={panelOpen || paletteOpen}
        returnFocus={settingsReturnFocus.current}>
        <SettingsSidebar initialSection={settingsSection}
          onOpenSkillPanel={handleOpenSkillPanel}
          onOpenEnvPanel={() => openEnvironment('all')}
          onOpenProviderKeys={() => openEnvironment('providers')}
          onOpenCronPanel={() => setCronPanelOpen(true)}
          onOpenMcpPanel={() => setMcpPanelOpen(true)}
          onOpenMarkdownPanel={() => setMarkdownPanelOpen(true)}
          onOpenMemoryPanel={() => setMemoryPanelOpen(true)}
        />
      </SettingsDialog>}

      {/* Keep lazy-loaded panels visible and dismissible while loading. */}
      <Suspense fallback={<PanelLoading onDismiss={() => {
        setSkillsPanelOpen(false);
        setEnvPanelOpen(false);
        setCronPanelOpen(false);
        setMcpPanelOpen(false);
        setMarkdownPanelOpen(false);
        setMemoryPanelOpen(false);
      }} />}>
        {skillsPanelOpen && (
          <SkillsPanel
            onClose={() => setSkillsPanelOpen(false)}
            initialSkillName={skillsPanelInitial}
          />
        )}
        {envPanelOpen && (
          <EnvPanel initialMode={envPanelMode} initialKey={envInitialKey} onChooseModel={chooseProviderModel}
            managed={configInfo?.executionEnvironment?.managed} onClose={() => setEnvPanelOpen(false)} />
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
        {memoryPanelOpen && (
          <MemoryPanel onClose={() => setMemoryPanelOpen(false)}
            onOpenFiles={() => { setMemoryPanelOpen(false); setMarkdownPanelOpen(true); }} />
        )}
      </Suspense>

      <CommandK
        open={paletteOpen}
        commands={paletteCommands}
        onClose={() => setPaletteOpen(false)}
      />
      <Toaster />
    </div>
  );
}

function NewChatConfirmation({ onCancel, onConfirm, blockedReason, returnFocus }: {
  onCancel: () => void;
  onConfirm: () => void;
  blockedReason?: string;
  returnFocus: HTMLElement | null;
}) {
  const ref = useFocusTrap<HTMLDivElement>(true, returnFocus);
  return <div className="new-chat-overlay" onClick={event => { if (event.target === event.currentTarget) onCancel(); }}>
    <div ref={ref} className="new-chat-confirm" role="alertdialog" aria-modal="true" aria-labelledby="new-chat-title" aria-describedby="new-chat-description"
      onKeyDown={event => {
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onCancel(); }
      }}>
      <h2 id="new-chat-title">Start a new chat?</h2>
      <p id="new-chat-description">This chat has an unsent or unconfirmed message. Keep it here to retry or continue editing.</p>
      {blockedReason && <p role="status">{blockedReason}</p>}
      <div>
        <button type="button" onClick={onCancel}>Keep chat</button>
        <button type="button" disabled={!!blockedReason} onClick={onConfirm}>Start new chat</button>
      </div>
    </div>
  </div>;
}

// Keep slow-loading panels dismissible.
function PanelLoading({ onDismiss }: { onDismiss: () => void }) {
  const ref = useFocusTrap<HTMLDivElement>();
  const [stuck, setStuck] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setStuck(true), 6000);
    return () => clearTimeout(t);
  }, []);

  return (
    <div
      ref={ref}
      role="dialog"
      aria-modal="true"
      aria-label="Loading panel"
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          onDismiss();
        }
      }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 60,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0, 0, 0, 0.35)',
        color: 'var(--text-muted)',
        fontSize: 13,
      }}
    >
      <span role="status">{stuck ? "This panel isn't loading." : 'Loading…'}</span>
      <button className="msg-action-btn" onClick={onDismiss}>Close</button>
    </div>
  );
}
