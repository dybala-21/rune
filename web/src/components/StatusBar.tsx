import { useEffect, useId, useRef, useState, type ReactNode, type Ref } from 'react';
import { useFocusTrap } from '../hooks/useFocusTrap';
import type { AgentState, StepInfo, TokenUsage as TokenUsageType } from '../types';
import { TokenUsage } from './TokenUsage';
import { RuneMark } from './RuneMark';

interface StatusBarProps {
  inert?: boolean;
  workspaceLabel?: string;
  onOpenSettings?: () => void;
  sidebarTriggerRef?: Ref<HTMLButtonElement>;
  settingsTriggerRef?: Ref<HTMLButtonElement>;
  trailing?: ReactNode;
  onToggleWorkbench?: () => void;
  workbenchOpen?: boolean;
  connected: boolean;
  state: AgentState;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  tokenUsage?: TokenUsageType | null;
  currentStepInfo?: StepInfo | null;
  currentActivity?: string | null;
  attentionLabel?: string;
  approvalMode?: string;
  activeModel?: { provider: string; model: string; source: 'active' | 'default' } | null;
  lastRunSuccess?: boolean | null;
  onOpenPalette?: () => void;
}

const STATE_LABELS: Record<AgentState, string> = {
  idle: 'Ready', submitting: 'Sending…', running: 'Working', stopping: 'Stopping…', waiting_approval: 'Waiting for approval', waiting_question: 'Waiting for your answer',
};

function formatTokensCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

export function StatusBar({
  inert, workspaceLabel, onOpenSettings, sidebarTriggerRef, settingsTriggerRef,
  trailing, connected, state, sidebarOpen, onToggleSidebar, tokenUsage,
  currentStepInfo, currentActivity, attentionLabel, activeModel,
  approvalMode, lastRunSuccess = null, onOpenPalette, onToggleWorkbench, workbenchOpen,
}: StatusBarProps) {
  const [showTokens, setShowTokens] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const usageButtonRef = useRef<HTMLButtonElement>(null);
  const usageId = useId();
  const usageRef = useFocusTrap<HTMLDivElement>(showTokens && !inert, usageButtonRef.current);
  useEffect(() => { if (inert) setShowTokens(false); }, [inert]);
  useEffect(() => {
    if (!showTokens) return;
    const dismiss = (event: MouseEvent) => {
      if (!popoverRef.current?.contains(event.target as Node)) setShowTokens(false);
    };
    document.addEventListener('mousedown', dismiss);
    return () => document.removeEventListener('mousedown', dismiss);
  }, [showTokens]);

  const markState = !connected ? 'warning' : attentionLabel ? 'warning' : state === 'running' ? 'working' : state !== 'idle' ? 'thinking'
    : lastRunSuccess === true ? 'passed' : lastRunSuccess === false ? 'failed' : 'idle';

  return <header className="app-toolbar" inert={inert}>
    <div className="toolbar-brand">
      {onToggleSidebar && <button ref={sidebarTriggerRef} className="toolbar-icon" onClick={onToggleSidebar}
        aria-label={sidebarOpen ? 'Hide chats' : 'Show chats'} aria-expanded={sidebarOpen}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <rect x="2" y="3" width="14" height="12" rx="2" /><path d="M7 3v12" />
        </svg>
      </button>}
      <div className="toolbar-logo"><RuneMark state={markState} size={32} title="Rune" /><span className="toolbar-wordmark">rune</span></div>
      <span className="toolbar-workspace-label" title={workspaceLabel ?? 'Workspace details are loading'}>
        <span role="img" aria-label={connected ? 'Connected' : 'Disconnected'}
          className={`status-dot ${connected ? 'status-dot--success' : 'status-dot--danger'}`} />
        <span>{workspaceLabel ?? 'Connecting'}</span>
      </span>
    </div>

    <div className="toolbar-actions">
      <div className="toolbar-workspace">{trailing}</div>
      {onToggleWorkbench && <button className="toolbar-button work-toggle" onClick={onToggleWorkbench}
        title="Toggle work panel (⌘J)" aria-label={workbenchOpen ? 'Hide work panel' : 'Show work panel'} aria-expanded={workbenchOpen}>
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true">
          <rect x="2" y="3" width="12" height="10" rx="2" /><path d="M9 3v10" />
        </svg><span>Work</span>
      </button>}
      {tokenUsage && <div ref={popoverRef} className="toolbar-tokens">
        <button ref={usageButtonRef} type="button" className="toolbar-button usage-trigger" aria-label="Usage and cost"
          aria-haspopup="dialog" aria-expanded={showTokens} aria-controls={showTokens ? usageId : undefined}
          onClick={() => setShowTokens(!showTokens)}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true"><path d="M3 12V9m5 3V4m5 8V7" strokeLinecap="round" /></svg>
          <span>{formatUsage(tokenUsage)}</span>
        </button>
        {showTokens && <div ref={usageRef} id={usageId} className="token-popover fade-scale" role="dialog" aria-label="Usage and cost"
          onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setShowTokens(false); } }}>
          <div className="usage-heading"><strong>Latest run</strong>
            <button type="button" className="toolbar-icon" aria-label="Close usage details" onClick={() => setShowTokens(false)}>×</button>
          </div>
          <TokenUsage usage={tokenUsage} provider={activeModel?.provider} />
        </div>}
      </div>}
      {onOpenPalette && <button className="toolbar-icon command-trigger" onClick={onOpenPalette}
        title="Command palette (⌘K)" aria-label="Open command palette">
        <svg width="17" height="17" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <circle cx="8.5" cy="8.5" r="5.5" /><path d="m13 13 4 4" strokeLinecap="round" />
        </svg>
      </button>}
      {onOpenSettings && <button ref={settingsTriggerRef} type="button" className="toolbar-icon" onClick={onOpenSettings}
        aria-label="Open settings" title="Settings">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <path d="M3 5h12M3 13h12" strokeLinecap="round" /><circle cx="7" cy="5" r="2" fill="var(--bg-primary)"/><circle cx="11" cy="13" r="2" fill="var(--bg-primary)"/>
        </svg>
      </button>}
    </div>

    {(state !== 'idle' || approvalMode === 'bypass') && <div className="toolbar-status">
      {state !== 'idle' && <div className="toolbar-activity" data-attention={!!attentionLabel || state !== 'running'}>
        <span className={`status-dot ${attentionLabel ? 'status-dot--warning' : 'status-dot--pulse'}`} />
        <span>{attentionLabel || STATE_LABELS[state]}</span>
        {state === 'running' && !attentionLabel && currentStepInfo && <span className="toolbar-step">
          Step {currentStepInfo.stepNumber}
        </span>}
        {state === 'running' && !attentionLabel && currentActivity && <span className="toolbar-current">{currentActivity}</span>}
      </div>}
      {approvalMode === 'bypass' && <span className="approval-mode-badge"
        title="Rune will not ask before risky commands, MCP writes, or network writes">Approvals off</span>}

    </div>}
  </header>;
}

function formatUsage(usage: TokenUsageType): string {
  const cost = usage.cost;
  if (cost?.pending || cost?.incomplete || cost?.unpricedCalls || usage.unreportedCalls) return 'Usage';
  if (cost?.usd != null) return `~$${cost.usd.toFixed(cost.usd < 0.01 ? 4 : 2)}`;
  return `${formatTokensCompact(usage.total)} tokens`;
}
