import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ModelPicker } from './ModelPicker';
import { ReasoningPicker } from './ReasoningPicker';
import type { ReasoningEffort } from '../api';
import { ThemeToggle } from './ThemeToggle';
import type { AgentState, StepInfo, TokenUsage as TokenUsageType } from '../types';
import { TokenUsage } from './TokenUsage';
import { RuneMark } from './RuneMark';

interface StatusBarProps {
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
  approvalMode?: string;
  activeModel?: { provider: string; model: string; source: 'active' | 'default' } | null;
  reasoningSupported?: boolean;
  reasoningEffort?: ReasoningEffort | null;
  reasoningOptions?: ReasoningEffort[];
  reasoningBudgets?: Partial<Record<ReasoningEffort, number>>;
  lastRunSuccess?: boolean | null;
  onOpenPalette?: () => void;
}

const STATE_LABELS: Record<AgentState, string> = {
  idle: 'Ready', running: 'Running', waiting_approval: 'Awaiting approval', waiting_question: 'Awaiting answer',
};

function formatTokensCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

export function StatusBar({
  trailing, connected, state, sidebarOpen, onToggleSidebar, tokenUsage,
  currentStepInfo, currentActivity, activeModel, reasoningSupported, reasoningEffort,
  reasoningOptions, reasoningBudgets, approvalMode, lastRunSuccess = null, onOpenPalette, onToggleWorkbench, workbenchOpen,
}: StatusBarProps) {
  const [showTokens, setShowTokens] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!showTokens) return;
    const dismiss = (event: MouseEvent) => {
      if (!popoverRef.current?.contains(event.target as Node)) setShowTokens(false);
    };
    document.addEventListener('mousedown', dismiss);
    return () => document.removeEventListener('mousedown', dismiss);
  }, [showTokens]);

  const markState = !connected ? 'warning' : state === 'running' ? 'working' : state !== 'idle' ? 'thinking'
    : lastRunSuccess === true ? 'passed' : lastRunSuccess === false ? 'failed' : 'idle';

  return <header className="app-toolbar">
    <div className="toolbar-brand">
      {onToggleSidebar && <button className="toolbar-icon" onClick={onToggleSidebar}
        aria-label={sidebarOpen ? 'Hide sessions' : 'Show sessions'} aria-expanded={sidebarOpen}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <rect x="2" y="3" width="14" height="12" rx="2" /><path d="M7 3v12" />
        </svg>
      </button>}
      <RuneMark state={markState} size={24} title={connected ? `RUNE (${STATE_LABELS[state]})` : 'RUNE — engine unreachable'} />
      <span className="toolbar-wordmark">RUNE</span>
    </div>

    <div className="toolbar-models" aria-label="Model settings">
      {activeModel && <ModelPicker active={activeModel} />}
      {activeModel && reasoningSupported && !!reasoningOptions?.length && <ReasoningPicker
        key={`${activeModel.provider}/${activeModel.model}`} model={activeModel}
        effort={reasoningEffort ?? null} options={reasoningOptions} budgets={reasoningBudgets} />}
    </div>

    <div className="toolbar-actions">
      <div className="toolbar-workspace">{trailing}</div>
      {onToggleWorkbench && <button className="toolbar-button work-toggle" onClick={onToggleWorkbench}
        title="Toggle Work panel (⌘J)" aria-label="Toggle Work panel" aria-pressed={workbenchOpen}>
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true">
          <rect x="2" y="3" width="12" height="10" rx="2" /><path d="M9 3v10" />
        </svg><span>Work</span>
      </button>}
      <ThemeToggle />
      {onOpenPalette && <button className="toolbar-icon command-trigger" onClick={onOpenPalette}
        title="Command palette (⌘K)" aria-label="Open command palette">
        <svg width="17" height="17" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <circle cx="8.5" cy="8.5" r="5.5" /><path d="m13 13 4 4" strokeLinecap="round" />
        </svg>
      </button>}
      <span title={connected ? 'Connected' : 'Disconnected'}
        className={`toolbar-connection status-dot ${connected ? 'status-dot--success' : 'status-dot--danger'}`} />
    </div>

    {(state !== 'idle' || approvalMode === 'bypass' || tokenUsage) && <div className="toolbar-status">
      {state !== 'idle' && <div className="toolbar-activity" data-attention={state !== 'running'}>
        <span className="status-dot status-dot--pulse" />
        <span>{STATE_LABELS[state]}</span>
        {state === 'running' && currentStepInfo && <span className="toolbar-step">
          Step {currentStepInfo.stepNumber}{currentStepInfo.tokens > 0 && ` · ${formatTokensCompact(currentStepInfo.tokens)}`}
        </span>}
        {state === 'running' && currentActivity && <span className="toolbar-current">{currentActivity}</span>}
      </div>}
      {approvalMode === 'bypass' && <span className="approval-mode-badge"
        title="RUNE will not ask before risky commands, MCP writes, or network writes">Approvals off</span>}
      {tokenUsage && <div ref={popoverRef} className="toolbar-tokens">
        <button className="toolbar-button" aria-label="Token usage" aria-expanded={showTokens}
          onClick={() => setShowTokens(!showTokens)}>{formatTokensCompact(tokenUsage.total)} tokens</button>
        {showTokens && <div className="token-popover fade-scale">
          <TokenUsage usage={tokenUsage} provider={activeModel?.provider} />
        </div>}
      </div>}
    </div>}
  </header>;
}
