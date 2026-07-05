import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { AgentState, StepInfo, TokenUsage as TokenUsageType } from '../types';
import { TokenUsage } from './TokenUsage';
import { PixelWolf } from './PixelWolf';

interface StatusBarProps {
  /** Trailing slot rendered before the palette button (e.g. workspace chip) */
  trailing?: ReactNode;
  /** Toggle the coding workbench (file/diff/terminal). */
  onToggleWorkbench?: () => void;
  workbenchOpen?: boolean;
  connected: boolean;
  state: AgentState;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  tokenUsage?: TokenUsageType | null;
  /** Current step info for running state detail */
  currentStepInfo?: StepInfo | null;
  /** Current tool activity label */
  currentActivity?: string | null;
  activeModel?: {
    provider: string;
    model: string;
    source: 'active' | 'default';
  } | null;
  /** Outcome of the last completed run; null when none this session. */
  lastRunSuccess?: boolean | null;
  /** Opens the ⌘K command palette. */
  onOpenPalette?: () => void;
}

const STATE_LABELS: Record<AgentState, string> = {
  idle: 'Ready',
  running: 'Running',
  waiting_approval: 'Awaiting Approval',
  waiting_question: 'Awaiting Answer',
};

function formatTokensCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

export function StatusBar({
  trailing,
  connected,
  state,
  sidebarOpen,
  onToggleSidebar,
  tokenUsage,
  currentStepInfo,
  currentActivity,
  activeModel,
  lastRunSuccess = null,
  onOpenPalette,
  onToggleWorkbench,
  workbenchOpen,
}: StatusBarProps) {
  const [showTokens, setShowTokens] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showTokens) return;
    const handleClick = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setShowTokens(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showTokens]);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '0 16px',
      height: 44,
      background: 'var(--bg-secondary)',
      borderBottom: '1px solid var(--border)',
      fontSize: 13,
      flexShrink: 0,
    }}>
      {/* Sidebar toggle */}
      {onToggleSidebar && (
        <button
          onClick={onToggleSidebar}
          title={sidebarOpen ? 'Hide sessions' : 'Show sessions'}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '4px 6px',
            borderRadius: 'var(--radius-sm)',
            color: sidebarOpen ? 'var(--text-primary)' : 'var(--text-muted)',
            fontSize: 15,
            lineHeight: 1,
            display: 'flex',
            alignItems: 'center',
            transition: 'color 0.15s',
          }}
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <rect x="2" y="3" width="14" height="12" rx="2" />
            <line x1="7" y1="3" x2="7" y2="15" />
          </svg>
        </button>
      )}

      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <PixelWolf
          state={
            !connected
              ? 'warning'
              : state === 'running'
                ? 'working'
                : state !== 'idle'
                  ? 'thinking'
                  : lastRunSuccess === true
                    ? 'passed'
                    : lastRunSuccess === false
                      ? 'failed'
                      : 'idle'
          }
          px={1.6}
          title={connected ? `RUNE (${STATE_LABELS[state]})` : 'RUNE — engine unreachable'}
        />
        <span style={{
          fontWeight: 700,
          fontSize: 14,
          letterSpacing: '1.5px',
          color: 'var(--accent)',
        }}>
          RUNE
        </span>
      </div>

      <div style={{ flex: 1 }} />

      {/* Agent state */}
      {state !== 'idle' && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          color: state === 'running' ? 'var(--accent)' : 'var(--warning)',
          fontWeight: 500,
        }}>
          <span
            className="status-dot status-dot--pulse"
            style={{
              width: 6,
              height: 6,
              background: state === 'running' ? 'var(--accent)' : 'var(--warning)',
            }}
          />
          {STATE_LABELS[state]}
          {state === 'running' && currentStepInfo && (
            <span style={{
              color: 'var(--text-muted)',
              fontWeight: 400,
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
            }}>
              {' \u00B7 '}step {currentStepInfo.stepNumber}
              {currentStepInfo.tokens > 0 && ` \u00B7 ${formatTokensCompact(currentStepInfo.tokens)}`}
            </span>
          )}
          {state === 'running' && currentActivity && (
            <span style={{
              color: 'var(--text-secondary)',
              fontWeight: 400,
              fontSize: 11,
              maxWidth: 200,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {' \u00B7 '}{currentActivity}
            </span>
          )}
        </div>
      )}

      {activeModel && (
        <div
          title={`Active model (${activeModel.source}): ${activeModel.provider}:${activeModel.model}`}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border)',
            borderRadius: '999px',
            color: 'var(--text-secondary)',
            fontSize: 11,
            minWidth: 0,
            maxWidth: 320,
          }}
        >
          <span style={{ color: 'var(--text-muted)' }}>Model</span>
          <span style={{
            fontFamily: 'var(--font-mono)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}>
            {activeModel.provider}:{activeModel.model}
          </span>
        </div>
      )}

      {/* Token usage compact */}
      {tokenUsage && (
        <div ref={popoverRef} style={{ position: 'relative' }}>
          <button
            onClick={() => setShowTokens(!showTokens)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              background: showTokens ? 'var(--bg-tertiary)' : 'transparent',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)',
              fontSize: 12,
              fontFamily: 'var(--font-mono)',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3">
              <circle cx="7" cy="7" r="5.5" />
              <path d="M5.5 5a1.5 1.5 0 0 1 3 0c0 1-1.5 1-1.5 2.5M7 9.5v0" strokeLinecap="round" />
            </svg>
            {formatTokensCompact(tokenUsage.total)}
          </button>

          {showTokens && (
            <div className="fade-scale" style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              marginTop: 6,
              width: 220,
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-lg)',
              boxShadow: 'var(--shadow-lg)',
              padding: '14px',
              zIndex: 100,
            }}>
              <TokenUsage usage={tokenUsage} />
            </div>
          )}
        </div>
      )}

      {/* Workbench toggle — always visible so the file/diff/terminal panel
          (and its Terminal tab) is discoverable without a shortcut. */}
      {onToggleWorkbench && (
        <button
          onClick={onToggleWorkbench}
          title="Toggle workbench: files, diff, terminal (⌘J)"
          aria-label="Toggle workbench"
          aria-pressed={workbenchOpen}
          style={{
            display: 'flex', alignItems: 'center', gap: 5, padding: '3px 8px',
            background: workbenchOpen ? 'var(--bg-tertiary)' : 'transparent',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            color: workbenchOpen ? 'var(--text-primary)' : 'var(--text-muted)',
            fontFamily: 'var(--font-mono)', fontSize: 11, cursor: 'pointer',
          }}
        >
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3">
            <rect x="1.5" y="2.5" width="11" height="9" rx="1" />
            <path d="M8.5 2.5v9" />
          </svg>
          Workbench
        </button>
      )}

      {/* Command palette */}
      {trailing}
      {onOpenPalette && (
        <button
          onClick={onOpenPalette}
          title="Command palette (⌘K)"
          aria-label="Open command palette"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '3px 8px',
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            cursor: 'pointer',
          }}
        >
          {'⌘K'}
        </button>
      )}

      {/* Connection indicator */}
      <div
        title={connected ? 'Connected' : 'Disconnected'}
        className={`status-dot ${connected ? 'status-dot--success' : 'status-dot--danger'}`}
      />
    </div>
  );
}
