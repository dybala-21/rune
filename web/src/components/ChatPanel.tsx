import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  ChatMessage,
  ToolCall,
  ThinkingBlock,
  ActivitySummary,
  DelegateItem,
  CompactionItem,
  StepInfo,
  PendingApproval,
} from '../types';
import { MessageBubble } from './MessageBubble';
import { ToolCallCard, getToolColor } from './ToolCallCard';
import { ThinkingBlockView } from './ThinkingBlock';
import { normalizeToolName, inferWorkPhase } from '../utils/tooling';
import { buildCompletionNarrative } from '../utils/completionSummary';
import {
  APPROVAL_COPY,
  QUESTION_COPY,
  getQuestionFieldPlaceholder,
} from '../shared/ui-copy.ts';

interface ChatPanelProps {
  messages: ChatMessage[];
  toolCalls: ToolCall[];
  thinkingBlocks: ThinkingBlock[];
  isRunning: boolean;
  activitySummary: ActivitySummary | null;
  delegateEvents: DelegateItem[];
  compactionEvents: CompactionItem[];
  currentStepInfo: StepInfo | null;
  pendingQuestion: {
    id: string;
    question: string;
    options?: Array<{ label: string; description?: string }>;
    inputMode?: 'text' | 'secret';
  } | null;
  onRespondQuestion: (answer: string, selectedIndex?: number) => void;
  pendingApproval: PendingApproval | null;
  onRespondApproval: (decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) => void;
}

export function ChatPanel({
  messages,
  toolCalls,
  thinkingBlocks,
  isRunning,
  activitySummary,
  delegateEvents,
  compactionEvents,
  currentStepInfo,
  pendingQuestion,
  onRespondQuestion,
  pendingApproval,
  onRespondApproval,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Small delay to ensure DOM is laid out before scrolling
    const timer = requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    });
    return () => cancelAnimationFrame(timer);
  }, [messages, toolCalls, thinkingBlocks, delegateEvents, compactionEvents, pendingApproval, pendingQuestion]);

  type TimelineItem =
    | { type: 'message'; item: ChatMessage }
    | { type: 'tool'; item: ToolCall }
    | { type: 'thinking'; item: ThinkingBlock }
    | { type: 'delegate'; item: DelegateItem }
    | { type: 'compaction'; item: CompactionItem };

  const timeline: TimelineItem[] = [
    ...messages
      .filter(m => m.content?.trim())
      .map(m => ({ type: 'message' as const, item: m, ts: m.timestamp })),
    ...toolCalls
      .filter(t => t.toolName?.trim())
      .map(t => ({ type: 'tool' as const, item: t, ts: t.timestamp })),
    ...thinkingBlocks
      .filter(t => t.text?.trim())
      .map(t => ({ type: 'thinking' as const, item: t, ts: t.timestamp })),
    ...delegateEvents
      .filter(d => d.stage?.trim() || d.message?.trim())
      .map(d => ({ type: 'delegate' as const, item: d, ts: d.timestamp })),
    ...compactionEvents
      .filter(c => c.message?.trim())
      .map(c => ({ type: 'compaction' as const, item: c, ts: c.timestamp })),
  ].sort((a, b) => a.ts - b.ts);

  const grouped = groupConsecutiveTools(timeline);

  const latestToolId = useMemo(() => {
    if (!isRunning || toolCalls.length === 0) return null;
    return toolCalls[toolCalls.length - 1].id;
  }, [isRunning, toolCalls]);

  const isEmpty = timeline.length === 0;

  return (
    <div ref={scrollContainerRef} style={{
      flex: 1,
      overflowY: 'auto',
      overflowX: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        maxWidth: 768,
        width: '100%',
        margin: '0 auto',
        padding: '24px 24px 160px',
        flex: isEmpty ? 1 : undefined,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}>
        {isEmpty && <EmptyState />}

        {grouped.map((entry, idx) => {
          if (entry.type === 'group') {
            return (
              <ToolGroup
                key={entry.key}
                tools={entry.tools}
                isLatest={latestToolId != null && entry.tools.some(t => t.id === latestToolId)}
                latestToolId={latestToolId}
              />
            );
          }
          const item = entry.item;
          if (item.type === 'message') {
            const prevItem = idx > 0 ? grouped[idx - 1] : null;
            const needsGap = prevItem && (prevItem.type === 'group' || (prevItem.type === 'single' && prevItem.item.type !== 'message'));
            return (
              <div key={item.item.id} style={{ marginTop: needsGap ? 12 : 0 }}>
                <MessageBubble message={item.item} />
              </div>
            );
          }
          if (item.type === 'tool') {
            const tc = item.item;
            if (tc.toolName === 'ask_user') {
              return (
                <InlineQuestionCard
                  key={tc.id}
                  toolCall={tc}
                  pendingQuestion={pendingQuestion}
                  onRespond={onRespondQuestion}
                />
              );
            }
            return (
              <ToolCallCard
                key={tc.id}
                toolCall={tc}
                isLatest={tc.id === latestToolId}
              />
            );
          }
          if (item.type === 'delegate') {
            return <DelegateNotice key={item.item.id} event={item.item} />;
          }
          if (item.type === 'compaction') {
            return <CompactionNotice key={item.item.id} event={item.item} />;
          }
          return <ThinkingBlockView key={item.item.id} block={item.item} />;
        })}

        {pendingApproval && (
          <InlineApprovalCard
            approval={pendingApproval}
            onRespond={onRespondApproval}
          />
        )}

        {isRunning && !pendingApproval && (
          <RunningIndicator toolCalls={toolCalls} currentStepInfo={currentStepInfo} />
        )}

        {!isRunning && activitySummary && (
          <>
            <CompletionNotice summary={activitySummary} />
            {activitySummary.totalToolCalls > 0 && (
              <ActivitySummaryBar summary={activitySummary} />
            )}
          </>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ── Empty state ──

function EmptyState() {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 16,
      padding: '40px 20px',
    }}>
      <div style={{
        width: 48,
        height: 48,
        borderRadius: 'var(--radius-lg)',
        background: 'var(--accent-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 20,
        fontWeight: 700,
        color: 'var(--accent)',
        letterSpacing: '1px',
      }}>
        R
      </div>
      <div style={{
        fontSize: 18,
        fontWeight: 600,
        color: 'var(--text-primary)',
      }}>
        How can I help you?
      </div>
      <div style={{
        fontSize: 14,
        color: 'var(--text-muted)',
        textAlign: 'center',
        maxWidth: 360,
      }}>
        RUNE can help with coding, analysis, web searches, file management, and more.
      </div>
      <div style={{
        display: 'flex',
        gap: 8,
        flexWrap: 'wrap',
        justifyContent: 'center',
        marginTop: 8,
      }}>
        {[
          'Analyze my project',
          'Search the web',
          'Read and edit files',
        ].map(text => (
          <div key={text} style={{
            padding: '8px 16px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            fontSize: 13,
            color: 'var(--text-secondary)',
            cursor: 'default',
          }}>
            {text}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Running indicator ──

const TOOL_ACTIVITIES: Record<string, string> = {
  file_read: 'Reviewing files...',
  file_write: 'Updating files...',
  file_edit: 'Editing code...',
  file_list: 'Scanning the workspace...',
  file_search: 'Searching the project...',
  bash: 'Running a command...',
  web_search: 'Looking things up...',
  web_fetch: 'Checking the source...',
  browser_navigate: 'Opening the page...',
  browser_observe: 'Inspecting the page...',
  browser_act: 'Working through the page...',
  think: 'Working through the request...',
  memory_search: 'Checking past context...',
  memory_save: 'Saving context...',
  project_map: 'Mapping the project...',
  code_analyze: 'Tracing the code path...',
  delegate_task: 'Handing off a subtask...',
  delegate_orchestrate: 'Coordinating the work...',
};

const PHASE_LABELS: Record<string, string> = {
  analyzing: 'Analyzing',
  implementing: 'Implementing',
  verifying: 'Verifying',
};

const PHASE_COLORS: Record<string, string> = {
  analyzing: 'var(--accent)',
  implementing: 'var(--warning)',
  verifying: 'var(--success)',
};

function RunningIndicator({ toolCalls, currentStepInfo }: { toolCalls: ToolCall[]; currentStepInfo: StepInfo | null }) {
  const pendingTool = [...toolCalls].reverse().find(tc => tc.result === undefined);
  const activityKey = pendingTool ? normalizeToolName(pendingTool.toolName).replace(/\./g, '_') : null;
  const activity = activityKey ? TOOL_ACTIVITIES[activityKey] : null;
  const phase = toolCalls.length > 0 ? inferWorkPhase(toolCalls) : null;

  return (
    <div className="fade-in" style={{
      padding: '10px 0',
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      color: 'var(--text-secondary)',
    }}>
      <span className="spinner" />
      <span style={{ fontSize: 13, fontWeight: 500 }}>
        {activity || 'Working through the request...'}
      </span>
      {currentStepInfo && (
        <span style={{
          fontSize: 11,
          color: 'var(--text-muted)',
          marginLeft: 'auto',
          fontFamily: 'var(--font-mono)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          {phase && (
            <span style={{
              color: PHASE_COLORS[phase],
              fontWeight: 600,
              padding: '1px 6px',
              borderRadius: 3,
              background: `color-mix(in srgb, ${PHASE_COLORS[phase]} 15%, transparent)`,
            }}>
              {PHASE_LABELS[phase]}
            </span>
          )}
          Step {currentStepInfo.stepNumber}
          {currentStepInfo.tokens > 0 && ` \u00B7 ${formatTokens(currentStepInfo.tokens)}`}
        </span>
      )}
    </div>
  );
}

// ── Activity summary ──

function CompletionNotice({ summary }: { summary: ActivitySummary }) {
  return (
    <div className="fade-in" style={{
      marginTop: 8,
      padding: '2px 0 0',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      color: 'var(--text-secondary)',
    }}>
      <span style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: summary.success ? 'var(--success)' : 'var(--danger)',
        flexShrink: 0,
      }} />
      <span style={{
        fontSize: 14,
        color: 'var(--text-primary)',
      }}>
        {buildCompletionNarrative(summary)}
      </span>
    </div>
  );
}

function ActivitySummaryBar({ summary }: { summary: ActivitySummary }) {
  const items: Array<{ label: string; value: number }> = [];
  if (summary.totalToolCalls > 0) items.push({ label: 'Tools', value: summary.totalToolCalls });
  if (summary.filesRead > 0) items.push({ label: 'Read', value: summary.filesRead });
  if (summary.filesWritten > 0) items.push({ label: 'Write', value: summary.filesWritten });
  if (summary.bashExecutions > 0) items.push({ label: 'Bash', value: summary.bashExecutions });
  if (summary.webSearches > 0) items.push({ label: 'Web', value: summary.webSearches });
  if (summary.browserActions > 0) items.push({ label: 'Browser', value: summary.browserActions });

  return (
    <div className="fade-in" style={{
      marginTop: 8,
      padding: '10px 14px',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-md)',
      display: 'flex',
      alignItems: 'center',
      gap: 16,
      fontSize: 12,
    }}>
      {items.map(item => (
        <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ color: 'var(--text-muted)' }}>{item.label}</span>
          <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {item.value}
          </span>
        </div>
      ))}
      {summary.totalDurationMs > 0 && (
        <span style={{
          marginLeft: 'auto',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          {formatDuration(summary.totalDurationMs)}
        </span>
      )}
    </div>
  );
}

// ── Delegate / Compaction notices ──

function DelegateNotice({ event }: { event: DelegateItem }) {
  if (!event.stage?.trim() && !event.message?.trim()) return null;
  return (
    <div className="fade-in" style={{
      padding: '6px 12px',
      fontSize: 12,
      color: 'var(--text-muted)',
      borderLeft: '2px solid var(--accent)',
      marginLeft: 4,
      marginTop: 2,
    }}>
      <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{event.stage}</span>
      {event.message && <span style={{ marginLeft: 6 }}>{event.message}</span>}
    </div>
  );
}

function CompactionNotice({ event }: { event: CompactionItem }) {
  if (!event.message?.trim()) return null;
  return (
    <div className="fade-in" style={{
      padding: '6px 12px',
      fontSize: 12,
      color: 'var(--warning)',
      borderLeft: '2px solid var(--warning)',
      marginLeft: 4,
      marginTop: 2,
    }}>
      {event.message}
    </div>
  );
}

// ── Tool grouping ──

type GroupedEntry =
  | { type: 'single'; item: TimelineItemType }
  | { type: 'group'; key: string; toolName: string; tools: ToolCall[] };

type TimelineItemType =
  | { type: 'message'; item: ChatMessage }
  | { type: 'tool'; item: ToolCall }
  | { type: 'thinking'; item: ThinkingBlock }
  | { type: 'delegate'; item: DelegateItem }
  | { type: 'compaction'; item: CompactionItem };

function groupConsecutiveTools(timeline: TimelineItemType[]): GroupedEntry[] {
  const result: GroupedEntry[] = [];
  let currentGroup: ToolCall[] = [];
  let currentToolName = '';

  const flushGroup = () => {
    if (currentGroup.length >= 3) {
      result.push({
        type: 'group',
        key: `group-${currentGroup[0].id}`,
        toolName: currentToolName,
        tools: currentGroup,
      });
    } else {
      for (const tc of currentGroup) {
        result.push({ type: 'single', item: { type: 'tool', item: tc } });
      }
    }
    currentGroup = [];
    currentToolName = '';
  };

  for (const entry of timeline) {
    if (entry.type === 'tool') {
      if (entry.item.toolName === currentToolName) {
        currentGroup.push(entry.item);
      } else {
        flushGroup();
        currentToolName = entry.item.toolName;
        currentGroup = [entry.item];
      }
    } else {
      flushGroup();
      result.push({ type: 'single', item: entry });
    }
  }
  flushGroup();

  return result;
}

function ToolGroup({ tools, isLatest = false, latestToolId }: { tools: ToolCall[]; isLatest?: boolean; latestToolId: string | null }) {
  const [manualOpen, setManualOpen] = useState<boolean | null>(null);

  const prevIsLatest = useRef(isLatest);
  useEffect(() => {
    if (prevIsLatest.current && !isLatest) {
      setManualOpen(null);
    }
    prevIsLatest.current = isLatest;
  }, [isLatest]);

  const open = manualOpen ?? false;
  const completed = tools.filter(t => t.result !== undefined).length;
  const toolName = tools[0].toolName;
  const toolColor = getToolColor(toolName);

  const handleToggle = () => {
    setManualOpen(!(manualOpen ?? false));
  };

  return (
    <div className="fade-in" style={{
      margin: '2px 0',
      border: '1px solid var(--border-subtle)',
      borderLeft: `2px solid ${toolColor}`,
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
    }}>
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={handleToggle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') handleToggle(); }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '7px 12px',
          color: 'var(--text-primary)',
          fontSize: 13,
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--text-secondary)',
        }}>
          {toolName}
        </span>
        <span style={{
          fontSize: 11,
          padding: '1px 7px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-tertiary)',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          {completed}/{tools.length}
        </span>
        <span style={{
          marginLeft: 'auto',
          fontSize: 10,
          color: 'var(--text-muted)',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }}>
          {'\u25BC'}
        </span>
      </div>
      {open && (
        <div className="expand-content" style={{ padding: '2px 6px 6px' }}>
          {tools.map(tc => (
            <ToolCallCard
              key={tc.id}
              toolCall={tc}
              isLatest={tc.id === latestToolId}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Utilities ──

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const min = Math.floor(ms / 60_000);
  const sec = Math.round((ms % 60_000) / 1000);
  return `${min}m${sec}s`;
}

function formatTokens(tokens: number): string {
  if (tokens < 1000) return `${tokens}`;
  return `${(tokens / 1000).toFixed(1)}k`;
}

// ── Inline question card (ask_user) ──

function InlineQuestionCard({
  toolCall,
  pendingQuestion,
  onRespond,
}: {
  toolCall: ToolCall;
  pendingQuestion: {
    id: string;
    question: string;
    options?: Array<{ label: string; description?: string }>;
    inputMode?: 'text' | 'secret';
  } | null;
  onRespond: (answer: string, selectedIndex?: number) => void;
}) {
  const [freeText, setFreeText] = useState('');
  const isPending = toolCall.result === undefined;

  const questionText = pendingQuestion?.question
    || (toolCall.args.question as string)
    || '';
  const options = pendingQuestion?.options
    || (toolCall.args.options as Array<{ label: string; description?: string }>)
    || [];
  const inputMode = pendingQuestion?.inputMode
    || (toolCall.args.inputMode as 'text' | 'secret' | undefined)
    || 'text';

  const handleOptionClick = (opt: { label: string; description?: string }, idx: number) => {
    if (isPending) {
      onRespond(opt.label, idx);
    }
  };

  const handleFreeTextSubmit = () => {
    if (freeText.trim()) {
      onRespond(freeText.trim());
      setFreeText('');
    }
  };

  return (
    <div className="slide-up" style={{
      margin: '4px 0',
      border: `1px solid ${isPending ? 'var(--accent)' : 'var(--border)'}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
      position: 'relative',
      zIndex: isPending ? 20 : 1,
    }}>
      <div style={{
        padding: '10px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <span style={{
          width: 20,
          height: 20,
          borderRadius: '50%',
          background: 'var(--accent-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 11,
          color: 'var(--accent)',
          flexShrink: 0,
        }}>?</span>
        <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--accent)' }}>
          {QUESTION_COPY.title}
        </span>
        {!isPending && (
          <span style={{
            marginLeft: 'auto',
            fontSize: 11,
            padding: '2px 8px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--success-subtle)',
            color: 'var(--success)',
            fontWeight: 600,
          }}>
            {QUESTION_COPY.answeredLabel}
          </span>
        )}
        {isPending && (
          <span style={{ marginLeft: 'auto' }}>
            <span className="status-dot status-dot--warning status-dot--pulse" />
          </span>
        )}
      </div>

      <div style={{
        padding: '12px 16px',
        fontSize: 14,
        color: 'var(--text-primary)',
        lineHeight: 1.6,
        whiteSpace: 'pre-wrap',
      }}>
        {questionText}
      </div>

      {options.length > 0 && (
        <div style={{ padding: '0 16px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {options.map((opt, idx) => (
            <button
              key={idx}
              onClick={() => handleOptionClick(opt, idx)}
              disabled={!isPending}
              style={{
                padding: '10px 14px',
                background: isPending ? 'var(--bg-tertiary)' : 'var(--bg-primary)',
                color: 'var(--text-primary)',
                borderRadius: 'var(--radius-md)',
                textAlign: 'left',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                gap: 10,
                alignItems: 'baseline',
                cursor: isPending ? 'pointer' : 'default',
                opacity: isPending ? 1 : 0.5,
                transition: 'background 0.15s',
                pointerEvents: isPending ? 'auto' : 'none',
              }}
            >
              <span style={{
                fontSize: 12,
                color: 'var(--accent)',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                minWidth: 20,
              }}>
                {idx + 1}.
              </span>
              <div>
                <div style={{ fontWeight: 500, fontSize: 14 }}>{opt.label}</div>
                {opt.description && (
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3, lineHeight: 1.5 }}>
                    {opt.description}
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
      )}

      {isPending && (
        <div style={{ padding: '0 16px 12px', display: 'flex', gap: 8 }}>
          <input
            type={inputMode === 'secret' ? 'password' : 'text'}
            value={freeText}
            onChange={e => setFreeText(e.target.value)}
            placeholder={getQuestionFieldPlaceholder(inputMode)}
            autoComplete={inputMode === 'secret' ? 'off' : undefined}
            spellCheck={inputMode === 'secret' ? false : undefined}
            style={{
              flex: 1,
              padding: '9px 14px',
              background: 'var(--bg-primary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontSize: 13,
            }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.nativeEvent.isComposing && freeText.trim()) {
                handleFreeTextSubmit();
              }
            }}
          />
          <button
            onClick={handleFreeTextSubmit}
            disabled={!freeText.trim()}
            style={{
              padding: '9px 16px',
              background: freeText.trim() ? 'var(--accent)' : 'var(--bg-tertiary)',
              color: freeText.trim() ? 'white' : 'var(--text-muted)',
              borderRadius: 'var(--radius-md)',
              fontWeight: 600,
              fontSize: 13,
            }}
          >
            {QUESTION_COPY.submitLabel}
          </button>
        </div>
      )}

      {!isPending && toolCall.result && (
        <div style={{
          padding: '8px 16px 12px',
          fontSize: 12,
          color: 'var(--text-secondary)',
        }}>
          <span style={{ fontWeight: 600, marginRight: 6, color: 'var(--text-muted)' }}>Answer:</span>
          {toolCall.result.length > 200 ? toolCall.result.slice(0, 200) + '...' : toolCall.result}
        </div>
      )}
    </div>
  );
}

// ── Inline approval card ──

const RISK_COLORS: Record<string, string> = {
  low: 'var(--success)',
  medium: 'var(--warning)',
  high: 'var(--danger)',
  critical: '#ff4444',
};

function InlineApprovalCard({
  approval,
  onRespond,
}: {
  approval: PendingApproval;
  onRespond: (decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) => void;
}) {
  const [showDenyInput, setShowDenyInput] = useState(false);
  const [guidance, setGuidance] = useState('');
  const [remaining, setRemaining] = useState(Math.ceil(approval.timeoutMs / 1000));

  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = Date.now() - approval.receivedAt;
      const left = Math.max(0, Math.ceil((approval.timeoutMs - elapsed) / 1000));
      setRemaining(left);
      if (left <= 0) clearInterval(interval);
    }, 1000);
    return () => clearInterval(interval);
  }, [approval]);

  const riskColor = RISK_COLORS[approval.riskLevel] || 'var(--text-secondary)';
  const progressPct = Math.max(0, (remaining / (approval.timeoutMs / 1000)) * 100);

  return (
    <div className="slide-up" style={{
      margin: '4px 0',
      border: `1px solid ${riskColor}`,
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
      position: 'relative',
      zIndex: 20,
    }}>
      {/* Timeout bar */}
      <div style={{ height: 2, background: 'var(--bg-tertiary)' }}>
        <div style={{
          height: '100%',
          width: `${progressPct}%`,
          background: riskColor,
          transition: 'width 1s linear',
        }} />
      </div>

      <div style={{
        padding: '10px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke={riskColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M8 1.5l6.5 12H1.5z" />
          <line x1="8" y1="6" x2="8" y2="9" />
          <circle cx="8" cy="11" r="0.5" fill={riskColor} />
        </svg>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{APPROVAL_COPY.title}</span>
        <span style={{
          fontSize: 10,
          padding: '2px 8px',
          borderRadius: 'var(--radius-sm)',
          background: riskColor + '18',
          color: riskColor,
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}>
          {approval.riskLevel}
        </span>
        <span style={{
          marginLeft: 'auto',
          fontSize: 12,
          color: remaining <= 10 ? 'var(--danger)' : 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          {remaining}s
        </span>
      </div>

      <div style={{ padding: '12px 16px' }}>
        <div style={{
          fontSize: 11,
          color: 'var(--text-muted)',
          marginBottom: 4,
          fontWeight: 600,
          letterSpacing: '0.3px',
        }}>
          {APPROVAL_COPY.commandLabel}
        </div>
        <pre style={{
          padding: '10px 14px',
          background: 'var(--code-bg)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          fontSize: 13,
          overflow: 'auto',
          maxHeight: 120,
          whiteSpace: 'pre-wrap',
          marginBottom: 10,
          color: 'var(--text-primary)',
        }}>
          {approval.command}
        </pre>

        {approval.reason && (
          <div style={{ marginBottom: 12 }}>
            <div style={{
              fontSize: 11,
              color: 'var(--text-muted)',
              marginBottom: 4,
              fontWeight: 600,
              letterSpacing: '0.3px',
            }}>
              {APPROVAL_COPY.reasonLabel}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {approval.reason}
            </div>
          </div>
        )}

        {approval.suggestions && approval.suggestions.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{
              fontSize: 11,
              color: 'var(--text-muted)',
              marginBottom: 4,
              fontWeight: 600,
              letterSpacing: '0.3px',
            }}>
              {APPROVAL_COPY.suggestionsLabel}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {approval.suggestions.map((suggestion, idx) => (
                <div
                  key={`${idx}-${suggestion}`}
                  style={{
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                    lineHeight: 1.5,
                  }}
                >
                  {'\u2022'} {suggestion}
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{
          fontSize: 11,
          color: 'var(--text-muted)',
          marginBottom: 6,
          fontWeight: 600,
          letterSpacing: '0.3px',
        }}>
          {APPROVAL_COPY.actionsLabel}
        </div>

        {showDenyInput ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <input
              value={guidance}
              onChange={e => setGuidance(e.target.value)}
              placeholder={APPROVAL_COPY.denialPlaceholder}
              autoFocus
              style={{
                flex: 1,
                padding: '9px 14px',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontSize: 13,
              }}
              onKeyDown={e => {
                if (e.key === 'Enter') onRespond('deny', guidance || undefined);
                if (e.key === 'Escape') { setShowDenyInput(false); setGuidance(''); }
              }}
            />
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => onRespond('deny', guidance || undefined)}
                style={{
                  padding: '9px 18px',
                  background: 'var(--danger)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: 600,
                }}
              >
                {APPROVAL_COPY.denyWithInstructionsLabel}
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => onRespond('approve_once')}
              style={{
                flex: 1,
                padding: '9px 16px',
                background: 'var(--success)',
                color: 'var(--bg-primary)',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                fontWeight: 600,
                fontSize: 13,
              }}
            >
              {APPROVAL_COPY.allowOnceLabel}
            </button>
            <button
              onClick={() => onRespond('approve_always')}
              style={{
                flex: 1,
                padding: '9px 16px',
                background: 'transparent',
                color: 'var(--success)',
                borderRadius: 'var(--radius-md)',
                fontWeight: 600,
                border: '1px solid var(--success)',
                fontSize: 13,
              }}
            >
              {APPROVAL_COPY.alwaysAllowLabel}
            </button>
            <button
              onClick={() => setShowDenyInput(true)}
              style={{
                flex: 1,
                padding: '9px 16px',
                background: 'transparent',
                color: 'var(--danger)',
                borderRadius: 'var(--radius-md)',
                fontWeight: 600,
                border: '1px solid var(--danger)',
                fontSize: 13,
              }}
            >
              {APPROVAL_COPY.denyLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
