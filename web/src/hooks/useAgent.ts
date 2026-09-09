import { useCallback, useEffect, useRef, useState } from 'react';
import { restoreRunMessages, type RunSnapshot } from '../utils/runSnapshot';
import { toast } from '../utils/toast';
import { useSSE } from './useSSE';
import * as api from '../api';
import { computeActivitySummary } from '../utils/tooling';
import { describeTrust } from '../utils/trust';
import { abortedMessage, belongsToConversation, upsertRunMessage } from '../utils/runEvents';
import type {
  AgentState,
  ChatMessage,
  FileChange,
  ToolCall,
  ThinkingBlock,
  TokenUsage,
  PendingApproval,
  PendingQuestion,
  PendingAttachment,
  ActivitySummary,
  DelegateItem,
  CompactionItem,
  StepInfo,
  AgentCompleteData,
  AgentErrorData,
  AgentAbortedData,
  AgentStartData,
  ToolCallData,
  ToolResultData,
  ThinkingData,
  TextDeltaData,
  ApprovalRequestData,
  QuestionData,
  StepStartData,
  ContextCompactionData,
  DelegateEventData,
  CommandResultData,
  GoalIterationData,
  OrchestrationStartedData,
  OrchestrationTaskProgressData,
  OrchestrationTaskRetryData,
  OrchestrationCompletedData,
  OrchestrationState,
  OrchestrationTask,
  TrustInfo,
  ProactiveSuggestion,
  SseEventType,
} from '../types';

let idCounter = 0;
function nextId(): string {
  return `msg-${++idCounter}-${Date.now()}`;
}

/** Merge progress updates for the same delegated task. */
function upsertTask(
  tasks: OrchestrationTask[],
  taskId: string,
  patch: Partial<OrchestrationTask>,
): OrchestrationTask[] {
  const idx = tasks.findIndex(t => t.taskId === taskId);
  if (idx === -1) {
    return [...tasks, { taskId, description: '', role: '', retries: 0, ...patch }];
  }
  const next = [...tasks];
  next[idx] = { ...next[idx], ...patch };
  return next;
}

const LIVE_STATE_STORAGE_KEY = 'rune:web:live-state:v1';
const MAX_MESSAGES = 1200;
const MAX_TOOL_CALLS = 3000;
const MAX_THINKING_BLOCKS = 1200;
const MAX_DELEGATE_EVENTS = 1200;
const MAX_COMPACTION_EVENTS = 600;

interface PersistedLiveState {
  version: 1;
  messages: ChatMessage[];
  toolCalls: ToolCall[];
  thinkingBlocks: ThinkingBlock[];
  tokenUsage: TokenUsage | null;
  activitySummary: ActivitySummary | null;
  lastTrust?: TrustInfo | null;
  delegateEvents: DelegateItem[];
  compactionEvents: CompactionItem[];
}

interface PersistedLiveEnvelope {
  version: 1;
  savedAt: number;
  state: PersistedLiveState;
}

interface SavedLiveDraftSummary {
  available: boolean;
  savedAt: number | null;
  messageCount: number;
  toolCallCount: number;
  thinkingCount: number;
}

interface LoadedLiveDraft {
  state: PersistedLiveState;
  savedAt: number | null;
}

const EMPTY_SAVED_DRAFT: SavedLiveDraftSummary = {
  available: false,
  savedAt: null,
  messageCount: 0,
  toolCallCount: 0,
  thinkingCount: 0,
};

function createEmptyLiveState(): PersistedLiveState {
  return {
    version: 1,
    messages: [],
    toolCalls: [],
    thinkingBlocks: [],
    tokenUsage: null,
    activitySummary: null,
    delegateEvents: [],
    compactionEvents: [],
  };
}

function trimTail<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  return items.slice(items.length - limit);
}

function appendWithLimit<T>(items: T[], item: T, limit: number): T[] {
  if (items.length < limit) return [...items, item];
  return [...items.slice(items.length - limit + 1), item];
}

function isLiveStateEmpty(state: PersistedLiveState): boolean {
  return state.messages.length === 0
    && state.toolCalls.length === 0
    && state.thinkingBlocks.length === 0
    && state.delegateEvents.length === 0
    && state.compactionEvents.length === 0;
}

function summarizeSavedDraft(state: PersistedLiveState, savedAt: number | null): SavedLiveDraftSummary {
  if (isLiveStateEmpty(state)) return EMPTY_SAVED_DRAFT;
  return {
    available: true,
    savedAt,
    messageCount: state.messages.length,
    toolCallCount: state.toolCalls.length,
    thinkingCount: state.thinkingBlocks.length,
  };
}

function loadPersistedLiveState(): LoadedLiveDraft {
  if (typeof window === 'undefined') {
    return {
      state: createEmptyLiveState(),
      savedAt: null,
    };
  }

  try {
    const raw = window.localStorage.getItem(LIVE_STATE_STORAGE_KEY);
    if (!raw) {
      return {
        state: createEmptyLiveState(),
        savedAt: null,
      };
    }

    const parsed = JSON.parse(raw) as (Partial<PersistedLiveEnvelope> & Partial<PersistedLiveState>) | null;
    if (!parsed || typeof parsed !== 'object') {
      return {
        state: createEmptyLiveState(),
        savedAt: null,
      };
    }

    // legacy format support: state가 바로 루트에 저장되던 포맷
    const stateCandidate = (parsed.state && typeof parsed.state === 'object')
      ? parsed.state as Partial<PersistedLiveState>
      : parsed as Partial<PersistedLiveState>;
    const savedAt = typeof parsed.savedAt === 'number' ? parsed.savedAt : null;

    return {
      state: {
        version: 1,
        messages: trimTail(Array.isArray(stateCandidate.messages) ? stateCandidate.messages as ChatMessage[] : [], MAX_MESSAGES),
        toolCalls: trimTail(Array.isArray(stateCandidate.toolCalls) ? stateCandidate.toolCalls as ToolCall[] : [], MAX_TOOL_CALLS),
        thinkingBlocks: trimTail(Array.isArray(stateCandidate.thinkingBlocks) ? stateCandidate.thinkingBlocks as ThinkingBlock[] : [], MAX_THINKING_BLOCKS),
        tokenUsage: stateCandidate.tokenUsage ?? null,
        activitySummary: stateCandidate.activitySummary ?? null,
        lastTrust: stateCandidate.lastTrust ?? null,
        delegateEvents: trimTail(Array.isArray(stateCandidate.delegateEvents) ? stateCandidate.delegateEvents as DelegateItem[] : [], MAX_DELEGATE_EVENTS),
        compactionEvents: trimTail(Array.isArray(stateCandidate.compactionEvents) ? stateCandidate.compactionEvents as CompactionItem[] : [], MAX_COMPACTION_EVENTS),
      },
      savedAt,
    };
  } catch {
    return {
      state: createEmptyLiveState(),
      savedAt: null,
    };
  }
}

/** Keep attachment names in drafts; base64 data can exceed localStorage's quota. */
function withoutAttachmentData(messages: ChatMessage[]): ChatMessage[] {
  return messages.map(m => (
    m.attachments?.length
      ? { ...m, attachments: m.attachments.map(({ name, mimeType }) => ({ name, mimeType })) }
      : m
  ));
}

function persistLiveState(state: PersistedLiveState): void {
  if (typeof window === 'undefined') return;
  try {
    const payload: PersistedLiveEnvelope = {
      version: 1,
      savedAt: Date.now(),
      state: { ...state, messages: withoutAttachmentData(state.messages) },
    };
    window.localStorage.setItem(LIVE_STATE_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // ignore quota / storage errors
  }
}

export function useAgent() {
  const { connected, addEventListener: sseOn, refresh } = useSSE();
  const initialStateRef = useRef<LoadedLiveDraft | null>(null);
  if (initialStateRef.current === null) {
    initialStateRef.current = loadPersistedLiveState();
  }
  const initialState = initialStateRef.current;
  const initialLiveState = createEmptyLiveState();
  const hasSavedDraft = !isLiveStateEmpty(initialState.state);
  const savedDraftStateRef = useRef<PersistedLiveState | null>(
    hasSavedDraft ? initialState.state : null,
  );

  const [state, setState] = useState<AgentState>('idle');
  const [messages, setMessages] = useState<ChatMessage[]>(initialLiveState.messages);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>(initialLiveState.toolCalls);
  const [thinkingBlocks, setThinkingBlocks] = useState<ThinkingBlock[]>(initialLiveState.thinkingBlocks);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(initialLiveState.tokenUsage);
  const [pendingApproval, setPendingApprovalState] = useState<PendingApproval | null>(null);
  const pendingApprovalRef = useRef<PendingApproval | null>(null);
  const setPendingApproval = useCallback((update: PendingApproval | null | ((current: PendingApproval | null) => PendingApproval | null)) => {
    const next = typeof update === 'function' ? update(pendingApprovalRef.current) : update;
    pendingApprovalRef.current = next;
    setPendingApprovalState(next);
  }, []);
  const [pendingQuestion, setPendingQuestion] = useState<PendingQuestion | null>(null);
  const pendingQuestionRef = useRef<PendingQuestion | null>(null);
  const [activitySummary, setActivitySummary] = useState<ActivitySummary | null>(initialLiveState.activitySummary);
  const [lastTrust, setLastTrust] = useState<TrustInfo | null>(null);
  const [fileChanges, setFileChanges] = useState<FileChange[]>([]);
  const [interruptedRun, setInterruptedRun] = useState<RunSnapshot | null>(null);
  const [delegateEvents, setDelegateEvents] = useState<DelegateItem[]>(initialLiveState.delegateEvents);
  const [compactionEvents, setCompactionEvents] = useState<CompactionItem[]>(initialLiveState.compactionEvents);
  const [currentStepInfo, setCurrentStepInfo] = useState<StepInfo | null>(null);
  const [orchestration, setOrchestration] = useState<OrchestrationState | null>(null);
  // tool_call 핸들러가 스텝 번호를 동기적으로 읽어야 하므로 state와 별도로 ref 유지
  const currentStepRef = useRef(0);
  // Step numbers restart each run. Continue from the restored run counter
  // so the timeline keeps calls from different turns apart.
  const runSeqRef = useRef(
    initialState.state.toolCalls.reduce((max, tc) => Math.max(max, tc.run ?? 0), 0),
  );
  const [savedDraft, setSavedDraft] = useState<SavedLiveDraftSummary>(
    summarizeSavedDraft(initialState.state, initialState.savedAt),
  );
  const [draftDecisionPending, setDraftDecisionPending] = useState<boolean>(hasSavedDraft);

  // Accumulated text for the current answer.
  const pendingTextRef = useRef('');
  const pendingPersistRef = useRef<PersistedLiveState | null>(null);
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // 현재 run의 assistant 메시지 ID (하나의 run에 하나의 assistant 메시지만 유지)
  const assistantMsgIdRef = useRef<string | null>(null);

  // Read the message ID inside the updater so batched deltas share one message.
  const flushTextDelta = useCallback(() => {
    const text = pendingTextRef.current;
    if (!text) return;

    setMessages(prev => {
      const msgId = assistantMsgIdRef.current;
      if (msgId) {
        const idx = prev.findIndex(m => m.id === msgId);
        if (idx !== -1 && prev[idx].content !== text) {
          const updated = [...prev];
          updated[idx] = { ...updated[idx], content: text };
          return updated;
        }
        return prev;
      }
      const newId = nextId();
      assistantMsgIdRef.current = newId;
      return appendWithLimit(prev, { id: newId, role: 'assistant' as const, content: text, timestamp: Date.now() }, MAX_MESSAGES);
    });
  }, []);

  // Refs keep beginLiveSession stable, avoiding SSE resubscriptions mid-run.
  const draftDecisionPendingRef = useRef(draftDecisionPending);
  const savedDraftAvailableRef = useRef(savedDraft.available);
  useEffect(() => { draftDecisionPendingRef.current = draftDecisionPending; }, [draftDecisionPending]);
  useEffect(() => { savedDraftAvailableRef.current = savedDraft.available; }, [savedDraft.available]);

  const beginLiveSession = useCallback(() => {
    if (!draftDecisionPendingRef.current && !savedDraftAvailableRef.current) return;
    savedDraftStateRef.current = null;
    setSavedDraft(EMPTY_SAVED_DRAFT);
    setDraftDecisionPending(false);
  }, []);

  const restoreSavedDraft = useCallback(() => {
    const draft = savedDraftStateRef.current;
    if (!draft) return;
    const restoredCalls = trimTail(draft.toolCalls, MAX_TOOL_CALLS);
    // Reserve the run numbers already used by restored calls.
    runSeqRef.current = restoredCalls.reduce((max, tc) => Math.max(max, tc.run ?? 0), 0);
    setMessages(trimTail(draft.messages, MAX_MESSAGES));
    setToolCalls(restoredCalls);
    setThinkingBlocks(trimTail(draft.thinkingBlocks, MAX_THINKING_BLOCKS));
    setTokenUsage(draft.tokenUsage ?? null);
    setActivitySummary(draft.activitySummary ?? null);
    setLastTrust(draft.lastTrust ?? null);
    setDelegateEvents(trimTail(draft.delegateEvents, MAX_DELEGATE_EVENTS));
    setCompactionEvents(trimTail(draft.compactionEvents, MAX_COMPACTION_EVENTS));
    savedDraftStateRef.current = null;
    setSavedDraft(EMPTY_SAVED_DRAFT);
    setDraftDecisionPending(false);
  }, []);

  const discardSavedDraft = useCallback(() => {
    if (!savedDraft.available) return;
    savedDraftStateRef.current = null;
    setSavedDraft(EMPTY_SAVED_DRAFT);
    setDraftDecisionPending(false);
    persistLiveState(createEmptyLiveState());
  }, [savedDraft.available]);

  // Shared reset for New Chat and /load; callers choose the next session ID.
  const clearConversationState = useCallback(() => {
    api.setCurrentRunId('');
    pendingTextRef.current = '';
    assistantMsgIdRef.current = null;
    savedDraftStateRef.current = null;
    currentStepRef.current = 0;
    runSeqRef.current = 0;
    setState('idle');
    setMessages([]);
    setToolCalls([]);
    setThinkingBlocks([]);
    setTokenUsage(null);
    setPendingApproval(null);
    pendingQuestionRef.current = null;
    setPendingQuestion(null);
    setActivitySummary(null);
    setLastTrust(null);
    setFileChanges([]);
    setInterruptedRun(null);
    setDelegateEvents([]);
    setCompactionEvents([]);
    setCurrentStepInfo(null);
    setSavedDraft(EMPTY_SAVED_DRAFT);
    setDraftDecisionPending(false);
    // Cancel any pending save of the conversation we just cleared.
    if (persistTimerRef.current) {
      clearTimeout(persistTimerRef.current);
      persistTimerRef.current = null;
    }
    pendingPersistRef.current = null;
    persistLiveState(createEmptyLiveState());
  }, []);

  const resetLiveConversation = useCallback(() => {
    api.rotateLiveSessionId();
    clearConversationState();
  }, [clearConversationState]);

  const followResumedRun = useCallback((sessionId: string) => {
    api.setLiveSessionId(sessionId);
    clearConversationState();
    refresh();
  }, [clearConversationState, refresh]);

  // Debounce draft saves to avoid serializing the conversation on every token.
  // Flush the pending save on unmount.
  useEffect(() => {
    if (draftDecisionPending) return;
    pendingPersistRef.current = {
      version: 1,
      messages: trimTail(messages, MAX_MESSAGES),
      toolCalls: trimTail(toolCalls, MAX_TOOL_CALLS),
      thinkingBlocks: trimTail(thinkingBlocks, MAX_THINKING_BLOCKS),
      tokenUsage,
      activitySummary,
      lastTrust,
      delegateEvents: trimTail(delegateEvents, MAX_DELEGATE_EVENTS),
      compactionEvents: trimTail(compactionEvents, MAX_COMPACTION_EVENTS),
    };
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      persistTimerRef.current = null;
      if (pendingPersistRef.current) {
        persistLiveState(pendingPersistRef.current);
        pendingPersistRef.current = null;
      }
    }, 400);
  }, [messages, toolCalls, thinkingBlocks, tokenUsage, activitySummary, lastTrust, delegateEvents, compactionEvents, draftDecisionPending]);

  useEffect(() => () => {
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    if (pendingPersistRef.current) persistLiveState(pendingPersistRef.current);
  }, []);

  useEffect(() => {
    const unsubs: (() => void)[] = [];

    unsubs.push(sseOn('run_snapshot', (raw) => {
      const run = raw as RunSnapshot;
      if (!run) {
        setInterruptedRun(null);
        setFileChanges([]);
        setState('idle');
        pendingQuestionRef.current = null;
        setPendingQuestion(null);
        setPendingApproval(null);
        return;
      }
      if (run.sessionId !== api.getLiveSessionId()) return;
      setInterruptedRun(run.status === 'interrupted' ? run : null);
      beginLiveSession();
      api.setCurrentRunId(run.runId);
      const active = !['completed', 'failed', 'cancelled', 'interrupted'].includes(run.status);
      pendingTextRef.current = active ? run.text : '';
      assistantMsgIdRef.current = active && run.text ? `${run.runId}:answer` : null;
      currentStepRef.current = run.stepNumber;
      runSeqRef.current = Math.max(runSeqRef.current, 1);
      setMessages(previous => restoreRunMessages(previous, run));
      const calls = run.toolCalls.map((call, index) => ({
        ...call, id: call.callId || `${run.runId}:tool:${index}`, run: runSeqRef.current,
      }));
      setToolCalls(calls);
      setLastTrust(run.trust);
      setFileChanges(run.fileChanges ?? []);
      pendingQuestionRef.current = run.question;
      setPendingQuestion(run.question);
      setPendingApproval(run.approval ? {
        ...run.approval, receivedAt: Date.now(),
        timeoutMs: Math.max(0, run.approval.expiresAt - Date.now()),
      } : null);
      setState(run.approval ? 'waiting_approval' : run.question ? 'waiting_question' : active ? 'running' : 'idle');
      setCurrentStepInfo(active ? { stepNumber: run.stepNumber, tokens: 0 } : null);
      setActivitySummary(active ? null : computeActivitySummary(calls, run.durationMs ?? 0, run.success === true));
    }));

    // Events are broadcast to all clients. Filter by conversation, while
    // accepting untagged events from older daemons.
    const onOwnRun = <T,>(event: SseEventType, handler: (data: T) => void) =>
      sseOn(event, (raw) => {
        const d = raw as { runId?: string; sessionId?: string };
        if (!belongsToConversation(d, api.getLiveSessionId(), api.getCurrentRunId())) return;
        handler(raw as T);
      });

    unsubs.push(sseOn('agent_start', (raw) => {
      const data = raw as AgentStartData;
      // Require an identity before replacing this conversation's run state.
      const ownRun = Boolean(data.sessionId || data.runId)
        && belongsToConversation(data, api.getLiveSessionId(), api.getCurrentRunId());
      if (!ownRun) return;
      setInterruptedRun(null);
      setFileChanges(data.fileChanges ?? []);
      if (data.runId) api.setCurrentRunId(data.runId);

      beginLiveSession();
      setState('running');
      pendingTextRef.current = '';
      assistantMsgIdRef.current = null;
      setActivitySummary(null);
      setLastTrust(null);
      setCurrentStepInfo(null);
      setOrchestration(null);
      setToolCalls([]);
      currentStepRef.current = 0;
      runSeqRef.current += 1;
    }));

    unsubs.push(sseOn('suggestion_created', (raw) => {
      // Suggestions appear as cards; acting on them requires a user reply.
      const d = raw as {
        id?: string; type?: string; title?: string; description?: string;
        confidence?: number; source?: string;
      };
      const conf = typeof d.confidence === 'number' ? d.confidence : 0.5;
      const intensity: ProactiveSuggestion['intensity'] =
        conf >= 0.8 ? 'intervene' : conf >= 0.6 ? 'suggest' : 'nudge';
      const suggestionId = d.id || nextId();
      setMessages(prev => {
        // Heartbeats can repeat an open suggestion.
        if (prev.some(m => m.suggestion?.id === suggestionId)) return prev;
        return appendWithLimit(prev, {
          id: nextId(),
          role: 'system',
          content: d.title || d.description || 'RUNE has a suggestion',
          timestamp: Date.now(),
          suggestion: {
            id: suggestionId,
            headline: d.title || '',
            body: d.description || '',
            actions: [],
            confidence: conf,
            intensity,
            timestamp: Date.now(),
          },
        }, MAX_MESSAGES);
      });
    }));

    unsubs.push(onOwnRun('agent_complete', (raw) => {
      const data = raw as AgentCompleteData;
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
      pendingQuestionRef.current = null;
      setPendingQuestion(null);
      setCurrentStepInfo(null);
      if (data.usage) setTokenUsage(data.usage);

      // Show a completion toast only when the tab is in the background.
      if (typeof document !== 'undefined' && document.hidden) {
        const ok = data.success !== false;
        toast[ok ? 'success' : 'error'](ok ? 'RUNE finished the task' : 'RUNE stopped — needs a look');
      }

      setToolCalls(prev => {
        setActivitySummary(computeActivitySummary(prev, data.durationMs ?? 0, data.success !== false));
        return prev;
      });

      // Clear refs inside the updater so queued deltas still find their message.
      if (data.answer) {
        setMessages(prev => {
          const msgId = assistantMsgIdRef.current;
          pendingTextRef.current = '';
          assistantMsgIdRef.current = null;

          if (msgId) {
            const idx = prev.findIndex(m => m.id === msgId);
            if (idx !== -1) {
              const updated = [...prev];
              updated[idx] = { ...updated[idx], content: data.answer, timestamp: Date.now() };
              return updated;
            }
          }
          return appendWithLimit(prev, {
            id: nextId(),
            role: 'assistant' as const,
            content: data.answer,
            timestamp: Date.now(),
          }, MAX_MESSAGES);
        });
      } else {
        setMessages(prev => {
          pendingTextRef.current = '';
          assistantMsgIdRef.current = null;
          return prev;
        });
      }

      const trust = data.trust;
      setLastTrust(trust ?? null);
      if (trust && describeTrust(trust).showCard) {
        setMessages(prev => appendWithLimit(prev, {
          id: nextId(),
          role: 'system' as const,
          content: '',
          timestamp: Date.now(),
          trust,
        }, MAX_MESSAGES));
      }
    }));

    unsubs.push(onOwnRun('agent_error', (raw) => {
      const data = raw as AgentErrorData;
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
      pendingQuestionRef.current = null;
      setPendingQuestion(null);
      setMessages(prev => {
        pendingTextRef.current = '';
        assistantMsgIdRef.current = null;
        return appendWithLimit(prev, {
          id: nextId(),
          role: 'system',
          content: `Error: ${data.error}`,
          timestamp: Date.now(),
          level: 'error',
        }, MAX_MESSAGES);
      });
    }));

    unsubs.push(sseOn('agent_aborted', (raw) => {
      const data = raw as AgentAbortedData;
      if ((data.runId && data.runId !== api.getCurrentRunId())
          || !belongsToConversation(data, api.getLiveSessionId(), api.getCurrentRunId())) {
        setMessages(prev => upsertRunMessage(prev, abortedMessage(data, nextId(), Date.now()), true));
        return;
      }
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
      pendingQuestionRef.current = null;
      setPendingQuestion(null);
      setCurrentStepInfo(null);
      setLastTrust(data.trust ?? null);
      setToolCalls(prev => {
        setActivitySummary(computeActivitySummary(prev, 0, false));
        return prev;
      });
      setMessages(prev => {
        pendingTextRef.current = '';
        assistantMsgIdRef.current = null;
        return trimTail(upsertRunMessage(prev, abortedMessage(data, nextId(), Date.now())), MAX_MESSAGES);
      });
    }));

    unsubs.push(onOwnRun('text_delta', (raw) => {
      const data = raw as TextDeltaData;
      // `delta` appends text; legacy `text` payloads replace the buffer.
      if (data.delta) {
        pendingTextRef.current += data.delta;
      } else if (data.text !== undefined) {
        if (!data.text.trim()) return;
        pendingTextRef.current = data.text;
      } else {
        return;
      }
      if (!pendingTextRef.current.trim()) return;
      flushTextDelta();
    }));

    unsubs.push(onOwnRun('thinking', (raw) => {
      const data = raw as ThinkingData;
      setThinkingBlocks(prev => appendWithLimit(prev, {
        id: nextId(),
        text: data.text,
        timestamp: Date.now(),
      }, MAX_THINKING_BLOCKS));
    }));

    unsubs.push(onOwnRun('tool_call', (raw) => {
      const data = raw as ToolCallData;
      if (!data.toolName?.trim()) return;  // 빈 도구 이름 무시
      flushTextDelta();
      setToolCalls(prev => appendWithLimit(prev, {
        id: nextId(),
        callId: data.callId,
        toolName: data.toolName,
        args: data.args ?? {},
        timestamp: Date.now(),
        step: currentStepRef.current,
        run: runSeqRef.current,
      }, MAX_TOOL_CALLS));
    }));

    unsubs.push(onOwnRun('tool_result', (raw) => {
      const data = raw as ToolResultData;
      if (data.fileChange) {
        const change = data.fileChange;
        setFileChanges(previous => [...previous.filter(item => item.id !== change.id), change].slice(-100));
      }
      const now = Date.now();
      setToolCalls(prev => {
        // Concurrent calls can finish out of order; prefer callId over tool name.
        const actualIdx = data.callId
          ? prev.findIndex(tc => tc.callId === data.callId)
          : prev.findIndex(
              tc =>
                tc.toolName === data.toolName &&
                tc.result === undefined &&
                tc.run === runSeqRef.current,
            );
        if (actualIdx === -1) return prev;
        const original = prev[actualIdx];
        const updated = [...prev];
        updated[actualIdx] = {
          ...original,
          result: data.result,
          success: data.success,
          completedAt: now,
          durationMs: Math.max(0, now - original.timestamp),
        };
        return updated;
      });
    }));

    unsubs.push(onOwnRun('approval_request', (raw) => {
      const data = raw as ApprovalRequestData;
      setState('waiting_approval');
      setPendingApproval({
        id: data.id,
        command: data.command,
        riskLevel: data.riskLevel,
        reason: data.reason,
        timeoutMs: data.timeoutMs,
        receivedAt: Date.now(),
      });
    }));

    unsubs.push(onOwnRun('approval_closed', (raw) => {
      const { id } = raw as { id: string };
      if (pendingApprovalRef.current?.id !== id) return;
      setPendingApproval(null);
      setState(current => current === 'waiting_approval' ? 'running' : current);
    }));

    unsubs.push(onOwnRun('question', (raw) => {
      const data = raw as QuestionData;
      setState('waiting_question');
      const question = {
        id: data.id,
        question: data.question,
        callId: data.callId,
        options: data.options,
        inputMode: data.inputMode,
      };
      pendingQuestionRef.current = question;
      setPendingQuestion(question);
    }));

    unsubs.push(onOwnRun('question_closed', (raw) => {
      const { id } = raw as { id: string };
      if (pendingQuestionRef.current?.id !== id) return;
      pendingQuestionRef.current = null;
      setPendingQuestion(null);
      setState(current => current === 'waiting_question' ? 'running' : current);
    }));

    unsubs.push(onOwnRun('step_start', (raw) => {
      const data = raw as StepStartData;
      currentStepRef.current = data.stepNumber;
      setCurrentStepInfo({ stepNumber: data.stepNumber, tokens: data.tokens });
    }));

    unsubs.push(onOwnRun('orchestration_started', (raw) => {
      const data = raw as OrchestrationStartedData;
      setOrchestration({
        description: data.description,
        completed: 0,
        total: data.taskCount,
        tasks: [],
      });
    }));

    unsubs.push(onOwnRun('orchestration_task_progress', (raw) => {
      const data = raw as OrchestrationTaskProgressData;
      setOrchestration(prev => {
        const base = prev ?? { description: '', completed: 0, total: data.total, tasks: [] };
        return {
          ...base,
          completed: data.completed,
          total: data.total,
          tasks: upsertTask(base.tasks, data.taskId, {
            description: data.description,
            role: data.role,
            success: data.success,
          }),
        };
      });
    }));

    unsubs.push(onOwnRun('orchestration_task_retry', (raw) => {
      const data = raw as OrchestrationTaskRetryData;
      setOrchestration(prev => prev && {
        ...prev,
        tasks: upsertTask(prev.tasks, data.taskId, {
          retries: data.attempt,
          success: undefined,
        }),
      });
    }));

    unsubs.push(onOwnRun('orchestration_completed', (raw) => {
      const data = raw as OrchestrationCompletedData;
      setOrchestration(prev => prev && { ...prev, completed: data.completedCount });
    }));

    unsubs.push(sseOn('context_compaction', (raw) => {
      const data = raw as ContextCompactionData;
      setCompactionEvents(prev => appendWithLimit(prev, {
        id: nextId(),
        message: data.message,
        timestamp: Date.now(),
      }, MAX_COMPACTION_EVENTS));
    }));

    unsubs.push(onOwnRun('delegate_event', (raw) => {
      const data = raw as DelegateEventData;
      setDelegateEvents(prev => appendWithLimit(prev, {
        id: nextId(),
        stage: data.stage,
        message: data.message,
        timestamp: Date.now(),
      }, MAX_DELEGATE_EVENTS));
    }));

    unsubs.push(sseOn('command_result', (raw) => {
      const data = raw as CommandResultData;
      // Command results are also broadcast to every tab.
      if (data.requestSessionId && data.requestSessionId !== api.getLiveSessionId()) return;
      // /load: pin the live chat to the loaded conversation and show its turns.
      if (data.data?.action === 'load_session' && data.data.sessionId) {
        clearConversationState();
        api.setLiveSessionId(data.data.sessionId);
        const turns = data.data.turns ?? [];
        setMessages(turns.map(t => ({
          id: nextId(),
          role: t.role === 'assistant' ? 'assistant' as const : 'user' as const,
          content: t.content,
          timestamp: Date.now(),
        })));
        // Refresh the workspace picker after loading a conversation.
        if (data.data.workspace) {
          window.dispatchEvent(new CustomEvent('rune:workspace-changed'));
        }
      }
      if (data.output) {
        setMessages(prev => appendWithLimit(prev, {
          id: nextId(),
          role: 'system',
          content: data.output,
          timestamp: Date.now(),
        }, MAX_MESSAGES));
      }
    }));

    unsubs.push(onOwnRun('goal_iteration', (raw) => {
      const d = raw as GoalIterationData;
      setMessages(prev => appendWithLimit(prev, {
        id: nextId(),
        role: 'system',
        content: `[goal ${d.n}] ${d.verdict} · ${d.reason} · evidence=${d.evidence.toFixed(2)} · ${d.tokens} tokens`,
        timestamp: Date.now(),
      }, MAX_MESSAGES));
    }));

    return () => { unsubs.forEach(fn => fn()); };
  }, [sseOn, flushTextDelta, beginLiveSession, clearConversationState]);


  // Snapshot for client-side slash commands (/retry, /copy, /export, /stats).
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const tokenUsageRef = useRef<TokenUsage | null>(null);
  useEffect(() => { tokenUsageRef.current = tokenUsage; }, [tokenUsage]);

  const pushSystem = useCallback((content: string) => {
    setMessages(prev => appendWithLimit(prev, {
      id: nextId(), role: 'system', content, timestamp: Date.now(),
    }, MAX_MESSAGES));
  }, []);

  const postToServer = useCallback((text: string, apiAttachments?: { name: string; mimeType: string; data: string }[]) => {
    const sessionId = api.getLiveSessionId();
    api.sendMessage(text, apiAttachments).catch(err => {
      if (sessionId !== api.getLiveSessionId()) return;
      setMessages(prev => appendWithLimit(prev, {
        id: nextId(),
        role: 'system',
        content: `Failed to send: ${err instanceof Error ? err.message : String(err)}`,
        timestamp: Date.now(),
        level: 'error',
      }, MAX_MESSAGES));
    });
  }, []);

  // Re-run the most recent user turn (the Regenerate action, same as /retry).
  const regenerate = useCallback(() => {
    const lastUser = [...messagesRef.current].reverse().find(m => m.role === 'user');
    if (!lastUser) return;
    beginLiveSession();
    setMessages(prev => appendWithLimit(prev, {
      id: nextId(), role: 'user', content: lastUser.content, timestamp: Date.now(),
    }, MAX_MESSAGES));
    postToServer(lastUser.content);
  }, [beginLiveSession, postToServer]);

  // Client-side slash commands; everything else goes to the server and
  // answers over the command_result SSE event.
  const handleClientCommand = useCallback((text: string): boolean => {
    const [cmd, ...rest] = text.trim().split(/\s+/);
    const args = rest.join(' ');
    const msgs = messagesRef.current;

    switch (cmd.toLowerCase()) {
      case '/clear':
      case '/cls':
        resetLiveConversation();
        return true;

      case '/retry':
      case '/r': {
        const lastUser = [...msgs].reverse().find(m => m.role === 'user');
        if (!lastUser) { pushSystem('Nothing to retry.'); return true; }
        setMessages(prev => appendWithLimit(prev, {
          id: nextId(), role: 'user', content: lastUser.content, timestamp: Date.now(),
        }, MAX_MESSAGES));
        postToServer(lastUser.content);
        return true;
      }

      case '/copy':
      case '/cp': {
        const lastAssistant = [...msgs].reverse().find(m => m.role === 'assistant');
        if (!lastAssistant) { pushSystem('No assistant message to copy.'); return true; }
        navigator.clipboard?.writeText(lastAssistant.content)
          .then(() => pushSystem('Response copied to clipboard.'))
          .catch(() => pushSystem('Clipboard unavailable.'));
        return true;
      }

      case '/export': {
        const fmt = (args || 'markdown').toLowerCase();
        if (!['markdown', 'json', 'md'].includes(fmt)) {
          pushSystem(`Unknown format: ${fmt}. Use: markdown, json`);
          return true;
        }
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        let blob: Blob; let name: string;
        if (fmt === 'json') {
          blob = new Blob(
            [JSON.stringify(msgs.map(m => ({ role: m.role, content: m.content, timestamp: m.timestamp })), null, 2)],
            { type: 'application/json' },
          );
          name = `rune-${stamp}.json`;
        } else {
          const md = ['# RUNE Conversation', ''];
          for (const m of msgs) {
            md.push(`## ${m.role}`, '', m.content, '');
          }
          blob = new Blob([md.join('\n')], { type: 'text/markdown' });
          name = `rune-${stamp}.md`;
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = name; a.click();
        URL.revokeObjectURL(url);
        pushSystem(`Exported ${msgs.length} messages to ${name}.`);
        return true;
      }

      case '/stats': {
        const users = msgs.filter(m => m.role === 'user').length;
        const assistants = msgs.filter(m => m.role === 'assistant').length;
        const t = tokenUsageRef.current;
        const tokens = t ? `${t.input} in / ${t.output} out` : 'n/a';
        const first = msgs[0]?.timestamp;
        const mins = first ? Math.round((Date.now() - first) / 60000) : 0;
        pushSystem(`Session stats: ${users} user / ${assistants} assistant messages · tokens ${tokens} · ${mins} min`);
        return true;
      }

      case '/cost': {
        const t = tokenUsageRef.current;
        pushSystem(t
          ? `Token usage: ${t.input} input / ${t.output} output (total ${t.total}). Local (ollama) runs are free; cloud cost depends on the active model.`
          : 'No token usage recorded yet.');
        return true;
      }

      case '/style':
      case '/compact':
      case '/normal':
      case '/verbose':
      case '/theme':
        pushSystem('Display options live in the app Settings sidebar; /style·/theme apply to the terminal UI.');
        return true;

      default:
        return false;
    }
  }, [pushSystem, postToServer, resetLiveConversation]);

  const sendMessage = useCallback((text: string, pendingAttachments?: PendingAttachment[]) => {
    if (text.trim().startsWith('/') && !pendingAttachments?.length && handleClientCommand(text)) {
      return;
    }
    beginLiveSession();
    setMessages(prev => appendWithLimit(prev, {
      id: nextId(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      attachments: pendingAttachments?.map(a => ({
        name: a.name,
        mimeType: a.mimeType,
        dataUrl: a.dataUrl,
      })),
    }, MAX_MESSAGES));
    const apiAttachments = pendingAttachments?.map(a => ({
      name: a.name,
      mimeType: a.mimeType,
      data: a.dataUrl.replace(/^data:[^;]+;base64,/, ''),
    }));
    postToServer(text, apiAttachments);
  }, [beginLiveSession, handleClientCommand, postToServer]);

  const pushSystemError = useCallback((prefix: string, err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    setMessages(prev => appendWithLimit(prev, {
      id: nextId(),
      role: 'system',
      content: `${prefix}: ${message}`,
      timestamp: Date.now(),
      level: 'error',
    }, MAX_MESSAGES));
  }, []);

  const abort = useCallback(() => {
    // Release the input immediately; agent_aborted confirms the server's stop.
    flushTextDelta();
    setState('idle');
    pendingQuestionRef.current = null;
    setPendingQuestion(null);
    api.sendAbort().catch(err => pushSystemError('Failed to stop the run', err));
  }, [pushSystemError, flushTextDelta]);

  const respondApproval = useCallback(async (decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) => {
    const approval = pendingApprovalRef.current;
    if (!approval) return;
    try {
      await api.sendApproval(approval.id, decision, userGuidance);
      if (pendingApprovalRef.current?.id === approval.id) {
        setPendingApproval(null);
        setState(current => current === 'waiting_approval' ? 'running' : current);
      }
    } catch (err) {
      if (pendingApprovalRef.current?.id === approval.id) {
        pushSystemError('Approval response failed', err);
        throw err;
      }
    }
  }, [setPendingApproval, pushSystemError]);

  const respondQuestion = useCallback(async (answer: string, selectedIndex?: number) => {
    const question = pendingQuestionRef.current;
    if (!question) throw new Error('This question is no longer waiting for an answer.');
    try {
      await api.sendQuestion(question.id, answer, selectedIndex);
      if (pendingQuestionRef.current?.id === question.id) {
        pendingQuestionRef.current = null;
        setPendingQuestion(null);
        setState(current => current === 'waiting_question' ? 'running' : current);
      }
    } catch (error) {
      if (pendingQuestionRef.current?.id === question.id) {
        pushSystemError('Question response failed', error);
        throw error;
      }
    }
  }, [pushSystemError]);

  return {
    connected,
    state,
    messages,
    toolCalls,
    thinkingBlocks,
    tokenUsage,
    pendingApproval,
    pendingQuestion,
    activitySummary,
    lastTrust,
    fileChanges,
    interruptedRun,
    followResumedRun,
    delegateEvents,
    compactionEvents,
    currentStepInfo,
    orchestration,
    savedDraft,
    restoreSavedDraft,
    discardSavedDraft,
    resetLiveConversation,
    sendMessage,
    regenerate,
    abort,
    respondApproval,
    respondQuestion,
  };
}
