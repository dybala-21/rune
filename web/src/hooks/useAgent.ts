import { useCallback, useEffect, useRef, useState } from 'react';
import { useSSE } from './useSSE';
import * as api from '../api';
import { computeActivitySummary } from '../utils/tooling';
import type {
  AgentState,
  ChatMessage,
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
} from '../types';

let idCounter = 0;
function nextId(): string {
  return `msg-${++idCounter}-${Date.now()}`;
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

function persistLiveState(state: PersistedLiveState): void {
  if (typeof window === 'undefined') return;
  try {
    const payload: PersistedLiveEnvelope = {
      version: 1,
      savedAt: Date.now(),
      state,
    };
    window.localStorage.setItem(LIVE_STATE_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // ignore quota / storage errors
  }
}

export function useAgent() {
  const { connected, addEventListener: sseOn } = useSSE();
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
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<PendingQuestion | null>(null);
  const [activitySummary, setActivitySummary] = useState<ActivitySummary | null>(initialLiveState.activitySummary);
  const [delegateEvents, setDelegateEvents] = useState<DelegateItem[]>(initialLiveState.delegateEvents);
  const [compactionEvents, setCompactionEvents] = useState<CompactionItem[]>(initialLiveState.compactionEvents);
  const [currentStepInfo, setCurrentStepInfo] = useState<StepInfo | null>(null);
  const [savedDraft, setSavedDraft] = useState<SavedLiveDraftSummary>(
    summarizeSavedDraft(initialState.state, initialState.savedAt),
  );
  const [draftDecisionPending, setDraftDecisionPending] = useState<boolean>(hasSavedDraft);

  // 현재 step의 텍스트를 보관 (교체 방식 — step.text는 delta가 아니라 해당 step 전체 텍스트)
  const pendingTextRef = useRef('');
  // 현재 run의 assistant 메시지 ID (하나의 run에 하나의 assistant 메시지만 유지)
  const assistantMsgIdRef = useRef<string | null>(null);

  // Flush text into a message.
  // NOTE: assistantMsgIdRef를 setMessages updater 안에서 읽어야 함.
  // React batching으로 여러 setMessages가 큐잉되면, 바깥에서 캡처한 ref 값은
  // 이전 updater가 설정한 값을 반영하지 못해 중복 메시지가 생김.
  const flushTextDelta = useCallback(() => {
    const text = pendingTextRef.current;
    if (!text) return;

    setMessages(prev => {
      const msgId = assistantMsgIdRef.current; // updater 안에서 읽기
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

  // Read draft state via refs so beginLiveSession stays a stable dep of the SSE
  // effect — otherwise its identity churns and the effect re-subscribes mid-run,
  // dropping events.
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
    setMessages(trimTail(draft.messages, MAX_MESSAGES));
    setToolCalls(trimTail(draft.toolCalls, MAX_TOOL_CALLS));
    setThinkingBlocks(trimTail(draft.thinkingBlocks, MAX_THINKING_BLOCKS));
    setTokenUsage(draft.tokenUsage ?? null);
    setActivitySummary(draft.activitySummary ?? null);
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

  const resetLiveConversation = useCallback(() => {
    // New chat = new server-side conversation.
    api.rotateLiveSessionId();
    pendingTextRef.current = '';
    assistantMsgIdRef.current = null;
    savedDraftStateRef.current = null;
    setState('idle');
    setMessages([]);
    setToolCalls([]);
    setThinkingBlocks([]);
    setTokenUsage(null);
    setPendingApproval(null);
    setPendingQuestion(null);
    setActivitySummary(null);
    setDelegateEvents([]);
    setCompactionEvents([]);
    setCurrentStepInfo(null);
    setSavedDraft(EMPTY_SAVED_DRAFT);
    setDraftDecisionPending(false);
    persistLiveState(createEmptyLiveState());
  }, []);

  useEffect(() => {
    if (draftDecisionPending) return;
    persistLiveState({
      version: 1,
      messages: trimTail(messages, MAX_MESSAGES),
      toolCalls: trimTail(toolCalls, MAX_TOOL_CALLS),
      thinkingBlocks: trimTail(thinkingBlocks, MAX_THINKING_BLOCKS),
      tokenUsage,
      activitySummary,
      delegateEvents: trimTail(delegateEvents, MAX_DELEGATE_EVENTS),
      compactionEvents: trimTail(compactionEvents, MAX_COMPACTION_EVENTS),
    });
  }, [messages, toolCalls, thinkingBlocks, tokenUsage, activitySummary, delegateEvents, compactionEvents, draftDecisionPending]);

  // sseOn (addEventListener)은 useCallback([], [])로 항상 동일 참조.
  // flushTextDelta도 useCallback([], [])로 안정.
  // → effect는 마운트 시 1회만 실행되고, 언마운트 시 정리됨.
  useEffect(() => {
    const unsubs: (() => void)[] = [];

    unsubs.push(sseOn('agent_start', (raw) => {
      beginLiveSession();
      const data = raw as AgentStartData;
      setState('running');
      pendingTextRef.current = '';
      assistantMsgIdRef.current = null;
      setActivitySummary(null);
      setCurrentStepInfo(null);
      setMessages(prev => appendWithLimit(prev, {
        id: nextId(),
        role: 'system',
        content: `Goal: ${data.goal}`,
        timestamp: Date.now(),
      }, MAX_MESSAGES));
    }));

    unsubs.push(sseOn('agent_complete', (raw) => {
      const data = raw as AgentCompleteData;
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
      setPendingQuestion(null);
      setCurrentStepInfo(null);
      if (data.usage) setTokenUsage(data.usage);

      // Compute activity summary from tool calls
      setToolCalls(prev => {
        setActivitySummary(computeActivitySummary(prev, data.durationMs ?? 0, data.success !== false));
        return prev;
      });

      // 최종 answer로 assistant 메시지 교체/생성 (권위적 최종 답변)
      // ref 정리를 setMessages updater 안에서 수행해야 함.
      // React 18+ batching으로 updater들은 핸들러 완료 후 순서대로 실행되므로,
      // 바깥에서 동기적으로 ref를 null로 리셋하면 flushTextDelta updater가
      // null을 읽어 중복 메시지를 생성하는 버그 발생.
      if (data.answer) {
        setMessages(prev => {
          const msgId = assistantMsgIdRef.current; // updater 안에서 읽기
          // 다음 run을 위해 리셋 (updater 안에서 해야 올바른 순서 보장)
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
        // answer가 없어도 ref 정리는 updater 안에서 수행 (batching 순서 보장)
        setMessages(prev => {
          pendingTextRef.current = '';
          assistantMsgIdRef.current = null;
          return prev;
        });
      }
    }));

    unsubs.push(sseOn('agent_error', (raw) => {
      const data = raw as AgentErrorData;
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
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

    unsubs.push(sseOn('agent_aborted', () => {
      flushTextDelta();
      setState('idle');
      setPendingApproval(null);
      setPendingQuestion(null);
      setMessages(prev => {
        pendingTextRef.current = '';
        assistantMsgIdRef.current = null;
        return appendWithLimit(prev, {
          id: nextId(),
          role: 'system',
          content: 'Execution aborted.',
          timestamp: Date.now(),
        }, MAX_MESSAGES);
      });
    }));

    unsubs.push(sseOn('text_delta', (raw) => {
      const data = raw as TextDeltaData;
      // 교체 방식: onStepFinish의 step.text는 해당 step 전체 텍스트 (delta 아님).
      // multi-pass에서 +=하면 이전 pass 텍스트가 누적되어 깨짐.
      // 공백만 있는 텍스트는 빈 메시지 버블을 만들므로 무시
      if (!data.text?.trim()) return;
      pendingTextRef.current = data.text;
      flushTextDelta();
    }));

    unsubs.push(sseOn('thinking', (raw) => {
      const data = raw as ThinkingData;
      setThinkingBlocks(prev => appendWithLimit(prev, {
        id: nextId(),
        text: data.text,
        timestamp: Date.now(),
      }, MAX_THINKING_BLOCKS));
    }));

    unsubs.push(sseOn('tool_call', (raw) => {
      const data = raw as ToolCallData;
      if (!data.toolName?.trim()) return;  // 빈 도구 이름 무시
      flushTextDelta();
      setToolCalls(prev => appendWithLimit(prev, {
        id: nextId(),
        toolName: data.toolName,
        args: data.args ?? {},
        timestamp: Date.now(),
      }, MAX_TOOL_CALLS));
    }));

    unsubs.push(sseOn('tool_result', (raw) => {
      const data = raw as ToolResultData;
      const now = Date.now();
      setToolCalls(prev => {
        // FIFO: results arrive in call order, so pair with the oldest unresolved call.
        const actualIdx = prev.findIndex(
          tc => tc.toolName === data.toolName && tc.result === undefined
        );
        if (actualIdx === -1) return prev;
        const original = prev[actualIdx];
        const updated = [...prev];
        updated[actualIdx] = {
          ...original,
          result: data.result,
          success: data.success,
          completedAt: now,
          durationMs: now - original.timestamp,
        };
        return updated;
      });
    }));

    unsubs.push(sseOn('approval_request', (raw) => {
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

    unsubs.push(sseOn('question', (raw) => {
      const data = raw as QuestionData;
      setState('waiting_question');
      setPendingQuestion({
        id: data.id,
        question: data.question,
        options: data.options,
        inputMode: data.inputMode,
      });
    }));

    unsubs.push(sseOn('step_start', (raw) => {
      const data = raw as StepStartData;
      setCurrentStepInfo({ stepNumber: data.stepNumber, tokens: data.tokens });
    }));

    unsubs.push(sseOn('context_compaction', (raw) => {
      const data = raw as ContextCompactionData;
      setCompactionEvents(prev => appendWithLimit(prev, {
        id: nextId(),
        message: data.message,
        timestamp: Date.now(),
      }, MAX_COMPACTION_EVENTS));
    }));

    unsubs.push(sseOn('delegate_event', (raw) => {
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
      // /load: pin the live chat to the loaded conversation and show its turns.
      if (data.data?.action === 'load_session' && data.data.sessionId) {
        api.setLiveSessionId(data.data.sessionId);
        const turns = data.data.turns ?? [];
        setMessages(turns.map(t => ({
          id: nextId(),
          role: t.role === 'assistant' ? 'assistant' as const : 'user' as const,
          content: t.content,
          timestamp: Date.now(),
        })));
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

    unsubs.push(sseOn('goal_iteration', (raw) => {
      const d = raw as GoalIterationData;
      setMessages(prev => appendWithLimit(prev, {
        id: nextId(),
        role: 'system',
        content: `[goal ${d.n}] ${d.verdict} · ${d.reason} · evidence=${d.evidence.toFixed(2)} · ${d.tokens} tokens`,
        timestamp: Date.now(),
      }, MAX_MESSAGES));
    }));

    return () => { unsubs.forEach(fn => fn()); };
  }, [sseOn, flushTextDelta, beginLiveSession]);

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
    api.sendMessage(text, apiAttachments).catch(err => {
      setMessages(prev => appendWithLimit(prev, {
        id: nextId(),
        role: 'system',
        content: `Failed to send: ${err instanceof Error ? err.message : String(err)}`,
        timestamp: Date.now(),
        level: 'error',
      }, MAX_MESSAGES));
    });
  }, []);

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
    const displayContent = pendingAttachments && pendingAttachments.length > 0
      ? `${text}\n[${pendingAttachments.map(a => a.name).join(', ')}]`
      : text;
    setMessages(prev => appendWithLimit(prev, {
      id: nextId(),
      role: 'user',
      content: displayContent,
      timestamp: Date.now(),
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
    api.sendAbort().catch(err => pushSystemError('Failed to stop the run', err));
  }, [pushSystemError]);

  const respondApproval = useCallback((decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) => {
    if (!pendingApproval) return;
    api.sendApproval(pendingApproval.id, decision, userGuidance)
      .then(() => {
        setPendingApproval(null);
        setState('running');
      })
      .catch(err => {
        setState('waiting_approval');
        pushSystemError('Approval response failed', err);
      });
  }, [pendingApproval, pushSystemError]);

  const respondQuestion = useCallback((answer: string, selectedIndex?: number) => {
    if (!pendingQuestion) return;
    api.sendQuestion(pendingQuestion.id, answer, selectedIndex)
      .then(() => {
        setPendingQuestion(null);
        setState('running');
      })
      .catch(err => {
        setState('waiting_question');
        pushSystemError('Question response failed', err);
      });
  }, [pendingQuestion, pushSystemError]);

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
    delegateEvents,
    compactionEvents,
    currentStepInfo,
    savedDraft,
    restoreSavedDraft,
    discardSavedDraft,
    resetLiveConversation,
    sendMessage,
    abort,
    respondApproval,
    respondQuestion,
  };
}
