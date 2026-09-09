import { useCallback, useRef, useState } from 'react';
import { fetchSessionTurns, type SessionTurn } from '../api';
import { restoreRunMessages, type RunSnapshot } from '../utils/runSnapshot';
import { computeActivitySummary } from '../utils/tooling';
import type {
  ChatMessage,
  ToolCall,
  ThinkingBlock,
  ActivitySummary,
  DelegateItem,
  CompactionItem,
  TrustInfo,
} from '../types';

let idCounter = 10000;
function nextId(): string {
  return `hist-${++idCounter}-${Date.now()}`;
}

export interface SessionHistoryState {
  messages: ChatMessage[];
  toolCalls: ToolCall[];
  thinkingBlocks: ThinkingBlock[];
  activitySummary: ActivitySummary | null;
  delegateEvents: DelegateItem[];
  compactionEvents: CompactionItem[];
  trust: TrustInfo | null;
  run: RunSnapshot | null;
}

function hydrateTurns(turns: SessionTurn[], run?: RunSnapshot | null): SessionHistoryState {
  const messages: ChatMessage[] = turns.map(t => ({
    id: nextId(),
    role: t.role === 'assistant' ? 'assistant' as const : 'user' as const,
    content: t.content,
    timestamp: new Date(t.timestamp).getTime() || Date.now(),
  }));
  const toolCalls = run ? run.toolCalls.map((call, index) => ({ ...call, id: call.callId || `${run.runId}:tool:${index}` })) : [];
  return {
    messages: run ? restoreRunMessages(messages, run) : messages,
    toolCalls,
    thinkingBlocks: [],
    activitySummary: run ? computeActivitySummary(toolCalls, run.durationMs ?? 0, run.success === true) : null,
    delegateEvents: [],
    compactionEvents: [],
    trust: run?.trust ?? null,
    run: run ?? null,
  };
}

export function useSessionHistory() {
  const [historyState, setHistoryState] = useState<SessionHistoryState | null>(null);
  const [viewingSessionId, setViewingSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Guards against a slow load for session A overwriting a later-selected B.
  const reqRef = useRef(0);

  const loadSession = useCallback(async (sessionId: string | null) => {
    const reqId = ++reqRef.current;
    if (!sessionId) {
      setViewingSessionId(null);
      setHistoryState(null);
      return;
    }

    setLoading(true);
    setViewingSessionId(sessionId);
    try {
      const result = await fetchSessionTurns(sessionId);
      if (reqId !== reqRef.current) return; // superseded by a newer selection
      setHistoryState(hydrateTurns(result.turns, result.run));
    } catch {
      if (reqId !== reqRef.current) return;
      setHistoryState({
        messages: [{ id: 'err', role: 'system', content: 'Failed to load session history.', timestamp: Date.now(), level: 'error' }],
        toolCalls: [],
        thinkingBlocks: [],
        activitySummary: null,
        delegateEvents: [],
        compactionEvents: [],
        trust: null,
        run: null,
      });
    } finally {
      if (reqId === reqRef.current) setLoading(false);
    }
  }, []);

  return {
    viewingSessionId,
    historyState,
    loading,
    loadSession,
  };
}
