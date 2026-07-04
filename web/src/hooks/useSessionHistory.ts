import { useCallback, useRef, useState } from 'react';
import { fetchSessionTurns, type SessionTurn } from '../api';
import type {
  ChatMessage,
  ToolCall,
  ThinkingBlock,
  ActivitySummary,
  DelegateItem,
  CompactionItem,
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
}

/**
 * 대화 턴(canonical conversation store)을 표시용 상태로 변환한다.
 * 툴콜/씽킹 로그는 대화 저장소에 없으므로 히스토리 뷰는 메시지만 보여준다.
 */
function hydrateTurns(turns: SessionTurn[]): SessionHistoryState {
  const messages: ChatMessage[] = turns.map(t => ({
    id: nextId(),
    role: t.role === 'assistant' ? 'assistant' as const : 'user' as const,
    content: t.content,
    timestamp: new Date(t.timestamp).getTime() || Date.now(),
  }));
  return {
    messages,
    toolCalls: [],
    thinkingBlocks: [],
    activitySummary: null,
    delegateEvents: [],
    compactionEvents: [],
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
      setHistoryState(hydrateTurns(result.turns));
    } catch {
      if (reqId !== reqRef.current) return;
      setHistoryState({
        messages: [{ id: 'err', role: 'system', content: 'Failed to load session history.', timestamp: Date.now(), level: 'error' }],
        toolCalls: [],
        thinkingBlocks: [],
        activitySummary: null,
        delegateEvents: [],
        compactionEvents: [],
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
