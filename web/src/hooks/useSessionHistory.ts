import { useCallback, useState } from 'react';
import { fetchSessionEvents, type EventLogEntry } from '../api';
import { computeActivitySummary } from '../utils/tooling';
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
 * 저장된 이벤트를 재생하여 세션 상태를 복원한다.
 */
function hydrateEvents(events: EventLogEntry[]): SessionHistoryState {
  const messages: ChatMessage[] = [];
  const toolCalls: ToolCall[] = [];
  const thinkingBlocks: ThinkingBlock[] = [];
  const delegateEvents: DelegateItem[] = [];
  const compactionEvents: CompactionItem[] = [];
  let currentAssistantId: string | null = null;
  let completionSuccess = true;
  let completionDurationMs = 0;

  for (const entry of events) {
    const ts = new Date(entry.timestamp).getTime();
    const data = entry.data as Record<string, unknown>;

    switch (entry.event) {
      case 'agent_start': {
        messages.push({
          id: nextId(),
          role: 'system',
          content: `Goal: ${data.goal ?? ''}`,
          timestamp: ts,
        });
        currentAssistantId = null;
        break;
      }

      case 'text_delta': {
        const text = String(data.text ?? '');
        if (!text.trim()) break;
        if (currentAssistantId) {
          const idx = messages.findIndex(m => m.id === currentAssistantId);
          if (idx !== -1) {
            messages[idx] = { ...messages[idx], content: text };
          }
        } else {
          currentAssistantId = nextId();
          messages.push({ id: currentAssistantId, role: 'assistant', content: text, timestamp: ts });
        }
        break;
      }

      case 'tool_call': {
        const toolName = String(data.toolName ?? '');
        if (!toolName.trim()) break;
        toolCalls.push({
          id: nextId(),
          toolName,
          args: (data.args as Record<string, unknown>) ?? {},
          timestamp: ts,
        });
        break;
      }

      case 'tool_result': {
        const toolName = String(data.toolName ?? '');
        const now = ts;
        const idx = [...toolCalls].reverse().findIndex(
          tc => tc.toolName === toolName && tc.result === undefined,
        );
        if (idx !== -1) {
          const actualIdx = toolCalls.length - 1 - idx;
          toolCalls[actualIdx] = {
            ...toolCalls[actualIdx],
            result: String(data.result ?? ''),
            success: data.success !== false,
            completedAt: now,
            durationMs: now - toolCalls[actualIdx].timestamp,
          };
        }
        break;
      }

      case 'thinking': {
        thinkingBlocks.push({
          id: nextId(),
          text: String(data.text ?? ''),
          timestamp: ts,
        });
        break;
      }

      case 'agent_complete': {
        const answer = String(data.answer ?? '');
        completionSuccess = data.success !== false;
        completionDurationMs =
          typeof data.durationMs === 'number' && Number.isFinite(data.durationMs) ? data.durationMs : 0;
        if (answer) {
          if (currentAssistantId) {
            const idx = messages.findIndex(m => m.id === currentAssistantId);
            if (idx !== -1) {
              messages[idx] = { ...messages[idx], content: answer, timestamp: ts };
            }
          } else {
            currentAssistantId = nextId();
            messages.push({ id: currentAssistantId, role: 'assistant', content: answer, timestamp: ts });
          }
        }
        currentAssistantId = null;
        break;
      }

      case 'agent_error': {
        completionSuccess = false;
        messages.push({
          id: nextId(),
          role: 'system',
          content: `Error: ${data.error ?? 'Unknown error'}`,
          timestamp: ts,
        });
        break;
      }

      case 'agent_aborted': {
        completionSuccess = false;
        messages.push({
          id: nextId(),
          role: 'system',
          content: 'Execution aborted.',
          timestamp: ts,
        });
        break;
      }

      case 'delegate_event': {
        delegateEvents.push({
          id: nextId(),
          stage: String(data.stage ?? ''),
          message: String(data.message ?? ''),
          timestamp: ts,
        });
        break;
      }

      case 'context_compaction': {
        compactionEvents.push({
          id: nextId(),
          message: String(data.message ?? ''),
          timestamp: ts,
        });
        break;
      }
    }
  }

  const activitySummary: ActivitySummary = computeActivitySummary(
    toolCalls,
    completionDurationMs,
    completionSuccess,
  );

  return {
    messages,
    toolCalls,
    thinkingBlocks,
    activitySummary: toolCalls.length > 0 || events.length > 0 ? activitySummary : null,
    delegateEvents,
    compactionEvents,
  };
}

export function useSessionHistory() {
  const [historyState, setHistoryState] = useState<SessionHistoryState | null>(null);
  const [viewingSessionId, setViewingSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadSession = useCallback(async (sessionId: string | null) => {
    if (!sessionId) {
      setViewingSessionId(null);
      setHistoryState(null);
      return;
    }

    setLoading(true);
    setViewingSessionId(sessionId);
    try {
      const result = await fetchSessionEvents(sessionId);
      const state = hydrateEvents(result.events);
      setHistoryState(state);
    } catch {
      setHistoryState({
        messages: [{ id: 'err', role: 'system', content: 'Failed to load session history.', timestamp: Date.now() }],
        toolCalls: [],
        thinkingBlocks: [],
        activitySummary: null,
        delegateEvents: [],
        compactionEvents: [],
      });
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    viewingSessionId,
    historyState,
    loading,
    loadSession,
  };
}
