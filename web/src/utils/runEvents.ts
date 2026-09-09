import type { AgentAbortedData, ChatMessage } from '../types';

export function belongsToConversation(
  event: { sessionId?: string | null; runId?: string }, sessionId: string, runId: string,
): boolean {
  if (event.sessionId) return event.sessionId === sessionId;
  if (event.runId) return event.runId === runId;
  return true;
}

export function abortedMessage(data: AgentAbortedData, fallbackId: string, timestamp: number): ChatMessage {
  return {
    id: data.runId ? `aborted-${data.runId}` : fallbackId,
    role: 'system',
    content: data.trust ? '' : 'Execution aborted.',
    timestamp,
    trust: data.trust,
  };
}

/** The final stop snapshot updates the first notification for this run. */
export function upsertRunMessage(messages: ChatMessage[], message: ChatMessage, existingOnly = false): ChatMessage[] {
  const index = messages.findIndex(item => item.id === message.id);
  if (index < 0) return existingOnly ? messages : [...messages, message];
  const next = [...messages];
  next[index] = message;
  return next;
}
