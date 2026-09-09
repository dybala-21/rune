import type { ChatMessage, FileChange, PendingApproval, PendingQuestion, ToolCall, TrustInfo } from '../types';
import { describeTrust } from './trust';
import { abortedMessage } from './runEvents';

export interface RunSnapshot {
  runId: string;
  parentRunId?: string;
  recoveryVersion?: number;
  fileChanges?: FileChange[];
  sessionId: string;
  seq: number;
  goal: string;
  status: 'queued' | 'running' | 'waiting_input' | 'waiting_approval' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  startedAt: number;
  updatedAt?: number;
  text: string;
  textStartedAt?: number;
  history?: ChatMessage[];
  answer?: string;
  error?: string;
  durationMs?: number;
  success?: boolean;
  stepNumber: number;
  toolCalls: Array<Omit<ToolCall, 'id'> & { callId?: string }>;
  question: (PendingQuestion & { expiresAt: number }) | null;
  approval: (PendingApproval & { expiresAt: number }) | null;
  trust: TrustInfo | null;
  interactions?: Array<{
    id: string;
    kind: 'question' | 'approval';
    request: { question?: string; command?: string; expiresAt?: number };
    status: string;
    response: { answer?: string; decision?: string } | null;
  }>;
}

export function restoreRunMessages(messages: ChatMessage[], run: RunSnapshot): ChatMessage[] {
  if (!messages.length && run.history) messages = run.history;
  const userId = `${run.runId}:user`;
  let index = messages.findIndex(message => message.id === userId);
  if (index < 0) {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user' && messages[i].content === run.goal) { index = i; break; }
    }
  }
  const restored = index < 0 ? [...messages] : messages.slice(0, index);
  restored.push({ id: userId, role: 'user', content: run.goal, timestamp: run.startedAt,
    ...(index >= 0 && messages[index].attachments ? { attachments: messages[index].attachments } : {}) });
  const text = run.answer ?? run.text;
  if (text) restored.push({ id: `${run.runId}:answer`, role: 'assistant', content: text,
    timestamp: run.answer === undefined ? run.textStartedAt ?? run.startedAt : run.updatedAt ?? run.startedAt });
  if (run.status === 'interrupted') {
    for (const interaction of run.interactions ?? []) {
      const prompt = interaction.kind === 'question' ? interaction.request.question : interaction.request.command;
      const response = interaction.response?.answer ?? interaction.response?.decision;
      restored.push({ id: `${run.runId}:${interaction.id}:receipt`, role: 'system',
        content: `${interaction.kind === 'question' ? 'Question' : 'Approval'}: ${prompt ?? ''}\n${response !== undefined ? `Recorded response: ${response}` : 'Closed when the run stopped.'}`,
        timestamp: run.updatedAt ?? run.startedAt });
    }
  }
  if (run.status === 'cancelled') {
    restored.push(abortedMessage({ runId: run.runId, trust: run.trust ?? undefined }, '', run.updatedAt ?? run.startedAt));
  } else if (run.trust && describeTrust(run.trust).showCard) {
    restored.push({ id: `${run.runId}:trust`, role: 'system', content: '', trust: run.trust, timestamp: run.updatedAt ?? run.startedAt });
  }
  if (run.error) restored.push({ id: `${run.runId}:error`, role: 'system', content: run.error, level: 'error', timestamp: run.updatedAt ?? run.startedAt });
  return restored;
}
