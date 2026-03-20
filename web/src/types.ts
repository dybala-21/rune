/** SSE 이벤트 타입 (서버에서 수신) */

export interface TokenUsage {
  total: number;
  input: number;
  output: number;
  cacheRead?: number;
  cacheCreation?: number;
}

export type SseEventType =
  | 'connected'
  | 'agent_start'
  | 'agent_complete'
  | 'agent_error'
  | 'agent_aborted'
  | 'step_start'
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'text_delta'
  | 'approval_request'
  | 'question'
  | 'context_compaction'
  | 'delegate_event'
  | 'suggestion_created'
  | 'proactive_execution_started'
  | 'proactive_execution_completed'
  | 'autonomy_level_changed';

export interface ConnectedData { clientId: string }
export interface AgentStartData { goal: string }
export interface AgentCompleteData { success: boolean; answer: string; durationMs: number; usage?: TokenUsage }
export interface AgentErrorData { error: string }
export interface StepStartData { stepNumber: number; tokens: number }
export interface ThinkingData { text: string }
export interface ToolCallData { toolName: string; args: Record<string, unknown> }
export interface ToolResultData { toolName: string; result: string; success: boolean }
export interface TextDeltaData { text: string }
export interface ApprovalRequestData { id: string; command: string; riskLevel: string; reason?: string; timeoutMs: number }
export interface QuestionData {
  id: string;
  question: string;
  options?: Array<{ label: string; description?: string }>;
  inputMode?: 'text' | 'secret';
}
export interface ContextCompactionData { message: string }
export interface DelegateEventData { stage: string; message: string }
export interface SuggestionCreatedData {
  id: string;
  type: string;
  description: string;
  priority: string;
  confidence: number;
  action?: { command?: string; autoExecutable?: boolean };
}
export interface ProactiveExecutionStartedData { suggestionId: string; goal: string }
export interface ProactiveExecutionCompletedData {
  suggestionId: string;
  success: boolean;
  executionTimeMs: number;
  error?: string;
}
export interface AutonomyLevelChangedData {
  domain: string;
  patternKey: string;
  previousLevel: number;
  newLevel: number;
  direction: 'promoted' | 'demoted';
  reason?: string;
}

/** 채팅 메시지 (UI 표시용) */
export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
}

/** 프로액티브 제안 (RUNE이 먼저 말을 걸 때) */
export interface ProactiveSuggestion {
  id: string;
  headline: string;
  body: string;
  actions: string[];
  confidence: number;
  intensity: 'nudge' | 'suggest' | 'intervene';
  timestamp: number;
}

/** 도구 호출 (UI 표시용) */
export interface ToolCall {
  id: string;
  toolName: string;
  args: Record<string, unknown>;
  result?: string;
  success?: boolean;
  timestamp: number;
  completedAt?: number;
  durationMs?: number;
}

/** thinking 블록 (UI 표시용) */
export interface ThinkingBlock {
  id: string;
  text: string;
  timestamp: number;
}

export type AgentState = 'idle' | 'running' | 'waiting_approval' | 'waiting_question';

/** 승인 요청 정보 */
export interface PendingApproval {
  id: string;
  command: string;
  riskLevel: string;
  reason?: string;
  suggestions?: string[];
  timeoutMs: number;
  receivedAt: number;
}

/** 활동 요약 (완료 후 표시) */
export interface ActivitySummary {
  success: boolean;
  totalToolCalls: number;
  filesRead: number;
  filesWritten: number;
  bashExecutions: number;
  webSearches: number;
  browserActions: number;
  totalDurationMs: number;
}

/** delegate 이벤트 (타임라인 표시용) */
export interface DelegateItem {
  id: string;
  stage: string;
  message: string;
  timestamp: number;
}

/** context compaction 이벤트 (타임라인 표시용) */
export interface CompactionItem {
  id: string;
  message: string;
  timestamp: number;
}

/** 스텝 진행 정보 */
export interface StepInfo {
  stepNumber: number;
  tokens: number;
}

/** 질문 정보 */
export interface PendingQuestion {
  id: string;
  question: string;
  options?: Array<{ label: string; description?: string }>;
  inputMode?: 'text' | 'secret';
}

/** Proactive Dashboard (API 응답) */
export interface ProactiveDashboard {
  stats: {
    totalExecutions: number;
    level1Executions: number;
    level2Executions: number;
    successRate: number;
    revertRate: number;
    patternsTracked: number;
    level1Patterns: number;
    level2Patterns: number;
  };
  patterns: Array<{
    patternKey: string;
    currentLevel: number;
    acceptCount: number;
    autoExecuteCount: number;
    autoSuccessCount: number;
    consecutiveFailures: number;
    lastUpdated: string;
  }>;
  recentExecutions: Array<{
    id: string;
    timestamp: string;
    level: number;
    domain: string;
    description: string;
    action: string;
    success: boolean;
    resultSummary: string;
    durationMs: number;
    userFeedback: string;
  }>;
  engine: {
    running: boolean;
    evaluationCount: number;
    acceptRate: number;
    pendingCount: number;
    interactionCount: number;
  };
  pendingSuggestions: Array<{
    id: string;
    type: string;
    priority: string;
    title: string;
    description: string;
    confidence: number;
    createdAt: string;
    action?: { command?: string; autoExecutable?: boolean };
  }>;
}

/** 첨부 파일 (전송 전 UI 상태) */
export interface PendingAttachment {
  id: string;
  name: string;
  mimeType: string;
  size: number;
  dataUrl: string;
  preview?: string;
}
