/** SSE 이벤트 타입 (서버에서 수신) */

export interface TokenUsage {
  total: number;
  input: number;
  output: number;
  cacheRead?: number;
  cacheCreation?: number;
}

/** The one list of subscribable SSE events — useSSE iterates it, the type
    derives from it, so a new event can't be typed without being wired. */
export const SSE_EVENT_TYPES = [
  'connected',
  'agent_start',
  'agent_complete',
  'agent_error',
  'agent_aborted',
  'step_start',
  'thinking',
  'tool_call',
  'tool_result',
  'text_delta',
  'approval_request',
  'question',
  'context_compaction',
  'delegate_event',
  'command_result',
  'goal_iteration',
  'orchestration_started',
  'orchestration_task_progress',
  'orchestration_task_retry',
  'orchestration_completed',
  'suggestion_created',
  'proactive_execution_started',
  'proactive_execution_completed',
  'autonomy_level_changed',
] as const;

export type SseEventType = typeof SSE_EVENT_TYPES[number];

export interface ConnectedData { clientId: string }
export interface AgentStartData {
  goal: string;
  /** Conversation that started the run — lets the originating tab skip the
      "Goal:" echo line while other surfaces still show it. */
  sessionId?: string | null;
}
export interface TrustInfo {
  verified: boolean;
  reason: string;
  /** A step hit the tool-round cap and was cut off without a final LLM turn —
      the answer may silently omit work that never ran. */
  budgetExhausted?: boolean;
  /** Project tests after the last code change: green, not green, or null when
      nothing was edited. Weaker than an Evidence Gate check — the suite may
      have passed before the change too — so it never reads as "verified". */
  testsPassedAfterEdit?: boolean | null;
  evidenceGate?: {
    hasCheck: boolean;
    lastVerdict: string;
    verdictCounts: Record<string, number>;
    lastEvidence: string;
  };
  honestNote?: string;
  escalationHint?: string;
}
export interface AgentCompleteData { success: boolean; answer: string; durationMs: number; usage?: TokenUsage; trust?: TrustInfo }
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
export interface CommandResultData {
  command: string;
  output: string;
  data?: {
    action?: string;
    sessionId?: string;
    turns?: { role: string; content: string; goalType?: string }[];
    workspace?: string;
  };
}
export interface GoalIterationData {
  n: number;
  verdict: string;
  reason: string;
  evidence: number;
  tokens: number;
}
export interface OrchestrationStartedData { runId?: string; taskCount: number; description: string }
export interface OrchestrationTaskProgressData {
  runId?: string;
  taskId: string;
  completed: number;
  total: number;
  success: boolean;
  description: string;
  role: string;
}
export interface OrchestrationTaskRetryData {
  runId?: string;
  taskId: string;
  failureType: string;
  attempt: number;
  error: string;
}
export interface OrchestrationCompletedData {
  runId?: string;
  success: boolean;
  durationMs: number;
  completedCount: number;
  failedCount: number;
}
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
  /** Severity for system messages; 'error' renders with a danger tone. */
  level?: 'info' | 'error';
  /** Attached to a 'trust' message: the verify-or-fail-honestly verdict. */
  trust?: TrustInfo;
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
  /** step_start 기준으로 이 호출이 속한 에이전트 스텝 번호 (진행 타임라인 그룹핑용) */
  step?: number;
}

/** 위임 실행의 태스크 체크리스트 항목 (orchestration_* 이벤트에서 수집) */
export interface OrchestrationTask {
  taskId: string;
  description: string;
  role: string;
  success?: boolean;
  retries: number;
}

/** 위임 실행 전체 상태 (진행 패널의 선행 체크리스트 데이터) */
export interface OrchestrationState {
  description: string;
  completed: number;
  total: number;
  tasks: OrchestrationTask[];
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
