/** REST API 호출 (Client → Server) */

import type { ProactiveFeedItem } from './types';

const BASE = '';

export async function respondToSuggestion(suggestionId: string, response: 'accept' | 'dismiss') {
  const result = await post<{ acknowledged: boolean; executionStatus: string }>('/api/v1/proactive/feedback', {
    suggestionId, response,
  });
  window.dispatchEvent(new Event('rune:proactive-changed'));
  return result;
}

export async function fetchProactiveSuggestions(signal?: AbortSignal): Promise<ProactiveFeedItem[]> {
  await ensureWebAuth();
  const response = await fetch('/api/v1/proactive/feed', { credentials: 'include', cache: 'no-store', signal });
  if (!response.ok) throw new ApiError('Could not refresh suggestions', response.status);
  const data = await response.json() as { suggestions: ProactiveFeedItem[] };
  return data.suggestions;
}

let _clientId: string | null = null;
let _webAuthReady = false;
let _webAuthPromise: Promise<void> | null = null;

async function bootstrapWebAuth(): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/auth/bootstrap`, {
    method: 'POST',
    credentials: 'include',
  });

  // Older daemon versions may not expose bootstrap endpoint.
  if (res.status === 404 || res.status === 405) {
    _webAuthReady = true;
    return;
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error((err as { error?: string }).error || (typeof err.detail === 'string' ? err.detail : res.statusText));
  }

  _webAuthReady = true;
}

/** SSE 연결 시 받은 clientId를 설정 (X-Client-Id 헤더로 전송) */
export function setClientId(id: string) {
  _clientId = id;
}

export async function ensureWebAuth(): Promise<void> {
  if (_webAuthReady) return;
  if (_webAuthPromise) return _webAuthPromise;

  _webAuthPromise = bootstrapWebAuth().finally(() => {
    if (!_webAuthReady) {
      _webAuthPromise = null;
    }
  });

  return _webAuthPromise;
}

/** Force the next request to re-bootstrap auth (e.g. after a 401 or a dropped stream). */
export function resetWebAuth(): void {
  _webAuthReady = false;
  _webAuthPromise = null;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export function uncertainDelivery(error: unknown): boolean {
  return error instanceof TypeError || error instanceof ApiError && error.status >= 500;
}

async function post<T>(path: string, body?: unknown, retried = false, signal?: AbortSignal): Promise<T> {
  await ensureWebAuth();

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (_clientId) {
    headers['X-Client-Id'] = _clientId;
  }

  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  // Auth expired: re-bootstrap once and retry, so the app recovers without a reload.
  if ((res.status === 401 || res.status === 403) && !retried) {
    resetWebAuth();
    return post<T>(path, body, true, signal);
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new ApiError((err as { error?: string }).error || (typeof err.detail === 'string' ? err.detail : res.statusText), res.status);
  }
  return res.json() as Promise<T>;
}

/** v1 RPC 호출 */
async function rpc<T>(method: string, params: unknown = {}, signal?: AbortSignal): Promise<T> {
  const result = await post<{ success: boolean; data?: T; error?: { message: string } }>(
    '/api/v1/rpc',
    { method, params }, false, signal,
  );
  if (!result.success) {
    throw new Error(result.error?.message || 'Request failed');
  }
  return result.data as T;
}

// Keep the conversation across reloads in this tab. New Chat rotates the ID.
const LIVE_SESSION_KEY = 'rune.live.sessionId';

export function rotateLiveSessionId(): string {
  setCurrentRunId('');
  const id = typeof crypto !== 'undefined' && crypto.randomUUID
    ? `web_${crypto.randomUUID()}`
    : `web_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  try { sessionStorage.setItem(LIVE_SESSION_KEY, id); } catch { /* storage unavailable */ }
  return id;
}

function liveSessionId(): string {
  try {
    return sessionStorage.getItem(LIVE_SESSION_KEY) ?? rotateLiveSessionId();
  } catch {
    return rotateLiveSessionId();
  }
}

// Pin the live chat to an existing conversation (used by /load).
export function setLiveSessionId(id: string): void {
  setCurrentRunId('');
  try { sessionStorage.setItem(LIVE_SESSION_KEY, id); } catch { /* storage unavailable */ }
}

export function getLiveSessionId(): string {
  return liveSessionId();
}

// Accept the first response or agent_start ID; clear it when switching conversations.
let _currentRunId = '';
export function setCurrentRunId(id: string): void { _currentRunId = id; }
export function getCurrentRunId(): string { return _currentRunId; }

export interface ComputerState {
  sessionId: string;
  runId?: string | null;
  state: 'unavailable' | 'idle' | 'running' | 'pausing' | 'paused' | 'manual' | 'stopped';
  lease: number;
  uncertainAction?: boolean;
  frameId?: string;
  title?: string;
  url?: string;
  capturedAt?: number;
  controls?: Array<{ ref: string; role: string; name: string; disabled: boolean }>;
  native?: boolean;
  tabs?: Array<{ id: string; title: string; url: string }>;
  tabId?: string;
  viewport?: { width: number; height: number };
}

declare global {
  interface Window {
    rune?: { desktop: boolean; browserLayout?: (value: { sessionId: string; bounds: { x: number; y: number; width: number; height: number } } | null) => Promise<boolean> };
  }
}

export async function fetchComputer(sessionId: string): Promise<ComputerState> {
  await ensureWebAuth();
  const response = await fetch(`/api/computer/state?sessionId=${encodeURIComponent(sessionId)}&preview=${!window.rune?.browserLayout}`, {
    credentials: 'include', cache: 'no-store', signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) throw new Error(`Could not read the browser (${response.status}).`);
  return response.json();
}

export function controlComputer(state: ComputerState, action: 'pause' | 'takeover' | 'resume' | 'close', instruction = '', acknowledgeUnknown = false): Promise<ComputerState> {
  return post('/api/computer/control', { sessionId: state.sessionId, lease: state.lease, action, instruction, acknowledgeUnknown });
}

export function actOnComputer(state: ComputerState, action: 'click' | 'type' | 'select' | 'check' | 'uncheck' | 'scroll', ref = '', value = ''): Promise<ComputerState> {
  return post('/api/computer/action', { sessionId: state.sessionId, lease: state.lease, frameId: state.frameId, action, ref, value });
}

export function navigateComputer(sessionId: string, action: 'open' | 'back' | 'forward' | 'reload' | 'select_tab' | 'new_tab' | 'close_tab', url = '', tabId = ''): Promise<ComputerState> {
  return post('/api/computer/navigate', { sessionId, action, url, tabId, preview: !window.rune?.browserLayout });
}

export interface BrowserInput {
  action: 'click' | 'drag' | 'scroll' | 'text' | 'key';
  x?: number; y?: number; dx?: number; dy?: number; text?: string;
}

export function inputComputer(state: ComputerState, input: BrowserInput): Promise<ComputerState> {
  return post('/api/computer/input', { sessionId: state.sessionId, lease: state.lease, frameId: state.frameId, ...input });
}

export function stopComputer(runId: string): Promise<{ ok: boolean }> {
  return post('/api/abort', { runId });
}

export async function fetchRunSnapshot(sessionId: string): Promise<{ run: import('./utils/runSnapshot').RunSnapshot | null; available?: boolean }> {
  await ensureWebAuth();
  const response = await fetch(`/api/runs/snapshot?sessionId=${encodeURIComponent(sessionId)}`, {
    credentials: 'include', signal: AbortSignal.timeout(5000),
  });
  if (response.status === 404 || (response.ok && response.headers.get('content-type')?.includes('text/html'))) {
    return { run: null, available: false };
  }
  if (!response.ok) throw new Error(`Could not restore run (${response.status}).`);
  return response.json();
}

export function resumeRun(runId: string): Promise<{ runId: string; sessionId: string }> {
  return post('/api/runs/resume', { runId }, false, AbortSignal.timeout(15000));
}

// Workspace API (directory pinned per conversation)

export async function fetchWorkspace(): Promise<{ path: string }> {
  return rpc('workspace.get', { sessionId: liveSessionId() });
}

export async function setWorkspace(path: string): Promise<{ path: string }> {
  return rpc('workspace.set', { sessionId: liveSessionId(), path });
}

export async function fetchWorkspaceRecents(): Promise<{ paths: string[] }> {
  return rpc('workspace.recents', {});
}

export async function listWorkspaceDirs(dir?: string): Promise<{ dir: string; parent: string; entries: string[] }> {
  return rpc('workspace.listdirs', dir ? { dir } : {});
}

export async function fetchWorkspaceDiff(): Promise<{ diff: string }> {
  return rpc('workspace.diff', { sessionId: liveSessionId() });
}

export async function readWorkspaceFile(path: string): Promise<{ path: string; content: string }> {
  return rpc('files.read', { sessionId: liveSessionId(), path });
}

// Embedded terminal

export async function fetchTerminalStatus(): Promise<{ enabled: boolean }> {
  return rpc('terminal.status', {});
}

export async function mintTerminalToken(): Promise<{ token: string; workspace: string }> {
  return rpc('terminal.token', { sessionId: liveSessionId() });
}

export interface EscalationStatus {
  enabled: boolean;
  provider: string;
  model: string;
  isCloud: boolean;
  suggestion?: string;
}

export async function fetchEscalationStatus(): Promise<EscalationStatus> {
  return rpc('escalation.status', {});
}

export async function setEscalation(provider: string, model: string): Promise<{ provider: string; model: string }> {
  return rpc('escalation.set', { provider, model });
}

/** Available models grouped by provider, for the model picker. */
export async function fetchModels(): Promise<Record<string, string[]>> {
  return rpc('models.list');
}

/** Switch the model new runs use. */
export async function setActiveModel(provider: string, model: string): Promise<{ provider: string; model: string }> {
  const result = await rpc<{ provider: string; model: string }>('model.set', { provider, model });
  window.dispatchEvent(new Event('rune:config-changed'));
  return result;
}

export type ReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/** Set reasoning depth for reasoning-capable models ('' clears to default). */
export async function setReasoningEffort(
  effort: '' | ReasoningEffort, model: { provider: string; model: string },
): Promise<{ reasoningEffort: ReasoningEffort | null }> {
  const result = await rpc<{ reasoningEffort: ReasoningEffort | null }>('reasoning.set', { ...model, effort });
  window.dispatchEvent(new Event('rune:config-changed'));
  return result;
}

export interface MessageAttachment {
  name: string;
  mimeType: string;
  data?: string;
  ref?: string;
}

export async function sendMessage(text: string, attachments?: MessageAttachment[], requestId: string = crypto.randomUUID()) {
  const sessionId = liveSessionId();
  const previousRunId = getCurrentRunId();
  type Reply = { ok: boolean; runId?: string; replayed?: boolean; attachments?: import('./types').SentAttachment[]; run?: import('./utils/runSnapshot').RunSnapshot };
  const send = () => post<Reply>('/api/message', { text, attachments, sessionId, requestId });
  let res: Reply;
  try { res = await send(); }
  catch (error) {
    if (!uncertainDelivery(error) || text.trimStart().startsWith('/')) throw error;
    res = await send();
  }
  if (res?.runId && sessionId === liveSessionId()
      && [previousRunId, res.runId].includes(getCurrentRunId())) setCurrentRunId(res.runId);
  return res;
}

export function sendAbort() {
  const runId = getCurrentRunId();
  if (!runId) return Promise.reject(new Error('The task is still being identified. Try stopping again shortly.'));
  return post<{ stopped?: boolean; stopping?: boolean }>('/api/abort', { runId });
}

export function transcribeAudio(audioBase64: string, mimeType: string) {
  return post<{ ok: boolean; text?: string; error?: string }>(
    '/api/voice/transcribe',
    { audio: audioBase64, mimeType },
  );
}

const interactionResponses = new Map<string, { body: string; responseId: string }>();
function interactionResponseId(id: string, payload: unknown): string {
  const body = JSON.stringify(payload);
  let previous = interactionResponses.get(id);
  if (previous?.body !== body) {
    previous = { body, responseId: crypto.randomUUID() };
    interactionResponses.set(id, previous);
    if (interactionResponses.size > 64) interactionResponses.delete(interactionResponses.keys().next().value!);
  }
  return previous.responseId;
}

export function sendApproval(id: string, decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) {
  const payload = { id, decision, userGuidance };
  return post('/api/approval', { ...payload, responseId: interactionResponseId(id, payload) });
}

export function sendQuestion(id: string, answer: string, selectedIndex?: number) {
  const payload = { id, answer, selectedIndex };
  return post('/api/question', { ...payload, responseId: interactionResponseId(id, payload) });
}

// Sessions API

export interface SessionInfo {
  id: string;
  userId: string;
  title: string;
  status: 'active' | 'archived';
  channel: string;
  turnCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface EventLogEntry {
  event: string;
  data: unknown;
  timestamp: string;
}

export async function fetchSessions(params?: {
  status?: 'active' | 'archived';
  limit?: number;
  offset?: number;
}): Promise<{ sessions: SessionInfo[]; total: number }> {
  return rpc('sessions.list', params ?? {});
}

export interface SessionTurn {
  attachments?: import('./types').SentAttachment[];
  requestId?: string;
  role: string;
  content: string;
  timestamp: string;
}

export interface SessionContents {
  turns: SessionTurn[];
  run?: import('./utils/runSnapshot').RunSnapshot | null;
}

export async function fetchSessionTurns(sessionId: string, signal?: AbortSignal): Promise<SessionContents> {
  return rpc('sessions.turns', { sessionId }, signal);
}

export async function fetchSessionEvents(sessionId: string, params?: {
  runId?: string;
  includeTools?: boolean;
  includeThinking?: boolean;
}): Promise<{ events: EventLogEntry[]; runs: string[] }> {
  return rpc('sessions.events', { sessionId, ...params });
}

// Skills API

export interface SkillInfo {
  name: string;
  description: string;
  scope: 'user' | 'project' | 'builtin';
  lifecycle: string;
  author?: string;
  version?: string;
  category?: string;
  tags?: string[];
  userInvocable?: boolean;
  createdAt?: string;
}

export interface SkillDetail extends SkillInfo {
  body: string;
  frontmatterRaw: string;
}

export async function fetchSkills(scope?: 'user' | 'project' | 'builtin'): Promise<{ skills: SkillInfo[]; projectPath: string; userPath: string }> {
  return rpc('skills.list', scope ? { scope } : {});
}

export async function fetchSkill(name: string): Promise<SkillDetail> {
  return rpc('skills.get', { name });
}

export async function createSkill(params: {
  name: string;
  description: string;
  body: string;
  scope: 'user' | 'project';
  projectPath?: string;
}): Promise<{ name: string; path: string }> {
  return rpc('skills.create', params);
}

export async function updateSkill(params: {
  name: string;
  description?: string;
  body?: string;
}): Promise<{ name: string; path: string }> {
  return rpc('skills.update', params);
}

export async function deleteSkill(name: string): Promise<void> {
  return rpc('skills.delete', { name });
}

// Env API

export interface EnvVarInfo {
  key: string;
  maskedValue: string;
  scope: 'user' | 'project';
  isSecret: boolean;
  category: string;
}

export async function fetchEnvVars(scope?: 'user' | 'project'): Promise<{ variables: EnvVarInfo[]; paths: { user: string; project: string } }> {
  return rpc('env.list', scope ? { scope } : {});
}

export async function setEnvVar(key: string, value: string, scope: 'user' | 'project' | 'effective'): Promise<void> {
  return rpc('env.set', { key, value, scope });
}

export async function unsetEnvVar(key: string, scope: 'user' | 'project'): Promise<void> {
  return rpc('env.unset', { key, scope });
}

// Config API

export interface ConfigInfo {
  executionEnvironment?: { backend: 'local' | 'container'; image: string; allowNetwork: boolean; managed?: boolean };
  proactiveEnabled: boolean;
  advisorEnabled: boolean;
  /** bypass | standard | strict — shown when approvals are switched off. */
  approvalMode?: string;
  gatewayChannels: string[];
  maxConcurrency: number;
  version: string;
  activeModel: {
    provider: string;
    model: string;
    source: 'active' | 'default';
  };
  /** Reasoning depth for the active model, when it accepts one. */
  reasoningEffort?: ReasoningEffort | null;
  reasoningSupported?: boolean;
  reasoningOptions?: ReasoningEffort[];
  reasoningBudgets?: Partial<Record<ReasoningEffort, number>>;
  decisionRouting?: {
    backend: 'connected' | 'jev';
    timeoutMs: number;
    effectiveBackend: 'connected' | 'jev';
    status: 'disabled' | 'ready' | 'unverified' | 'local' | 'automatic_model' | 'missing_key' | 'cooldown' | 'auth_error';
    hasKey: boolean;
    keyScope: 'user' | 'project' | 'process' | null;
  };
  /** Settings consumed by the memory pipeline. */
  memoryTuning: {
    preset: 'speed' | 'balanced' | 'accuracy' | null;
    policyMode: 'legacy' | 'shadow' | 'balanced' | 'strict';
    semanticLimit: number;
    semanticMinScore: number;
    uncertainSemanticLimit: number;
    uncertainSemanticMinScore: number;
    maxEpisodes: number;
    contextMaxChars: number;
  };
  safetyTuning: {
    preset: 'conservative' | 'balanced' | 'developer' | null;
    rolloutMode: 'auto' | 'shadow' | 'balanced' | 'strict' | 'legacy';
    autoEnabled: boolean;
  };
}

/** Run ids the daemon still has in flight. */
export async function fetchActiveRuns(): Promise<{ runIds: string[] }> {
  return rpc('runs.active', {});
}

export async function fetchConfig(): Promise<ConfigInfo> {
  return rpc('config.get', {});
}

export async function patchConfig(params: {
  executionEnvironment?: { backend?: 'local' | 'container'; image?: string };
  proactiveEnabled?: boolean;
  advisorEnabled?: boolean;
  decisionRouting?: { backend?: 'connected' | 'jev'; timeoutMs?: number };
  memoryTuning?: {
    scope?: 'user' | 'project';
    preset?: 'speed' | 'balanced' | 'accuracy';
    policyMode?: 'legacy' | 'shadow' | 'balanced' | 'strict';
    semanticLimit?: number;
    semanticMinScore?: number;
    uncertainSemanticLimit?: number;
    uncertainSemanticMinScore?: number;
    maxEpisodes?: number;
    contextMaxChars?: number;
  };
}): Promise<void> {
  return rpc('config.patch', params);
}

// Cron API

export type CronJobType = 'briefing' | 'check_in' | 'monitoring' | 'reminder' | 'learning' | 'custom';
export type CronJobConditionType = 'skip_if_interacted_today' | 'skip_weekends' | 'skip_if_idle_over' | 'require_channel';

export interface CronJobConditionInfo {
  type: CronJobConditionType;
  params?: Record<string, unknown>;
}

export interface CronJobActorInfo {
  userId: string;
  workspaceId?: string;
  tenantId?: string;
}

export interface CronJobTargetInfo {
  channel?: string;
  recipientId?: string;
  sessionId?: string;
  threadId?: string;
}

export interface CronJobInfo {
  id: string;
  name: string;
  schedule: string;
  command: string;
  goal?: string;
  notifyChannel?: string;
  policy?: RoutinePolicy;
  recentRuns?: RoutineRun[];
  enabled: boolean;
  createdAt: string;
  lastRunAt?: string;
  runCount: number;
  maxRuns?: number;
  type?: CronJobType;
  conditions?: CronJobConditionInfo[];
  dependsOn?: string[];
  actor?: CronJobActorInfo;
  target?: CronJobTargetInfo;
}

export interface RoutinePolicy {
  workspace: string;
  max_steps: number;
  timeout_seconds: number;
  token_budget: number;
  deadline: string | null;
  verification: string[];
  input_paths?: string[];
  output_paths?: string[];
  notify: 'always' | 'changes' | 'failures';
}

export interface RoutineRun {
  id: string;
  started_at: number;
  blocked: boolean;
  result: { status: string; output?: string; error?: string; note?: string; verified?: boolean } | null;
}

export interface CronBuiltinTaskInfo {
  id: string;
  name: string;
  enabled: boolean;
}

export async function fetchCronJobs(params?: { includeBuiltin?: boolean }): Promise<{
  jobs: CronJobInfo[];
  builtinTasks?: CronBuiltinTaskInfo[];
  heartbeatActive: boolean;
}> {
  return rpc('cron.list', params ?? {});
}

export async function createCronJob(params: {
  name: string;
  schedule: string;
  command: string;
  goal?: string;
  notifyChannel?: string;
  policy?: RoutinePolicy;
  enabled?: boolean;
  maxRuns?: number;
  type?: CronJobType;
  conditions?: CronJobConditionInfo[];
  dependsOn?: string[];
  actor?: CronJobActorInfo;
  target?: CronJobTargetInfo;
}): Promise<{ job: CronJobInfo }> {
  return rpc('cron.create', params);
}

export async function updateCronJob(params: {
  jobId: string;
  name?: string;
  schedule?: string;
  command?: string;
  goal?: string;
  notifyChannel?: string;
  policy?: RoutinePolicy;
  enabled?: boolean;
  maxRuns?: number | null;
  type?: CronJobType;
  conditions?: CronJobConditionInfo[];
  dependsOn?: string[];
  actor?: CronJobActorInfo;
  target?: CronJobTargetInfo;
}): Promise<{ job: CronJobInfo }> {
  return rpc('cron.update', params);
}

export async function deleteCronJob(jobId: string): Promise<{ jobId: string }> {
  return rpc('cron.delete', { jobId });
}

export async function reconcileCronJob(jobId: string, operationId: string, note: string): Promise<{ reconciled: boolean }> {
  return rpc('cron.reconcile', { jobId, operationId, note });
}

// Health API

export interface HealthInfo {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  uptime: number;
  subsystems: {
    memory: 'ok' | 'error';
    proactive: 'ok' | 'disabled' | 'error';
    gateway: 'ok' | 'error';
    mcp: 'ok' | 'disabled' | 'error';
    scheduler: { queued: number; running: number; maxConcurrency: number };
  };
}

export async function fetchHealth(): Promise<HealthInfo> {
  return rpc('health', {});
}

// Channels API

export interface ChannelInfo {
  name: string;
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  type: 'in-process' | 'api-client';
  sessionCount: number;
}

export async function fetchChannels(): Promise<{ channels: ChannelInfo[] }> {
  return rpc('channels.list', {});
}

export async function restartChannel(name: string): Promise<void> {
  return rpc('channels.restart', { name });
}

// MCP Servers
export interface MCPServerInfo {
  name: string;
  command: string | null;
  args: string[];
  transport: 'stdio' | 'sse' | 'streamable-http';
  url: string | null;
  disabled: boolean;
  has_env: boolean;
  has_headers: boolean;
}

export interface MCPTestResult {
  name: string;
  success: boolean;
  message: string;
  tools_count: number;
}

export async function fetchMCPServers(): Promise<{ servers: MCPServerInfo[]; count: number }> {
  return rpc('mcp.list');
}

export async function addMCPServer(params: {
  name: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  transport?: 'stdio' | 'sse' | 'streamable-http';
  url?: string;
  headers?: Record<string, string>;
  disabled?: boolean;
}): Promise<MCPServerInfo> {
  return rpc('mcp.add', params);
}

export async function updateMCPServer(name: string, params: {
  name: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  transport?: 'stdio' | 'sse' | 'streamable-http';
  url?: string;
  headers?: Record<string, string>;
  disabled?: boolean;
}): Promise<MCPServerInfo> {
  return rpc('mcp.update', { originalName: name, ...params });
}

export async function deleteMCPServer(name: string): Promise<void> {
  return rpc('mcp.delete', { name });
}

export async function testMCPServer(name: string): Promise<MCPTestResult> {
  return rpc('mcp.test', { name });
}

// Markdown file editor
export interface MarkdownFileInfo {
  key: string;
  label: string;
  description: string;
  exists: boolean;
  size: number;
}

export async function fetchMarkdownFiles(): Promise<MarkdownFileInfo[]> {
  return rpc('markdown.list', {});
}

export async function readMarkdownFile(key: string): Promise<{ key: string; content: string }> {
  return rpc('markdown.read', { key });
}

export async function writeMarkdownFile(key: string, content: string): Promise<{ key: string; saved: boolean }> {
  return rpc('markdown.write', { key, content });
}
