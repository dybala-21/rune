/** REST API 호출 (Client → Server) */

const BASE = '';

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
    throw new Error((err as { error?: string }).error || res.statusText);
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

async function post<T>(path: string, body?: unknown): Promise<T> {
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
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error((err as { error?: string }).error || res.statusText);
  }
  return res.json() as Promise<T>;
}

/** v1 RPC 호출 */
async function rpc<T>(method: string, params: unknown = {}): Promise<T> {
  const result = await post<{ success: boolean; data?: T; error?: { message: string } }>(
    '/api/v1/rpc',
    { method, params },
  );
  if (!result.success && result.error) {
    throw new Error(result.error.message);
  }
  return result.data as T;
}

export interface MessageAttachment {
  name: string;
  mimeType: string;
  data: string;  // base64
}

export function sendMessage(text: string, attachments?: MessageAttachment[]) {
  return post('/api/message', { text, attachments });
}

export function sendAbort() {
  return post('/api/abort');
}

export function sendApproval(id: string, decision: 'approve_once' | 'approve_always' | 'deny', userGuidance?: string) {
  return post('/api/approval', { id, decision, userGuidance });
}

export function sendQuestion(id: string, answer: string, selectedIndex?: number) {
  return post('/api/question', { id, answer, selectedIndex });
}

// ── Sessions API ──

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

export async function fetchSessionEvents(sessionId: string, params?: {
  runId?: string;
  includeTools?: boolean;
  includeThinking?: boolean;
}): Promise<{ events: EventLogEntry[]; runs: string[] }> {
  return rpc('sessions.events', { sessionId, ...params });
}

// ── Skills API ──

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

// ── Env API ──

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

export async function setEnvVar(key: string, value: string, scope: 'user' | 'project'): Promise<void> {
  return rpc('env.set', { key, value, scope });
}

export async function unsetEnvVar(key: string, scope: 'user' | 'project'): Promise<void> {
  return rpc('env.unset', { key, scope });
}

// ── Config API ──

export interface ConfigInfo {
  proactiveEnabled: boolean;
  gatewayChannels: string[];
  maxConcurrency: number;
  version: string;
  activeModel: {
    provider: string;
    model: string;
    source: 'active' | 'default';
  };
  memoryTuning: {
    preset: 'speed' | 'balanced' | 'accuracy' | null;
    policyMode: 'auto' | 'legacy' | 'shadow' | 'balanced' | 'strict';
    uncertainScoreThreshold: number;
    uncertainRelevanceFloor: number;
    uncertainSemanticLimit: number;
    uncertainSemanticMinScore: number;
    rolloutObservationWindowDays: number;
    rolloutMinShadowSamples: number;
    rolloutPromoteBalancedMinSuccessRate: number;
    rolloutRollbackMaxP95Ms: number;
  };
  safetyTuning: {
    preset: 'conservative' | 'balanced' | 'developer' | null;
    rolloutMode: 'auto' | 'shadow' | 'balanced' | 'strict' | 'legacy';
    autoEnabled: boolean;
  };
}

export async function fetchConfig(): Promise<ConfigInfo> {
  return rpc('config.get', {});
}

export async function patchConfig(params: {
  proactiveEnabled?: boolean;
  memoryTuning?: {
    scope?: 'user' | 'project';
    preset?: 'speed' | 'balanced' | 'accuracy';
    policyMode?: 'auto' | 'legacy' | 'shadow' | 'balanced' | 'strict';
    uncertainScoreThreshold?: number;
    uncertainRelevanceFloor?: number;
    uncertainSemanticLimit?: number;
    uncertainSemanticMinScore?: number;
    rolloutObservationWindowDays?: number;
    rolloutMinShadowSamples?: number;
    rolloutPromoteBalancedMinSuccessRate?: number;
    rolloutRollbackMaxP95Ms?: number;
  };
  safetyTuning?: {
    preset?: 'conservative' | 'balanced' | 'developer';
  };
}): Promise<void> {
  return rpc('config.patch', params);
}

// ── Cron API ──

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
  enabled?: boolean;
  maxRuns?: number;
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

// ── Health API ──

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

// ── Channels API ──

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

// ---------------------------------------------------------------------------
// MCP Servers
// ---------------------------------------------------------------------------

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
