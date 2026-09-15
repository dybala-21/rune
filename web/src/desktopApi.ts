import { ensureWebAuth, type ComputerState } from './api';

export interface DesktopState {
  sessionId: string;
  enabled: boolean;
  revision: number;
  apps?: Array<{ id: string; name: string }>;
  runId?: string | null;
  runState?: ComputerState['state'];
  uncertainAction?: boolean;
  nativeReview?: boolean;
  waiting?: boolean;
  progress?: { change: 'initial' | 'changed' | 'unchanged'; unchangedObservations: number };
  inputObservation?: { change: 'changed' | 'no_visible_change'; unchangedAttempts: number };
  conditionCheck?: { status: 'matched'; scope: 'visible_app_state'; polls: number; elapsedMs: number };
  accessRequested?: boolean;
  connectionError?: string;
  expiresIn?: number;
  observation?: string;
  app?: string;
  title?: string;
  width?: number;
  height?: number;
  pending?: { id: string; app: string; appName: string; action: Record<string, unknown>; expiresAt: number;
    target?: { name: string; role: string; bounds?: { x: number; y: number; width: number; height: number } } | null } | null;
}

export interface DesktopSetup {
  available: boolean;
  accessibility?: boolean;
  screenRecording?: boolean;
  appPath?: string;
  signing?: 'ad_hoc' | 'certificate' | 'unknown';
  apps: Array<{ id: string; name: string }>;
  error?: string;
}

export async function desktopRequest<T>(path: string, body?: unknown): Promise<T> {
  await ensureWebAuth();
  const response = await fetch(`/api/desktop/${path}`, {
    method: body === undefined ? 'GET' : 'POST', credentials: 'include', cache: 'no-store',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(body === undefined ? 20000 : 190000),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Desktop request failed (${response.status}).`);
  }
  return response.json();
}
