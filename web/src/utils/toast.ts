// A tiny toast bus. It is a plain module emitter rather than React context so
// non-component code (api helpers, clipboard handlers) can raise a toast too.
// The <Toaster> component subscribes and renders.

export type ToastKind = 'info' | 'success' | 'error';
export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
  ttl: number;
}

type Listener = (toasts: Toast[]) => void;

let toasts: Toast[] = [];
let seq = 0;
const listeners = new Set<Listener>();
const timers = new Map<number, ReturnType<typeof setTimeout>>();

function emit() {
  const snapshot = toasts.slice();
  listeners.forEach(l => l(snapshot));
}

export function subscribe(l: Listener): () => void {
  listeners.add(l);
  l(toasts.slice());
  return () => { listeners.delete(l); };
}

export function dismissToast(id: number): void {
  const t = timers.get(id);
  if (t) { clearTimeout(t); timers.delete(id); }
  toasts = toasts.filter(x => x.id !== id);
  emit();
}

export function showToast(message: string, kind: ToastKind = 'info', ttl = 3000): number {
  const id = ++seq;
  toasts = [...toasts, { id, kind, message, ttl }].slice(-4); // cap visible
  emit();
  if (ttl > 0) {
    timers.set(id, setTimeout(() => dismissToast(id), ttl));
  }
  return id;
}

export const toast = {
  info: (m: string, ttl?: number) => showToast(m, 'info', ttl),
  success: (m: string, ttl?: number) => showToast(m, 'success', ttl),
  error: (m: string, ttl?: number) => showToast(m, 'error', ttl ?? 5000),
};
