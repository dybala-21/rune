import { useEffect, useRef, useState } from 'react';
import { fetchComputer, inputComputer, type BrowserInput, type ComputerState } from '../api';

export function BrowserSurface({ view, active = true, onView, onTakeControl, onError }: {
  active?: boolean;
  view: ComputerState; onView: (state: ComputerState) => void;
  onTakeControl: () => void; onError: (message: string) => void;
}) {
  const surface = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const latest = useRef(view);
  latest.current = view;
  const queue = useRef(Promise.resolve());
  const generation = useRef(0);
  const pending = useRef(0);
  const composing = useRef(false);
  const origin = useRef<{ x: number; y: number } | null>(null);
  const [mounted, setMounted] = useState(false);
  const [concealed, setConcealed] = useState(false);
  const native = Boolean(view.native && window.rune?.browserLayout);
  useEffect(() => { generation.current++; }, [view.sessionId, view.lease, active]);
  useEffect(() => () => { generation.current++; }, []);

  useEffect(() => {
    if (!native || !surface.current || !active) return;
    let disposed = false;
    let lastLayout = '';
    const update = () => {
      const rect = surface.current?.getBoundingClientRect();
      if (!rect) return;
      const hidden = !active || !rect.width || !rect.height || document.hidden || Boolean(document.querySelector('[role="dialog"], [role="menu"], [aria-modal="true"]'));
      setConcealed(hidden);
      const layout = hidden ? null : { sessionId: view.sessionId,
        bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height } };
      const key = JSON.stringify(layout);
      if (key === lastLayout) return;
      lastLayout = key;
      void window.rune!.browserLayout!(layout)
        .then(ok => { if (!disposed) { setMounted(ok && !hidden); if (!ok) lastLayout = ''; } })
        .catch(error => { if (!disposed) onError(String(error)); });
    };
    const resize = new ResizeObserver(update);
    resize.observe(surface.current);
    const mutations = new MutationObserver(update);
    mutations.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-modal', 'role'] });
    window.addEventListener('scroll', update, true);
    document.addEventListener('visibilitychange', update);
    update();
    return () => {
      disposed = true; resize.disconnect(); mutations.disconnect();
      window.removeEventListener('scroll', update, true);
      document.removeEventListener('visibilitychange', update);
      void window.rune?.browserLayout?.(null);
    };
  }, [native, view.sessionId, onError, active]);

  const send = (event: BrowserInput) => {
    const version = generation.current;
    pending.current++;
    queue.current = queue.current.then(async () => {
      const state = latest.current;
      if (!active || version !== generation.current || state.state !== 'manual' || state.uncertainAction) return;
      try {
        const next = await inputComputer(state, event);
        latest.current = next;
        onView(next);
      } catch (error) {
        generation.current++;
        onError(error instanceof Error ? error.message : String(error));
        try { const next = await fetchComputer(state.sessionId); latest.current = next; onView(next); }
        catch { /* Keep the dispatch error until the user refreshes. */ }
      }
    }).finally(() => { pending.current--; });
  };
  const point = (clientX: number, clientY: number) => {
    const rect = surface.current!.getBoundingClientRect();
    const viewport = latest.current.viewport!;
    return { x: Math.max(0, Math.min(viewport.width - 1, (clientX - rect.x) * viewport.width / rect.width)),
      y: Math.max(0, Math.min(viewport.height - 1, (clientY - rect.y) * viewport.height / rect.height)) };
  };

  const wheel = useRef<(event: WheelEvent) => void>(() => {});
  wheel.current = event => {
    if (native || latest.current.state !== 'manual' || !latest.current.viewport) return;
    event.preventDefault(); event.stopPropagation();
    if (pending.current) return;
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? latest.current.viewport.height : 1;
    send({ action: 'scroll', ...point(event.clientX, event.clientY),
      dx: Math.max(-10000, Math.min(10000, event.deltaX * unit)), dy: Math.max(-10000, Math.min(10000, event.deltaY * unit)) });
  };
  useEffect(() => {
    const element = surface.current;
    const handle = (event: WheelEvent) => wheel.current(event);
    element?.addEventListener('wheel', handle, { passive: false });
    return () => element?.removeEventListener('wheel', handle);
  }, []);

  return <div ref={surface} className={`browser-surface ${native ? 'native' : 'remote'}`}
    aria-label="Browser page"
    onPointerDown={event => {
      if (native) return;
      if (view.state !== 'manual') { onTakeControl(); return; }
      if (!view.viewport || pending.current || view.uncertainAction) return;
      event.preventDefault(); input.current?.focus();
      origin.current = point(event.clientX, event.clientY);
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onPointerUp={event => {
      if (!origin.current || native) return;
      const start = origin.current, end = point(event.clientX, event.clientY);
      origin.current = null;
      const dx = end.x - start.x, dy = end.y - start.y;
      send(Math.hypot(dx, dy) > 4 ? { action: 'drag', ...start, dx, dy } : { action: 'click', ...start });
    }}
    onPointerCancel={() => { origin.current = null; }}>
    {native ? !mounted && !concealed && <span className="browser-loading">Connecting browser…</span>
      : view.frameId && <img draggable={false} src={`/api/computer/frame/${view.frameId}?sessionId=${encodeURIComponent(view.sessionId)}`} alt={view.title || 'Browser page'} />}
    {!native && <textarea ref={input} className="browser-keyboard" aria-label="Type in browser"
      autoComplete="off" autoCapitalize="off" spellCheck={false}
      onCompositionStart={() => { composing.current = true; }}
      onCompositionEnd={event => {
        composing.current = false;
        if (event.currentTarget.value) send({ action: 'text', text: event.currentTarget.value });
        event.currentTarget.value = '';
      }}
      onChange={event => {
        if (!composing.current && event.target.value) { send({ action: 'text', text: event.target.value }); event.target.value = ''; }
      }}
      onKeyDown={event => {
        if (event.nativeEvent.isComposing || composing.current) return;
        const keys = ['Enter', 'Tab', 'Backspace', 'Delete', 'Escape', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'];
        if (keys.includes(event.key)) {
          event.preventDefault(); event.stopPropagation();
          send({ action: 'key', text: event.key === 'Tab' && event.shiftKey ? 'Shift+Tab' : event.key });
        } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
          event.preventDefault(); send({ action: 'key', text: 'ControlOrMeta+A' });
        }
      }} />}
  </div>;
}
