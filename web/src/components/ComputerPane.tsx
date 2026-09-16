import { useCallback, useEffect, useRef, useState } from 'react';
import { actOnComputer, controlComputer, fetchComputer, stopComputer, type ComputerState } from '../api';
import './ComputerPane.css';
import { DesktopPane } from './DesktopPane';
import { desktopRequest, type DesktopState } from '../desktopApi';

const labels: Record<ComputerState['state'], string> = {
  unavailable: 'No browser open', idle: 'Ready', running: 'Rune is working',
  pausing: 'Finishing the current tool…', paused: 'Paused', manual: 'You have control', stopped: 'Stopping…',
};

export function ComputerPane({ sessionId }: { sessionId: string }) {
  const [surface, setSurface] = useState<'browser' | 'desktop'>('browser');
  const chosen = useRef(false);
  const [connectionError, setConnectionError] = useState('');
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    chosen.current = false;
    setSurface('browser');
    const poll = async () => {
      try {
        const state = await desktopRequest<DesktopState>(`status?sessionId=${encodeURIComponent(sessionId)}`);
        if (disposed) return;
        setConnectionError('');
        if (!chosen.current && (state.accessRequested || state.enabled)) {
          setSurface('desktop');
          return;
        }
      } catch (error) {
        if (!disposed) setConnectionError(error instanceof Error ? error.message : String(error));
      }
      if (!disposed && !chosen.current) timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [sessionId]);
  return <><div className="computer-surface" role="tablist" aria-label="Computer surface">
    <button role="tab" aria-selected={surface === 'browser'} onClick={() => { chosen.current = true; setSurface('browser'); }}>Browser</button>
    <button role="tab" aria-selected={surface === 'desktop'} onClick={() => { chosen.current = true; setSurface('desktop'); }}>This Mac</button>
  </div>{connectionError && <div role="alert" className="computer-error">{connectionError}</div>}
  {surface === 'desktop' ? <DesktopPane key={sessionId} sessionId={sessionId} /> : <BrowserPane key={sessionId} sessionId={sessionId} />}</>;
}

function BrowserPane({ sessionId }: { sessionId: string }) {
  const [view, setView] = useState<ComputerState | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [instruction, setInstruction] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [target, setTarget] = useState('');
  const [value, setValue] = useState('');
  const alive = useRef(true);
  const updating = useRef(false);
  const polling = useRef<Promise<ComputerState> | null>(null);
  const latest = useRef(view);
  latest.current = view;

  const refresh = useCallback(async () => {
    const next = await fetchComputer(sessionId);
    if (alive.current) setView(next);
    return next;
  }, [sessionId]);

  useEffect(() => {
    alive.current = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      // Pause polling during manual control so targets stay aligned with the preview.
      if (!updating.current && latest.current?.state !== 'manual' && !document.hidden) {
        polling.current = refresh();
        try { await polling.current; if (alive.current) setError(''); }
        catch (e) { if (alive.current) setError(e instanceof Error ? e.message : String(e)); }
        finally { polling.current = null; }
      }
      if (alive.current) timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => { alive.current = false; clearTimeout(timer); };
  }, [refresh]);

  const perform = async (fn: () => Promise<ComputerState>) => {
    if (updating.current || busy) return;
    updating.current = true;
    setBusy(true);
    setError('');
    try {
      if (polling.current) await polling.current.catch(() => undefined);
      const next = await fn();
      if (!alive.current) return;
      setView(next);
      setTarget('');
      setValue('');
      setConfirmed(false);
      await refresh();
    } catch (e) {
      if (alive.current) setError(e instanceof Error ? e.message : String(e));
      try { await refresh(); } catch { /* Keep the original action error visible. */ }
    } finally {
      updating.current = false;
      if (alive.current) setBusy(false);
    }
  };

  const command = (action: 'pause' | 'takeover' | 'resume' | 'close') => {
    if (!view) return;
    void perform(async () => {
      const next = await controlComputer(view, action, instruction, confirmed);
      if (action === 'resume') setInstruction('');
      return next;
    });
  };
  const manual = view?.state === 'manual';
  const paused = view?.state === 'paused';
  const selected = view?.controls?.find(control => control.ref === target);
  const editable = selected && ['textbox', 'searchbox', 'spinbutton', 'combobox'].includes(selected.role);

  return <section className="computer-pane" aria-label="Computer">
    <div className="computer-toolbar">
      <span className={`computer-status ${manual || paused ? 'held' : ''}`}>
        <span className="computer-status-dot" />{view ? labels[view.state] : 'Connecting…'}
      </span>
      <div className="computer-actions">
        {view?.runId && <button disabled={busy || view.state === 'stopped'} onClick={() => void perform(async () => {
          await stopComputer(view.runId!);
          return fetchComputer(sessionId);
        })}>Stop task</button>}
        {view?.state === 'running' && <button disabled={busy} onClick={() => command('pause')}>Pause</button>}
        {(paused || view?.state === 'idle') && <button disabled={busy || !view.frameId} onClick={() => command('takeover')}>Take control</button>}
        {view && !view.runId && view.state !== 'unavailable' && <button disabled={busy} onClick={() => command('close')}>Close browser</button>}
      </div>
    </div>
    {error && <div role="alert" className="computer-error">{error}</div>}
    {view?.uncertainAction && <div className="computer-warning">
      The last action may have taken effect. Check the page before continuing.
      {(manual || paused) && <label><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />I checked the previous action’s effects</label>}
    </div>}
    {view?.frameId ? <div className="computer-browser">
      <div className="computer-address" title={view.url}><span>↗</span>{view.url}</div>
      <img src={`/api/computer/frame/${view.frameId}?sessionId=${encodeURIComponent(sessionId)}`}
        alt={view.title ? `Browser preview: ${view.title}` : 'Current browser preview'} />
      <div className="computer-caption"><span>{view.title || 'Browser'}</span><span>Preview · {view.capturedAt ? new Date(view.capturedAt).toLocaleTimeString() : ''}</span></div>
    </div> : <div className="computer-empty">
      <div className="computer-empty-icon">▤</div>
      <strong>{view?.state === 'unavailable' ? 'A place to work together' : 'Waiting for the browser'}</strong>
      <p>Ask Rune to open a website. Its screen will appear here, and you can pause to make changes.</p>
    </div>}
    {manual && view && <div className="computer-card">
      <div className="computer-card-title"><strong>Page controls</strong><button disabled={busy} onClick={() => void perform(refresh)}>Refresh page view</button></div>
      <select aria-label="Page control" value={target} disabled={busy} onChange={e => { setTarget(e.target.value); setValue(''); }}>
        <option value="">Choose a control…</option>
        {view.controls?.map(control => <option key={control.ref} value={control.ref} disabled={control.disabled}>{control.name} · {control.role}</option>)}
      </select>
      {editable && <input aria-label="Control value" value={value} disabled={busy} placeholder="Enter a value" onChange={e => setValue(e.target.value)} />}
      <div className="computer-actions">
        <button disabled={busy || !selected || !!view.uncertainAction} onClick={() => void perform(() => actOnComputer(view, editable ? (selected?.role === 'combobox' ? 'select' : 'type') : 'click', target, value))}>{editable ? 'Set value' : 'Click'}</button>
        <button disabled={busy || !view.frameId || !!view.uncertainAction} onClick={() => void perform(() => actOnComputer(view, 'scroll', '', 'up'))}>Scroll up</button>
        <button disabled={busy || !view.frameId || !!view.uncertainAction} onClick={() => void perform(() => actOnComputer(view, 'scroll', '', 'down'))}>Scroll down</button>
      </div>
    </div>}
    {(paused || manual) && view && <div className="computer-card">
      {view.runId && <><label htmlFor="computer-instruction">Update the task</label><textarea id="computer-instruction" rows={3} value={instruction} onChange={e => setInstruction(e.target.value)} placeholder="Tell Rune what to change before continuing…" /></>}
      <button className="computer-resume" disabled={busy || (!!view.uncertainAction && !confirmed)} onClick={() => command('resume')}>{view.runId ? 'Resume Rune' : 'Return control to Rune'}</button>
      <p className="computer-note">Rune will read the page again before making changes.</p>
    </div>}
  </section>;
}
