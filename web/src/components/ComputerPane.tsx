import { useCallback, useEffect, useRef, useState } from 'react';
import { controlComputer, fetchComputer, navigateComputer, stopComputer, type ComputerState } from '../api';
import './ComputerPane.css';
import { RetainedPane } from './RetainedPane';
import { DesktopPane } from './DesktopPane';
import { desktopRequest, type DesktopState } from '../desktopApi';
import { BrowserSurface } from './BrowserSurface';
import { ComputerIcon } from './ComputerIcon';

const labels: Record<ComputerState['state'], string> = {
  unavailable: 'No browser open', idle: 'Ready', running: 'Rune is browsing',
  pausing: 'Pausing safely…', paused: 'Rune is paused', manual: 'You have control', stopped: 'Stopping…',
};

export function ComputerPane({ sessionId, active = true }: { sessionId: string; active?: boolean }) {
  const [surface, setSurface] = useState<'browser' | 'desktop'>('browser');
  const chosen = useRef(false);
  const [connectionError, setConnectionError] = useState('');
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    if (!active || chosen.current) return;
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
  }, [sessionId, active]);
  return <div className="computer-workspace"><div className="computer-surface-bar">
    <div className="computer-surface" role="group" aria-label="Computer surface">
      <button aria-pressed={surface === 'browser'} onClick={() => { chosen.current = true; setSurface('browser'); }}><ComputerIcon name="browser" />Browser</button>
      <button aria-pressed={surface === 'desktop'} onClick={() => { chosen.current = true; setSurface('desktop'); }}><ComputerIcon name="computer" />This Mac</button>
    </div>
  </div>{surface === 'desktop' && connectionError && <div role="alert" className="computer-error">{connectionError}</div>}
  <RetainedPane active={surface === 'desktop'}><DesktopPane key={sessionId} sessionId={sessionId} visible={active && surface === 'desktop'} /></RetainedPane>
  <RetainedPane active={surface === 'browser'}><BrowserPane key={sessionId} sessionId={sessionId} active={active && surface === 'browser'} /></RetainedPane></div>;
}

function BrowserPane({ sessionId, active }: { sessionId: string; active: boolean }) {
  const [view, setView] = useState<ComputerState | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [instruction, setInstruction] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  useEffect(() => { setConfirmed(false); }, [active, view?.lease, view?.runId, view?.uncertainAction]);
  const [address, setAddress] = useState('');
  const addressInput = useRef<HTMLInputElement>(null);
  const editingAddress = useRef(false);
  const alive = useRef(true);
  const updating = useRef(false);
  const polling = useRef<Promise<ComputerState> | null>(null);
  const latest = useRef(view);
  latest.current = view;
  useEffect(() => {
    if (!editingAddress.current) setAddress(view?.url === 'about:blank' ? '' : view?.url ?? '');
  }, [view?.url]);

  const refresh = useCallback(async () => {
    const next = await fetchComputer(sessionId);
    if (alive.current) setView(next);
    return next;
  }, [sessionId, active]);

  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  useEffect(() => {
    if (!active) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      // Pause polling during manual control so targets stay aligned with the preview.
      if (!updating.current && (latest.current?.state !== 'manual' || latest.current?.native && window.rune?.browserLayout) && !document.hidden) {
        polling.current = refresh();
        try { await polling.current; if (alive.current) setError(''); }
        catch (e) { if (alive.current) setError(e instanceof Error ? e.message : String(e)); }
        finally { polling.current = null; }
      }
      if (!disposed) timer = setTimeout(poll, latest.current?.native && window.rune?.browserLayout ? 500 : 1000);
    };
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [refresh, active]);

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
  const takeControl = () => void perform(async () => {
    let state = await fetchComputer(sessionId);
    if (state.state === 'running') state = await controlComputer(state, 'pause');
    const deadline = Date.now() + 15000;
    while (state.state === 'pausing' && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 100));
      state = await fetchComputer(sessionId);
    }
    return state.state === 'manual' ? state : controlComputer(state, 'takeover');
  });
  const navigate = (action: Parameters<typeof navigateComputer>[1], tabId = '') => {
    const url = /^https?:\/\//i.test(address.trim()) ? address.trim() : `https://${address.trim()}`;
    void perform(() => navigateComputer(sessionId, action, action === 'new_tab' ? '' : url, tabId));
  };
  const hasPage = Boolean(view && (view.frameId || view.native));
  const navigationLocked = busy || Boolean(view?.runId && !manual) || Boolean(view?.uncertainAction);
  const transitioning = view?.state === 'pausing' || view?.state === 'stopped';
  const canTakeControl = hasPage && !manual && !transitioning;

  return <section className="computer-pane browser-pane" aria-label="Browser workspace" aria-busy={busy}>
    <div className="browser-window">
    <div className="browser-chrome">
    {view?.native && <div className="browser-tabs" aria-label="Browser tabs">
      <div className="browser-tab-list" role="tablist" aria-label="Open pages">
        {view.tabs?.map(tab => <div className="browser-tab" data-selected={tab.id === view.tabId} key={tab.id}>
          <button role="tab" aria-selected={tab.id === view.tabId} title={tab.url} disabled={navigationLocked} onClick={() => navigate('select_tab', tab.id)}><ComputerIcon name="globe" size={13} /><span>{tab.title || 'New tab'}</span></button>
          <button className="browser-tab-close" aria-label={`Close ${tab.title || 'tab'}`} disabled={navigationLocked} onClick={() => navigate('close_tab', tab.id)}><ComputerIcon name="close" size={12} /></button>
        </div>)}
      </div>
      <button className="browser-icon-button" title="New tab" aria-label="New browser tab" disabled={navigationLocked} onClick={() => navigate('new_tab')}><ComputerIcon name="plus" /></button>
    </div>}
    <form className="browser-navigation" onSubmit={event => { event.preventDefault(); navigate('open'); }}>
      <button className="browser-icon-button" type="button" title="Back" aria-label="Back" disabled={navigationLocked || !view?.url} onClick={() => navigate('back')}><ComputerIcon name="back" /></button>
      <button className="browser-icon-button" type="button" title="Forward" aria-label="Forward" disabled={navigationLocked || !view?.url} onClick={() => navigate('forward')}><ComputerIcon name="forward" /></button>
      <button className="browser-icon-button" type="button" title="Reload page" aria-label="Reload page" disabled={navigationLocked || !view?.url} onClick={() => navigate('reload')}><ComputerIcon name="reload" /></button>
      <div className="browser-address-field"><ComputerIcon name="globe" size={14} />
      <input ref={addressInput} readOnly={navigationLocked} aria-label="Website address" placeholder="Enter a website address" value={address} autoComplete="off" spellCheck={false}
        onFocus={() => { editingAddress.current = true; }} onBlur={() => { editingAddress.current = false; }}
        onChange={event => setAddress(event.target.value)} />
      <button className="browser-icon-button" type="submit" title="Open website" aria-label="Open website" disabled={navigationLocked || !address.trim()}><ComputerIcon name="forward" size={14} /></button>
      </div>
    </form>
    </div>
    {error && <div role="alert" className="computer-error">{error}</div>}
    {view?.uncertainAction && <div className="computer-warning">
      The last action may have taken effect. Check the page before continuing.
      {(manual || paused) && <label><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />I checked the previous action’s effects</label>}
    </div>}
    {view && hasPage ? <div className="browser-page">
      <BrowserSurface active={active} view={view} onView={setView} onTakeControl={takeControl} onError={setError} />
    </div> : <div className="browser-empty">
      <span className="browser-empty-icon" aria-hidden="true"><ComputerIcon name="browser" size={28} /></span>
      <h3>{view?.runId ? 'Waiting for the browser' : 'Open a website'}</h3>
      <p>{view?.runId ? 'The page will appear here when it’s ready.' : 'Enter an address above or ask Rune to browse.'}</p>
      {!view?.runId && <button className="browser-open-button" onClick={() => addressInput.current?.focus()}><ComputerIcon name="plus" size={14} />Enter address</button>}
    </div>}
    </div>
    {(paused || manual) && view?.runId && <details className="browser-instruction">
      <summary>Add instructions before resuming{instruction.trim() && <span>Draft</span>}</summary>
      <textarea aria-label="Instructions for Rune" rows={2} value={instruction} onChange={e => setInstruction(e.target.value)} placeholder="Tell Rune what to change…" />
    </details>}
    {(hasPage || view?.runId || transitioning) && <div className="browser-control-bar" data-state={view?.state ?? 'connecting'}>
      <div className="browser-owner">
        <span className="browser-owner-icon">{busy || transitioning ? <span className="spinner" /> : <ComputerIcon name={manual ? 'pointer' : 'computer'} size={17} />}</span>
        <div role="status"><strong>{view ? labels[view.state] : 'Connecting…'}</strong>
          {(paused || manual && view?.runId || transitioning) && <span>{manual ? 'Make your changes, then resume.'
            : paused ? 'Take control or resume Rune.' : 'Finishing the current action…'}</span>}
        </div>
      </div>
      <div className="browser-control-actions">
        {view && !view.runId && hasPage && <button className="browser-icon-button" title="Close browser" aria-label="Close browser" disabled={busy} onClick={() => command('close')}><ComputerIcon name="close" /></button>}
        {view?.state === 'running' && <button className="browser-icon-button" title="Pause Rune" aria-label="Pause Rune" disabled={busy} onClick={() => command('pause')}><ComputerIcon name="pause" /></button>}
        {view?.runId && <button className="browser-icon-button" title="Stop task" aria-label="Stop task" disabled={busy || view.state === 'stopped'} onClick={() => void perform(async () => {
          await stopComputer(view.runId!);
          return fetchComputer(sessionId);
        })}><ComputerIcon name="stop" /></button>}
        {canTakeControl && <button className="browser-primary" disabled={busy} onClick={takeControl}><ComputerIcon name="pointer" size={14} />Take control</button>}
        {(paused || manual) && view?.runId && <button className="browser-primary" disabled={busy || (!!view.uncertainAction && !confirmed)} onClick={() => command('resume')}><ComputerIcon name="play" size={13} />Resume Rune</button>}
      </div>
    </div>}
  </section>;
}
