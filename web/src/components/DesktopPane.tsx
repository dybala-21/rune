import { useEffect, useId, useRef, useState } from 'react';
import { controlComputer, fetchComputer, stopComputer } from '../api';
import { desktopRequest, type DesktopSetup, type DesktopState } from '../desktopApi';

export function DesktopPane({ sessionId, setupOnMount = false }: { sessionId: string; setupOnMount?: boolean }) {
  const instructionId = useId();
  const [view, setView] = useState<DesktopState | null>(null);
  const [setup, setSetup] = useState<DesktopSetup | null>(null);
  const [apps, setApps] = useState<string[]>([]);
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [instruction, setInstruction] = useState('');
  const [checked, setChecked] = useState(false);
  const active = useRef(true);
  const inflight = useRef(false);
  const latest = useRef(view);
  latest.current = view;

  const refresh = async () => {
    const next = await desktopRequest<DesktopState>(`status?sessionId=${encodeURIComponent(sessionId)}`);
    if (active.current) setView(next);
    return next;
  };

  useEffect(() => {
    if (!setupOnMount) return;
    let disposed = false;
    void desktopRequest<DesktopSetup>('setup').then(next => {
      if (!disposed) setSetup(next);
    }).catch(e => { if (!disposed) setError(e instanceof Error ? e.message : String(e)); });
    return () => { disposed = true; };
  }, [sessionId, setupOnMount]);

  useEffect(() => {
    if (!setup?.available || setup.accessibility && setup.screenRecording) return;
    let disposed = false;
    let checking = false;
    const check = async () => {
      if (checking || inflight.current || document.hidden) return;
      checking = true;
      try {
        const next = await desktopRequest<DesktopSetup>('setup');
        if (!disposed) setSetup(next);
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      } finally { checking = false; }
    };
    const timer = setInterval(() => void check(), 3000);
    window.addEventListener('focus', check);
    return () => { disposed = true; clearInterval(timer); window.removeEventListener('focus', check); };
  }, [setup?.available, setup?.accessibility, setup?.screenRecording]);

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    active.current = true;
    const poll = async () => {
      if (!inflight.current && !document.hidden) {
        try {
          const next = await desktopRequest<DesktopState>(`status?sessionId=${encodeURIComponent(sessionId)}`);
          if (!disposed) setView(next);
        } catch (e) { if (!disposed) setError(e instanceof Error ? e.message : String(e)); }
      }
      if (!disposed) timer = setTimeout(poll, 1000);
    };
    void poll();
    return () => { disposed = true; active.current = false; clearTimeout(timer); };
  }, [sessionId]);

  const perform = async (action: () => Promise<unknown>) => {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setError('');
    try { await action(); if (active.current) await refresh(); }
    catch (e) { if (active.current) setError(e instanceof Error ? e.message : String(e)); }
    finally { inflight.current = false; if (active.current) setBusy(false); }
  };
  const command = (action: string, extra: Record<string, unknown> = {}) => perform(() => desktopRequest('control', {
    sessionId, action, revision: latest.current?.revision ?? 0, ...extra,
  }));
  const pending = view?.pending;
  const paused = view?.runState === 'paused';
  const stop = async () => {
    if (!view?.runId || stopping) return;
    setStopping(true);
    try { await stopComputer(view.runId); await refresh(); }
    catch (e) { if (active.current) setError(e instanceof Error ? e.message : String(e)); }
    finally { if (active.current) setStopping(false); }
  };

  return <section className="computer-pane" aria-label="This Mac">
    <div className="computer-toolbar">
      <span className={`computer-status ${pending || paused ? 'held' : ''}`}><span className="computer-status-dot" />
        {!view ? 'Reading app access…' : view.accessRequested ? 'Connect apps to continue this task' : paused ? 'Paused' : view.nativeReview ? 'Waiting for approval on Mac' : pending ? 'Review next action' : view.waiting ? 'Waiting for the app…' : view.enabled ? 'Selected apps connected' : 'Desktop access is off'}
      </span>
      <div className="computer-actions">
        {view?.runId && <button disabled={stopping || view.runState === 'stopped'} onClick={() => void stop()}>Stop task</button>}
        {view?.runState === 'running' && !view.accessRequested && <button disabled={busy} onClick={() => void perform(async () => {
          const computer = await fetchComputer(sessionId);
          await controlComputer(computer, 'pause');
        })}>Pause</button>}
        {view?.enabled && <button disabled={busy} onClick={() => void command('disconnect')}>Disconnect apps</button>}
      </div>
    </div>
    {error && <div role="alert" className="computer-error">{error}</div>}
    {view?.connectionError && <p role="status" className="computer-warning">{view.connectionError}</p>}
    {view && !view.enabled && <div className="computer-card">
      <strong>Work in your Mac apps</strong>
      <p className="computer-note">Choose the apps Rune may see and use for 30 minutes. Rune uses them when the task needs app access; other requests can use answers, search, coding and file tools. Rune Computer asks you to confirm access in a macOS dialog. Each input also needs native approval in this preview.</p>
      {view?.accessRequested && <p role="status" className="computer-note">This request is waiting for app access. Connect the requested app below and Rune will continue the same task.</p>}
      <button disabled={busy || !!view?.runId && !view.accessRequested} onClick={() => void perform(async () => {
        const next = await desktopRequest<DesktopSetup>('setup');
        if (active.current) setSetup(next);
      })}>{setup ? 'Refresh permissions and apps' : 'Set up app access'}</button>
      {setup && !setup.available && <p role="status" className="computer-note">{setup.error}</p>}
      {setup?.available && <>
        <p className="computer-note">Accessibility: {setup.accessibility ? 'Allowed' : 'Required'} · Screen Recording: {setup.screenRecording ? 'Allowed' : 'Required'}</p>
        {(!setup.accessibility || !setup.screenRecording) && <p className="computer-note">
          In System Settings → Privacy &amp; Security, enable Rune Computer under {!setup.accessibility ? 'Accessibility' : 'Screen Recording'}.
          {!setup.accessibility && !setup.screenRecording && ' Enable Screen Recording as well.'} Rune checks again when you return.
        </p>}
        {(!setup.accessibility || !setup.screenRecording) && <button disabled={busy} onClick={() => void perform(() => desktopRequest('permissions', {}))}>
          {busy ? 'Opening System Settings…' : !setup.accessibility ? 'Open Accessibility settings' : 'Open Screen Recording settings'}
        </button>}
        {(!setup.accessibility || !setup.screenRecording) && <details className="computer-note">
          <summary>Already enabled in System Settings?</summary>
          <p>Restart Rune Computer to check the permissions in a new process. If access is still required, remove only the old Rune Computer entry from Accessibility and Screen Recording, then add the app shown below and enable it.</p>
          {setup.signing === 'ad_hoc' && <p>This local build can need permission registration again after an update.</p>}
          {setup.appPath && <p>{setup.appPath}</p>}
          <div className="computer-actions">
            <button disabled={busy || !!view?.runId} onClick={() => void perform(async () => {
              const next = await desktopRequest<DesktopSetup>('restart', {});
              if (active.current) setSetup(next);
            })}>Restart and check permissions</button>
            <button disabled={busy || !!view?.runId} onClick={() => void perform(() => desktopRequest('reveal', {}))}>Show Rune Computer in Finder</button>
          </div>
        </details>}
        <div className="desktop-app-list" aria-label="Allowed apps">
          {setup.apps.map(app => <label key={app.id}><input type="checkbox" checked={apps.includes(app.id)} disabled={busy}
            onChange={e => setApps(old => e.target.checked ? [...old, app.id] : old.filter(id => id !== app.id))} />
            <span>{app.name}<small>{app.id}</small></span></label>)}
        </div>
        <label className="desktop-consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} />
          <span>I allow the selected app windows and their text to be sent to this conversation’s model provider.</span></label>
        <button disabled={busy || !consent || apps.length === 0 || apps.length > 12 || !setup.accessibility || !setup.screenRecording || !!view?.runId && !view.accessRequested}
          onClick={() => void command('grant', { apps })}>Connect selected apps</button>
      </>}
    </div>}
    {view?.enabled && <p className="computer-note">{view.apps?.map(app => app.name).join(' · ')} · {Math.ceil((view.expiresIn ?? 0) / 60)} min remaining. Ask Rune to work in one of these apps.</p>}
    {view?.nativeReview && !paused && <p role="status" className="computer-note">Check the Rune Computer dialog on your Mac. Pause or Stop task cancels any input that has not been sent.</p>}
    {view?.uncertainAction && <div className="computer-warning">
      The previous action may have taken effect. Pause the task and inspect the app before continuing.
      {(paused || !view.runId) && <><label className="desktop-consent"><input type="checkbox" checked={checked} onChange={e => setChecked(e.target.checked)} />I checked the app and its previous action</label>
        <button disabled={busy || !checked} onClick={() => void command('acknowledge', { approved: true })}>Confirm checked</button></>}
    </div>}
    {view?.observation && <div className="computer-browser">
      <div className="computer-address">{view.title || view.app}</div>
      <img src={`/api/desktop/frame/${encodeURIComponent(view.observation)}?sessionId=${encodeURIComponent(sessionId)}`} alt={`App preview: ${view.title || view.app}`} />
      <div className="computer-caption"><span>Window preview · {view.width} × {view.height}</span>
        <span>{view.conditionCheck ? 'Screen condition matched' : view.progress?.change === 'changed' ? 'Screen changed' : view.progress?.change === 'unchanged' ? 'Screen unchanged' : 'Observed state'}</span></div>
      {view.inputObservation?.change === 'no_visible_change' && <p role="status" className="computer-note">No visible change after the last input. Rune should check the result before repeating it.</p>}
    </div>}
    {pending && <div className="computer-card" aria-label="Review desktop action">
      <strong>{pending.appName} · {String(pending.action.action)}</strong>
      {pending.target && <p className="computer-note">Control: {pending.target.name || pending.target.role}</p>}
      <dl className="desktop-action-detail">{Object.entries(pending.action).filter(([key]) => !['observation', 'action'].includes(key)).map(([key, value]) =>
        <div key={key}><dt>{key}</dt><dd>{Array.isArray(value) ? value.join(' + ') : String(value)}</dd></div>)}</dl>
      <p className="computer-note">Review the action in the Rune Computer dialog on your Mac. Rune will recheck the target window before sending input.</p>
      <div className="computer-actions">
        <button disabled={busy} onClick={() => void command('decide', { actionId: pending.id, approved: false })}>Decline</button>
      </div>
    </div>}
    {paused && view?.enabled && <div className="computer-card">
      <label htmlFor={instructionId}>Update the task</label>
      <textarea id={instructionId} rows={3} value={instruction} onChange={e => setInstruction(e.target.value)} placeholder="Make changes in the app, then tell Rune how to continue…" />
      <button disabled={busy || view.uncertainAction} onClick={() => void perform(async () => {
        const computer = await fetchComputer(sessionId);
        await controlComputer(computer, 'resume', instruction);
        setInstruction('');
      })}>Resume Rune</button>
    </div>}
  </section>;
}
