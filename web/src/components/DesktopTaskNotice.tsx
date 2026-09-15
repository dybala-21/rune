import { useState } from 'react';
import { DesktopPane } from './DesktopPane';
import { stopComputer } from '../api';
import './ComputerPane.css';

export function DesktopTaskNotice({ sessionId, runId, kind }: {
  sessionId: string;
  runId?: string;
  kind: 'connection' | 'action' | 'setup';
}) {
  const [expanded, setExpanded] = useState(kind !== 'connection');
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState('');
  const connecting = kind !== 'action';

  const stop = async () => {
    if (!runId || stopping) return;
    setStopping(true);
    try { await stopComputer(runId); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setStopping(false); }
  };

  return <section className="desktop-task-notice" aria-label="Desktop task needs your attention">
    <div role="status">
      <strong>{kind === 'setup' ? 'App access required' : connecting ? 'Waiting for app access' : 'Review the app action'}</strong>
      <p>{kind === 'setup'
        ? 'The task ended before any app input. Enable the required macOS permissions and connect an app below, then retry the request.'
        : connecting
        ? 'Rune is waiting for you to connect an app. Open the settings below, allow the macOS permissions, and select the app for this task. No app input is sent before you connect.'
        : 'This step needs your review. Check the action below and the Rune Computer dialog on your Mac.'}</p>
      {kind === 'connection' && <p>The request can wait for up to 10 minutes. Connecting continues this same task.</p>}
    </div>
    {error && <p role="alert" className="computer-error">{error}</p>}
    <div className="computer-actions">
      <button aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Hide controls' : connecting ? 'Connect an app' : 'Show app controls'}
      </button>
      {!expanded && runId && <button disabled={stopping} onClick={() => void stop()}>Cancel task</button>}
    </div>
    {expanded && <DesktopPane sessionId={sessionId} setupOnMount={connecting} />}
  </section>;
}
