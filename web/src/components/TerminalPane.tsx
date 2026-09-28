import { useEffect, useRef, useState } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { fetchTerminalStatus, mintTerminalToken } from '../api';

type Phase = 'checking' | 'disabled' | 'idle' | 'connecting' | 'connected' | 'closed';

/** The shell lives until it exits, is explicitly ended, or the conversation closes. */
export function TerminalPane({ active = true, onConnectionChange }: { active?: boolean; onConnectionChange?: (connected: boolean) => void }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const generation = useRef(0);
  const [phase, setPhase] = useState<Phase>('checking');
  const [error, setError] = useState('');
  useEffect(() => { onConnectionChange?.(phase === 'connected' || phase === 'connecting'); }, [phase, onConnectionChange]);

  const disconnect = () => {
    generation.current++;
    const ws = wsRef.current;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      ws.close();
    }
    wsRef.current = null;
    termRef.current?.dispose();
    termRef.current = null;
    fitRef.current = null;
  };

  useEffect(() => {
    let live = true;
    fetchTerminalStatus()
      .then(r => { if (live) setPhase(r.enabled ? 'idle' : 'disabled'); })
      .catch(e => { if (live) { setPhase('idle'); setError(String(e)); } });
    return () => { live = false; disconnect(); };
  }, []);

  useEffect(() => {
    if (!active || !hostRef.current) return;
    const fit = () => {
      if (hostRef.current?.clientWidth && hostRef.current.clientHeight) fitRef.current?.fit();
    };
    const observer = new ResizeObserver(fit);
    observer.observe(hostRef.current);
    fit();
    return () => observer.disconnect();
  }, [active, phase]);

  const connect = async () => {
    if (!hostRef.current || phase === 'connecting' || phase === 'connected') return;
    disconnect();
    const version = generation.current;
    setError('');
    setPhase('connecting');
    try {
      const { token } = await mintTerminalToken();
      if (version !== generation.current || !hostRef.current) return;
      const term = new Terminal({
        fontSize: 12.5, fontFamily: 'ui-monospace, monospace', cursorBlink: true,
        theme: { background: '#0E1116', foreground: '#E8EDF2', cursor: '#7DD3E8' },
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      term.open(hostRef.current);
      termRef.current = term;
      fitRef.current = fit;
      if (hostRef.current.clientWidth && hostRef.current.clientHeight) fit.fit();
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws/terminal?token=${encodeURIComponent(token)}`);
      wsRef.current = ws;
      ws.onopen = () => {
        setPhase('connected');
        ws.send(JSON.stringify(['set_size', term.rows, term.cols]));
        term.onData(data => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(['stdin', data])));
        term.onResize(({ rows, cols }) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(['set_size', rows, cols])));
      };
      ws.onmessage = event => {
        try {
          const message = JSON.parse(event.data);
          if (Array.isArray(message) && message[0] === 'stdout') term.write(message[1]);
          else if (Array.isArray(message) && message[0] === 'disconnect') {
            term.write('\r\n[process exited]\r\n');
            setPhase('closed');
          }
        } catch (error) { console.debug('Invalid terminal message', error); }
      };
      ws.onerror = () => { setError('Terminal connection failed.'); setPhase('closed'); };
      ws.onclose = () => setPhase('closed');
    } catch (error) {
      if (version !== generation.current) return;
      disconnect();
      setError(error instanceof Error ? error.message : 'Could not open the shell.');
      setPhase('idle');
    }
  };

  if (phase === 'checking') {
    return <Centered>Checking terminal availability…</Centered>;
  }
  if (phase === 'disabled') {
    return (
      <Centered>
        <div style={{ maxWidth: 360, textAlign: 'center' }}>
          <div style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 6 }}>
            Terminal is off
          </div>
          <div style={{ fontSize: 12.5, lineHeight: 1.6 }}>
            An embedded shell is a powerful capability, so it ships disabled.
            Enable it by starting the daemon with{' '}
            <code style={{ color: 'var(--accent)' }}>RUNE_TERMINAL_ENABLED=1</code>.
          </div>
        </div>
      </Centered>
    );
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {(phase === 'connected' || phase === 'connecting') && <div className="workbench-view-toolbar"><span>{phase === 'connecting' ? 'Connecting…' : 'Shell running'}</span><button type="button" onClick={() => { disconnect(); setPhase('closed'); }}>End shell</button></div>}
      {phase === 'idle' || phase === 'closed' ? (
        <div style={{ padding: 14 }}>
          <button
            type="button"
            onClick={connect}
            style={{
              background: 'var(--accent)', color: '#0A1319', border: 'none',
              borderRadius: 'var(--radius-sm)', padding: '8px 16px',
              fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
            }}
          >
            {phase === 'closed' ? 'Start a new shell' : 'Open a shell here'}
          </button>
          <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--text-muted)' }}>
            Runs in this conversation's workspace.
          </div>
          {error && <div style={{ color: 'var(--danger)', fontSize: 11.5, marginTop: 8 }}>{error}</div>}
        </div>
      ) : null}
      <div
        ref={hostRef}
        style={{
          flex: 1, minHeight: 0, padding: phase === 'connected' || phase === 'connecting' ? 8 : 0,
          display: phase === 'connected' || phase === 'connecting' || phase === 'closed' ? 'block' : 'none',
        }}
      />
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 20, color: 'var(--text-muted)', fontSize: 12.5,
    }}>
      {children}
    </div>
  );
}
