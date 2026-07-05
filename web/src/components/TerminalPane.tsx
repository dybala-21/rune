import { useEffect, useRef, useState } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { fetchTerminalStatus, mintTerminalToken } from '../api';

type Phase = 'checking' | 'disabled' | 'idle' | 'connecting' | 'connected' | 'closed';

/**
 * Embedded shell tab. Off unless the daemon has the terminal capability
 * enabled; then it mints a one-shot token and opens the PTY WebSocket.
 * Runs in the conversation's workspace.
 */
export function TerminalPane() {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [phase, setPhase] = useState<Phase>('checking');
  const [error, setError] = useState('');

  useEffect(() => {
    let live = true;
    fetchTerminalStatus()
      .then(r => { if (live) setPhase(r.enabled ? 'idle' : 'disabled'); })
      .catch(() => { if (live) setPhase('disabled'); });
    return () => { live = false; };
  }, []);

  const connect = async () => {
    if (!hostRef.current) return;
    setError('');
    setPhase('connecting');
    let token: string;
    try {
      const r = await mintTerminalToken();
      token = r.token;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not get a terminal token');
      setPhase('idle');
      return;
    }

    const term = new Terminal({
      fontSize: 12.5,
      fontFamily: 'var(--font-mono), ui-monospace, monospace',
      cursorBlink: true,
      theme: { background: '#0E1116', foreground: '#E8EDF2', cursor: '#7DD3E8' },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(hostRef.current);
    fit.fit();
    termRef.current = term;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/terminal?token=${encodeURIComponent(token)}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setPhase('connected');
      const send = () => ws.readyState === WebSocket.OPEN
        && ws.send(JSON.stringify(['set_size', term.rows, term.cols]));
      send();
      term.onData(d => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(['stdin', d])));
      term.onResize(({ rows, cols }) =>
        ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(['set_size', rows, cols])));
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (Array.isArray(msg) && msg[0] === 'stdout') term.write(msg[1]);
        else if (Array.isArray(msg) && msg[0] === 'disconnect') { term.write('\r\n[process exited]\r\n'); setPhase('closed'); }
      } catch { /* ignore */ }
    };
    ws.onerror = () => {
      setError('Terminal connection failed.');
      // Fall back so the retry button + message reappear instead of a blank
      // pane stuck on "connecting".
      setPhase(p => (p === 'connected' ? 'closed' : 'idle'));
    };
    ws.onclose = () => setPhase(p => (p === 'connected' ? 'closed' : 'idle'));

    const onWinResize = () => { try { fit.fit(); } catch { /* not attached */ } };
    window.addEventListener('resize', onWinResize);
    (term as unknown as { _cleanup?: () => void })._cleanup = () =>
      window.removeEventListener('resize', onWinResize);
  };

  useEffect(() => () => {
    wsRef.current?.close();
    const t = termRef.current as unknown as { _cleanup?: () => void } | null;
    t?._cleanup?.();
    termRef.current?.dispose();
  }, []);

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
