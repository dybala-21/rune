import type { AuthFailure } from '../api';
import { isLocalSession } from '../utils/sessionContext';
import { useEffect, useState } from 'react';
import { MACHINE_EVENT, machineStatus } from '../utils/presence';

export function ConnectionBanner({ authFailure = null, onRetry }: { authFailure?: AuthFailure; onRetry?: () => void }) {
  const [machine, setMachine] = useState(machineStatus);
  useEffect(() => {
    const onMachine = (event: Event) => setMachine((event as CustomEvent<string>).detail);
    window.addEventListener(MACHINE_EVENT, onMachine);
    return () => window.removeEventListener(MACHINE_EVENT, onMachine);
  }, []);
  if (authFailure) return <div role="status" className="fade-in" style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 12, padding: '12px 16px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
    <span>{authFailure === 'required'
      ? isLocalSession() ? 'Authentication is required. Open Rune through your signed-in gateway.' : 'Sign in again to reconnect to your work.'
      : 'Access was denied. Check your server permissions.'}</span>
    {authFailure === 'required' && !isLocalSession() && <a href="/">Sign in</a>}
    {onRetry && <button type="button" onClick={onRetry}>Try again</button>}
  </div>;
  const local = machine === null;
  const tone = local ? 'var(--danger)' : 'var(--text-secondary)';
  return <div className="fade-in" style={{
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '7px 16px',
    background: local ? 'var(--danger-subtle)' : 'var(--bg-secondary)',
    borderBottom: '1px solid var(--border)',
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    color: tone,
  }}>
    <span className="spinner" style={{ width: 11, height: 11, borderTopColor: tone }} />
    {local ? <>
      <span>Engine unreachable — reconnecting…</span>
      <span style={{ color: 'var(--text-muted)', marginLeft: 'auto' }}>
        start it with: rune daemon start
      </span>
    </> : <span>{hostedMessage(machine)}</span>}
  </div>;
}

export function hostedMessage(machine: string): string {
  if (machine === 'suspended') return 'Rune is asleep — it wakes up when you use this page';
  if (machine === 'running') return 'Reconnecting…';
  if (machine === 'starting' || machine === 'suspending') return 'Rune is waking up…';
  return 'Your Rune machine is not running — contact the operator';
}
