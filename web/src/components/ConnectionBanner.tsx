import { useEffect, useState } from 'react';
import { MACHINE_EVENT, machineStatus } from '../utils/presence';

/** Shown while the event stream is down. Behind the hosted gateway it says what the machine is
 * doing (asleep, waking up) instead of pointing at a local daemon. */
export function ConnectionBanner() {
  const [machine, setMachine] = useState(machineStatus);
  useEffect(() => {
    const onMachine = (event: Event) => setMachine((event as CustomEvent<string>).detail);
    window.addEventListener(MACHINE_EVENT, onMachine);
    return () => window.removeEventListener(MACHINE_EVENT, onMachine);
  }, []);
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
