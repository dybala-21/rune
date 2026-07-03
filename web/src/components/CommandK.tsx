import { useEffect, useMemo, useRef, useState } from 'react';
import { useFocusTrap } from '../hooks/useFocusTrap';

export interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

interface CommandKProps {
  open: boolean;
  commands: Command[];
  onClose: () => void;
}

/** ⌘K action palette: filter, arrow-navigate, Enter to run. App owns the actions. */
export function CommandK({ open, commands, onClose }: CommandKProps) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const trapRef = useFocusTrap<HTMLDivElement>(open);

  useEffect(() => {
    if (open) {
      setQuery('');
      setActive(0);
      // focus after the element mounts
      const id = requestAnimationFrame(() => inputRef.current?.focus());
      return () => cancelAnimationFrame(id);
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(c => c.label.toLowerCase().includes(q) || c.hint?.toLowerCase().includes(q));
  }, [query, commands]);

  useEffect(() => {
    if (active >= filtered.length) setActive(filtered.length > 0 ? filtered.length - 1 : 0);
  }, [filtered, active]);

  if (!open) return null;

  const runAt = (i: number) => {
    const cmd = filtered[i];
    if (cmd) {
      onClose();
      cmd.run();
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(a => Math.min(a + 1, filtered.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(a => Math.max(a - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); runAt(active); }
    else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '14vh',
        background: 'rgba(0, 0, 0, 0.45)',
      }}
    >
      <div
        ref={trapRef}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={e => e.stopPropagation()}
        className="glass fade-scale"
        style={{
          width: 'min(560px, 92vw)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          boxShadow: 'var(--shadow-lg)',
          overflow: 'hidden',
        }}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={e => { setQuery(e.target.value); setActive(0); }}
          onKeyDown={onKeyDown}
          placeholder="Type a command…"
          aria-label="Command palette search"
          style={{
            width: '100%',
            padding: '14px 16px',
            background: 'transparent',
            border: 'none',
            borderBottom: '1px solid var(--border)',
            borderRadius: 0,
            color: 'var(--text-primary)',
            fontSize: 15,
            outline: 'none',
          }}
        />
        <div style={{ maxHeight: 320, overflowY: 'auto', padding: 6 }}>
          {filtered.length === 0 ? (
            <div style={{ padding: '18px 12px', color: 'var(--text-muted)', fontSize: 13, textAlign: 'center' }}>
              No matching command
            </div>
          ) : (
            filtered.map((cmd, i) => (
              <button
                key={cmd.id}
                onClick={() => runAt(i)}
                onMouseEnter={() => setActive(i)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  width: '100%',
                  padding: '9px 12px',
                  background: i === active ? 'var(--accent-subtle)' : 'transparent',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  color: 'var(--text-primary)',
                  fontSize: 14,
                  textAlign: 'left',
                  cursor: 'pointer',
                }}
              >
                <span style={{ flex: 1 }}>{cmd.label}</span>
                {cmd.hint && (
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--text-muted)',
                  }}>
                    {cmd.hint}
                  </span>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default CommandK;
