import { useState, useEffect, useRef, useCallback } from 'react';

interface Command {
  name: string;
  description: string;
  usage: string;
  aliases: string[];
}

interface Props {
  filter: string;
  onSelect: (command: string) => void;
  onClose: () => void;
}

export function CommandPalette({ filter, onSelect, onClose }: Props) {
  const [commands, setCommands] = useState<Command[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('/api/v1/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: 'commands.list', params: {} }),
    })
      .then(r => r.json())
      .then(res => {
        if (res.success && Array.isArray(res.data)) {
          setCommands(res.data);
        }
      })
      .catch(() => {});
  }, []);

  const filtered = commands.filter(c => {
    const q = filter.toLowerCase();
    return (
      c.name.toLowerCase().includes(q) ||
      c.aliases.some(a => a.toLowerCase().includes(q)) ||
      c.description.toLowerCase().includes(q)
    );
  });

  useEffect(() => {
    setSelectedIdx(0);
  }, [filter]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIdx(i => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIdx(i => Math.max(i - 1, 0));
      } else if (e.key === 'Enter' && filtered.length > 0) {
        e.preventDefault();
        onSelect(filtered[selectedIdx].name);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    },
    [filtered, selectedIdx, onSelect, onClose]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Scroll selected into view
  useEffect(() => {
    const el = listRef.current?.children[selectedIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [selectedIdx]);

  if (filtered.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        bottom: '100%',
        left: 0,
        right: 0,
        maxHeight: '240px',
        overflowY: 'auto',
        background: '#1a1a2e',
        border: '1px solid #333',
        borderRadius: '8px',
        marginBottom: '4px',
        zIndex: 100,
      }}
      ref={listRef}
    >
      {filtered.map((cmd, i) => (
        <div
          key={cmd.name}
          onClick={() => onSelect(cmd.name)}
          style={{
            padding: '8px 12px',
            cursor: 'pointer',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: i === selectedIdx ? '#2a2a4a' : 'transparent',
            borderBottom: i < filtered.length - 1 ? '1px solid #222' : 'none',
          }}
        >
          <span style={{ color: '#7dd3fc', fontFamily: 'monospace', fontSize: '13px' }}>
            {cmd.name}
          </span>
          <span style={{ color: '#888', fontSize: '12px', marginLeft: '12px' }}>
            {cmd.description}
          </span>
        </div>
      ))}
    </div>
  );
}
