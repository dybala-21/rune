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

const ITEM_STYLE = {
  padding: '8px 12px',
  cursor: 'pointer',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  borderBottom: '1px solid #222',
};

async function rpc(method: string, params: Record<string, unknown> = {}) {
  const res = await fetch('/api/v1/rpc', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ method, params }),
  });
  const data = await res.json();
  return data.success ? data.data : null;
}

export function CommandPalette({ filter, onSelect, onClose }: Props) {
  const [commands, setCommands] = useState<Command[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [subMenu, setSubMenu] = useState<'none' | 'provider' | 'model'>('none');
  const [providers, setProviders] = useState<Record<string, string[]>>({});
  const [selectedProvider, setSelectedProvider] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  // Load commands
  useEffect(() => {
    rpc('commands.list').then(data => {
      if (Array.isArray(data)) setCommands(data);
    }).catch(() => {
      setCommands([
        { name: '/help', description: 'Show available commands', usage: '', aliases: [] },
        { name: '/model', description: 'Switch LLM model', usage: '', aliases: [] },
        { name: '/memory', description: 'Memory management', usage: '', aliases: [] },
        { name: '/clear', description: 'Clear conversation', usage: '', aliases: [] },
      ]);
    });
  }, []);

  const filtered = subMenu === 'none'
    ? commands.filter(c => {
        const q = filter.toLowerCase();
        return c.name.includes(q) || c.aliases.some(a => a.includes(q));
      })
    : [];

  const providerList = subMenu === 'provider' ? Object.keys(providers) : [];
  const modelList = subMenu === 'model' ? (providers[selectedProvider] || []) : [];
  const currentItems = subMenu === 'none' ? filtered : subMenu === 'provider' ? providerList : modelList;

  useEffect(() => { setSelectedIdx(0); }, [filter, subMenu]);

  const handleSelect = useCallback((idx: number) => {
    if (subMenu === 'none') {
      const cmd = filtered[idx];
      if (!cmd) return;
      if (cmd.name === '/model') {
        // Open provider submenu
        rpc('models.list').then(data => {
          if (data && typeof data === 'object') {
            setProviders(data as Record<string, string[]>);
            setSubMenu('provider');
          }
        });
        return;
      }
      onSelect(cmd.name);
    } else if (subMenu === 'provider') {
      setSelectedProvider(providerList[idx]);
      setSubMenu('model');
    } else if (subMenu === 'model') {
      const model = modelList[idx];
      // Apply model change via config.patch
      rpc('config.patch', { activeModel: { provider: selectedProvider, model } });
      onSelect('');  // close palette, don't insert text
      onClose();
    }
  }, [subMenu, filtered, providerList, modelList, selectedProvider, onSelect, onClose]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const len = currentItems.length;
      if (!len) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedIdx(i => Math.min(i + 1, len - 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedIdx(i => Math.max(i - 1, 0)); }
      else if (e.key === 'Enter') { e.preventDefault(); handleSelect(selectedIdx); }
      else if (e.key === 'Escape') {
        e.preventDefault();
        if (subMenu === 'model') setSubMenu('provider');
        else if (subMenu === 'provider') setSubMenu('none');
        else onClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentItems, selectedIdx, subMenu, handleSelect, onClose]);

  useEffect(() => {
    const el = listRef.current?.children[selectedIdx] as HTMLElement | undefined;
    el?.scrollIntoView({ block: 'nearest' });
  }, [selectedIdx]);

  if (currentItems.length === 0 && subMenu === 'none' && filter) return null;

  const title = subMenu === 'provider' ? 'Select provider' : subMenu === 'model' ? `${selectedProvider} models` : '';

  return (
    <div
      style={{
        position: 'absolute', bottom: '100%', left: 0, right: 0,
        maxHeight: '280px', overflowY: 'auto',
        background: '#1a1a2e', border: '1px solid #333',
        borderRadius: '8px', marginBottom: '4px', zIndex: 100,
      }}
      ref={listRef}
    >
      {title && (
        <div style={{ padding: '6px 12px', color: '#666', fontSize: '11px', borderBottom: '1px solid #333' }}>
          {title}
        </div>
      )}
      {subMenu === 'none' && filtered.map((cmd, i) => (
        <div
          key={cmd.name}
          onClick={() => handleSelect(i)}
          style={{ ...ITEM_STYLE, background: i === selectedIdx ? '#2a2a4a' : 'transparent' }}
        >
          <span style={{ color: '#7dd3fc', fontFamily: 'monospace', fontSize: '13px' }}>{cmd.name}</span>
          <span style={{ color: '#888', fontSize: '12px', marginLeft: '12px' }}>{cmd.description}</span>
        </div>
      ))}
      {subMenu === 'provider' && providerList.map((prov, i) => (
        <div
          key={prov}
          onClick={() => handleSelect(i)}
          style={{ ...ITEM_STYLE, background: i === selectedIdx ? '#2a2a4a' : 'transparent' }}
        >
          <span style={{ color: '#c4b5fd', fontFamily: 'monospace', fontSize: '13px' }}>{prov}</span>
          <span style={{ color: '#666', fontSize: '12px' }}>{(providers[prov] || []).length} models</span>
        </div>
      ))}
      {subMenu === 'model' && modelList.map((model, i) => (
        <div
          key={model}
          onClick={() => handleSelect(i)}
          style={{ ...ITEM_STYLE, background: i === selectedIdx ? '#2a2a4a' : 'transparent' }}
        >
          <span style={{ color: '#7dd3fc', fontFamily: 'monospace', fontSize: '13px' }}>{model}</span>
        </div>
      ))}
    </div>
  );
}
