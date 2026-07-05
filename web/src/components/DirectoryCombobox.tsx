import { useCallback, useEffect, useRef, useState } from 'react';
import { listWorkspaceDirs } from '../api';

interface DirectoryComboboxProps {
  /** Recent workspaces shown when the field is empty. */
  recents?: string[];
  /** Commit a chosen absolute path. */
  onChoose: (path: string) => void;
  autoFocus?: boolean;
  placeholder?: string;
}

function dirname(p: string): string {
  const t = p.replace(/\/+$/, '');
  const i = t.lastIndexOf('/');
  return i <= 0 ? '/' : t.slice(0, i);
}
function basename(p: string): string {
  const parts = p.replace(/\/+$/, '').split('/');
  return parts[parts.length - 1] || p;
}

/**
 * A path field that lists matching subdirectories as you type. Typing filters
 * the current directory's children; clicking a row (or → ) drills in; Enter
 * accepts the typed path. Arrow keys move the highlight.
 */
export function DirectoryCombobox({ recents = [], onChoose, autoFocus, placeholder }: DirectoryComboboxProps) {
  const [value, setValue] = useState('');
  const [entries, setEntries] = useState<string[]>([]);
  const [listedDir, setListedDir] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Which directory to list, and the prefix to filter by, from the raw input.
  const endsWithSep = value.endsWith('/');
  const listDir = value === '' ? '' : endsWithSep ? value : dirname(value);
  const filter = value === '' || endsWithSep ? '' : basename(value);

  const load = useCallback((dir: string) => {
    listWorkspaceDirs(dir || undefined)
      .then(r => { setEntries(r.entries); setListedDir(r.dir); })
      .catch(() => { setEntries([]); });
  }, []);

  useEffect(() => {
    if (!open) return;
    if (value === '') { setEntries([]); return; }
    load(listDir);
    setActive(0);
  }, [open, listDir, value, load]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const showRecents = open && value === '' && recents.length > 0;
  const matches = value === ''
    ? []
    : entries.filter(n => n.toLowerCase().startsWith(filter.toLowerCase()));
  const rows: { label: string; path: string; kind: 'recent' | 'dir' }[] = showRecents
    ? recents.map(p => ({ label: p, path: p, kind: 'recent' as const }))
    : matches.map(n => ({ label: n, path: `${listedDir.replace(/\/$/, '')}/${n}`, kind: 'dir' as const }));

  const drillInto = (path: string) => {
    setValue(path + '/');
    inputRef.current?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open) setOpen(true);
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(a => Math.min(a + 1, rows.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(a => Math.max(a - 1, 0)); }
    else if (e.key === 'Escape') { setOpen(false); }
    else if (e.key === 'Tab' && rows[active]?.kind === 'dir') { e.preventDefault(); drillInto(rows[active].path); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      const row = rows[active];
      // Enter on a recent or a highlighted dir accepts it; otherwise accept
      // whatever path is typed.
      if (row && (showRecents || filter === '')) onChoose(row.path);
      else if (value.trim()) onChoose(value.trim());
    }
  };

  return (
    <div ref={rootRef} style={{ position: 'relative', flex: 1 }}>
      <input
        ref={inputRef}
        value={value}
        autoFocus={autoFocus}
        onChange={e => { setValue(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        placeholder={placeholder ?? '/path/to/project — type to browse'}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        style={{
          width: '100%', background: 'var(--bg-primary)', color: 'var(--text-primary)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
          padding: '7px 9px', fontSize: 12.5, fontFamily: 'var(--font-mono)',
        }}
      />
      {open && rows.length > 0 && (
        <div
          role="listbox"
          style={{
            position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0, zIndex: 70,
            maxHeight: 240, overflowY: 'auto', background: 'var(--bg-secondary)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-lg)', padding: 4,
          }}
        >
          {rows.map((row, i) => (
            <div
              key={row.path}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={e => {
                // mousedown (not click) so it fires before the input blur.
                e.preventDefault();
                if (row.kind === 'dir') drillInto(row.path);
                else onChoose(row.path);
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '6px 9px',
                borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontSize: 12.5,
                background: i === active ? 'var(--bg-tertiary)' : 'transparent',
                color: 'var(--text-primary)',
              }}
            >
              <span aria-hidden="true" style={{ opacity: 0.7 }}>📁</span>
              <span style={{
                fontFamily: row.kind === 'recent' ? 'var(--font-mono)' : 'var(--font-sans)',
                fontSize: row.kind === 'recent' ? 11.5 : 12.5,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>{row.label}</span>
              {row.kind === 'dir' && (
                <button
                  type="button"
                  title="Choose this folder"
                  onMouseDown={e => { e.preventDefault(); e.stopPropagation(); onChoose(row.path); }}
                  style={{
                    marginLeft: 'auto', background: 'var(--accent)', color: '#0A1319',
                    border: 'none', borderRadius: 'var(--radius-sm)', padding: '2px 8px',
                    fontSize: 11, fontWeight: 600, cursor: 'pointer',
                  }}
                >Use</button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
