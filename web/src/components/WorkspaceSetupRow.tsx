import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchWorkspace, fetchWorkspaceRecents, setWorkspace } from '../api';
import { DirectoryCombobox } from './DirectoryCombobox';
import { FolderIcon } from './icons';

function basename(p: string): string {
  const parts = p.replace(/\/+$/, '').split('/');
  return parts[parts.length - 1] || p;
}

export function WorkspaceSetupRow() {
  const [path, setPath] = useState('');
  const [open, setOpen] = useState(false);
  const [recents, setRecents] = useState<string[]>([]);
  const [error, setError] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(() => {
    fetchWorkspace().then(r => setPath(r.path)).catch(() => setPath(''));
  }, []);

  useEffect(() => {
    refresh();
    // The chip and the inline picker set the same folder.
    window.addEventListener('rune:workspace-changed', refresh);
    return () => window.removeEventListener('rune:workspace-changed', refresh);
  }, [refresh]);

  useEffect(() => {
    if (!open) return;
    fetchWorkspaceRecents().then(r => setRecents(r.paths)).catch(() => setRecents([]));
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const pick = async (p: string) => {
    setError('');
    try {
      const r = await setWorkspace(p);
      setPath(r.path);
      setOpen(false);
      window.dispatchEvent(new CustomEvent('rune:workspace-changed'));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't set the workspace");
    }
  };

  return (
    <div ref={rootRef}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        title={path || 'Choose the folder to work in'}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          width: '100%',
          padding: '11px 4px',
          background: 'none',
          border: 'none',
          color: path ? 'var(--text-secondary)' : 'var(--text-primary)',
          fontSize: 14,
          textAlign: 'left',
          cursor: 'pointer',
          borderRadius: 0,
        }}
        aria-expanded={open}
      >
        <span style={{ color: 'var(--accent)', display: 'flex' }}>
          <FolderIcon size={16} />
        </span>
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {path ? `Working folder · ${basename(path)}` : 'Working folder'}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          {path ? 'Change' : 'Choose'}
        </span>
      </button>

      {open && (
        <div style={{ padding: '4px 4px 10px' }}>
          <DirectoryCombobox recents={recents} onChoose={pick} />
          {path && (
            <div style={{
              marginTop: 8, fontSize: 11, color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', overflow: 'hidden',
              textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              Current folder: {path}
            </div>
          )}
          {error && (
            <div style={{ color: 'var(--danger)', fontSize: 11.5, marginTop: 6 }}>{error}</div>
          )}
        </div>
      )}
    </div>
  );
}
