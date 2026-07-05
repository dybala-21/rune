import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchWorkspace, fetchWorkspaceRecents, setWorkspace } from '../api';
import { DirectoryCombobox } from './DirectoryCombobox';

function basename(p: string): string {
  const parts = p.replace(/\/+$/, '').split('/');
  return parts[parts.length - 1] || p;
}

/**
 * Titlebar chip showing the conversation's pinned workspace.
 * Click → recents + manual path entry. The agent runs in this directory;
 * the workbench Diff/Files tabs read from it.
 */
export function WorkspaceChip() {
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
    // The inline picker (or another tab) may pin the workspace elsewhere.
    window.addEventListener('rune:workspace-changed', refresh);
    return () => window.removeEventListener('rune:workspace-changed', refresh);
  }, [refresh]);

  useEffect(() => {
    if (!open) return;
    fetchWorkspaceRecents()
      .then(r => setRecents(r.paths))
      .catch(() => setRecents([]));
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
      // Let the inline picker and any other listeners refresh.
      window.dispatchEvent(new CustomEvent('rune:workspace-changed'));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not set workspace');
    }
  };

  return (
    <div ref={rootRef} style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        title={path ? `Workspace: ${path}` : 'Set the folder the agent works in'}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'var(--bg-tertiary)',
          border: `1px solid ${path ? 'var(--border-strong, var(--border))' : 'var(--border)'}`,
          borderRadius: 99, padding: '3px 12px',
          fontSize: 11.5, fontFamily: 'var(--font-mono)',
          color: path ? 'var(--text-primary)' : 'var(--text-muted)',
          cursor: 'pointer', maxWidth: 220,
        }}
      >
        <span aria-hidden="true">📁</span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {path ? basename(path) : 'Workspace'}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 9 }}>▾</span>
      </button>

      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', right: 0, zIndex: 60,
          width: 340, background: 'var(--bg-secondary)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-lg)', padding: 10,
        }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
            Agent works in this folder. Type to browse subfolders.
          </div>
          <DirectoryCombobox recents={recents} onChoose={pick} />
          {path && (
            <div style={{
              marginTop: 8, fontSize: 11, color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', overflow: 'hidden',
              textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              Current: {path}
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
