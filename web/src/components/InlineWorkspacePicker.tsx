import { useEffect, useState } from 'react';
import { fetchWorkspace, fetchWorkspaceRecents, setWorkspace } from '../api';
import { DirectoryCombobox } from './DirectoryCombobox';

/**
 * Shown in the chat stream the first time the agent touches files while no
 * workspace is pinned — asks for the folder in place, rather than up front.
 * Renders nothing once a workspace exists.
 */
export function InlineWorkspacePicker() {
  const [checked, setChecked] = useState(false);
  const [needed, setNeeded] = useState(false);
  const [recents, setRecents] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let live = true;
    fetchWorkspace()
      .then(r => { if (live) { setNeeded(!r.path); setChecked(true); } })
      .catch(() => { if (live) { setNeeded(true); setChecked(true); } });
    fetchWorkspaceRecents().then(r => live && setRecents(r.paths)).catch(() => {});
    return () => { live = false; };
  }, []);

  const pick = async (p: string) => {
    setError('');
    try {
      await setWorkspace(p);
      setNeeded(false);
      window.dispatchEvent(new CustomEvent('rune:workspace-changed'));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not set workspace');
    }
  };

  if (!checked || !needed || dismissed) return null;

  return (
    <div style={{
      alignSelf: 'flex-start',
      maxWidth: 520,
      margin: '4px 0',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--accent-subtle)',
      borderRadius: 'var(--radius-lg)',
      borderBottomLeftRadius: 4,
      padding: '13px 15px',
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline' }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
          Which folder should I work in?
        </div>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          title="Keep using the current folder"
          style={{
            marginLeft: 'auto', background: 'none', border: 'none',
            color: 'var(--text-muted)', fontSize: 11.5, cursor: 'pointer',
          }}
        >
          Use current folder
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 }}>
        I'll read and edit files here. This pins the folder for the rest of the chat.
      </div>
      <DirectoryCombobox recents={recents} onChoose={pick} autoFocus />
      {error && <div style={{ color: 'var(--danger)', fontSize: 11.5, marginTop: 6 }}>{error}</div>}
    </div>
  );
}
