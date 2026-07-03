import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchMarkdownFiles,
  readMarkdownFile,
  writeMarkdownFile,
  type MarkdownFileInfo,
} from '../api';
import { useFocusTrap } from '../hooks/useFocusTrap';

interface MarkdownPanelProps {
  onClose: () => void;
}

export function MarkdownPanel({ onClose }: MarkdownPanelProps) {
  const [files, setFiles] = useState<MarkdownFileInfo[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [content, setContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [listError, setListError] = useState('');
  const trapRef = useFocusTrap<HTMLDivElement>();
  // Guards against a slow read for file A overwriting a later-selected file B.
  const reqIdRef = useRef(0);

  const requestClose = useCallback(() => {
    // Don't discard unsaved edits silently — tell the user why it won't close.
    if (content !== originalContent) {
      setSaveMsg('Save or discard your changes first');
      return;
    }
    onClose();
  }, [content, originalContent, onClose]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') requestClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [requestClose]);

  const loadFiles = useCallback(async () => {
    try {
      const list = await fetchMarkdownFiles();
      setFiles(list);
      setListError('');
    } catch {
      setListError('Could not load the file list.');
    }
  }, []);

  useEffect(() => { void loadFiles(); }, [loadFiles]);

  const handleSelect = async (key: string) => {
    const reqId = ++reqIdRef.current;
    setSelectedKey(key);
    setLoading(true);
    setSaveMsg('');
    setLoadError('');
    setContent('');
    setOriginalContent('');
    try {
      const result = await readMarkdownFile(key);
      if (reqId !== reqIdRef.current) return; // superseded by a newer selection
      setContent(result.content);
      setOriginalContent(result.content);
    } catch {
      if (reqId !== reqIdRef.current) return;
      setLoadError('Could not read this file.');
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!selectedKey) return;
    setSaving(true);
    setSaveMsg('');
    try {
      await writeMarkdownFile(selectedKey, content);
      setOriginalContent(content);
      setSaveMsg('Saved');
      setTimeout(() => setSaveMsg(''), 2000);
    } catch {
      setSaveMsg('Save failed');
    }
    setSaving(false);
  };

  const hasChanges = content !== originalContent;

  const FILE_ICONS: Record<string, string> = {
    heartbeat: '\u2764\uFE0F',
    memory: '\uD83E\uDDE0',
    learned: '\uD83D\uDCA1',
    profile: '\uD83D\uDC64',
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.5)',
    }} onClick={(e) => { if (e.target === e.currentTarget) requestClose(); }}>
      <div ref={trapRef} role="dialog" aria-modal="true" aria-label="Configuration files" className="glass" style={{
        width: '90vw', maxWidth: 900, height: '80vh',
        borderRadius: 'var(--radius-xl)',
        border: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center',
          padding: '16px 20px',
          borderBottom: '1px solid var(--border)',
          gap: 12,
        }}>
          <span style={{ fontSize: 16, fontWeight: 600, flex: 1 }}>
            {'\u270F\uFE0F'} Configuration Files
          </span>
          <button onClick={onClose} aria-label="Close" style={{
            background: 'none', border: 'none',
            color: 'var(--text-muted)', fontSize: 18, cursor: 'pointer',
          }}>{'\u2715'}</button>
        </div>

        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* File list sidebar */}
          <div style={{
            width: 220, borderRight: '1px solid var(--border)',
            overflow: 'auto', flexShrink: 0,
          }}>
            {files.map(f => (
              <button
                key={f.key}
                onClick={() => handleSelect(f.key)}
                style={{
                  width: '100%', textAlign: 'left',
                  padding: '12px 16px',
                  background: selectedKey === f.key ? 'var(--bg-surface)' : 'transparent',
                  border: 'none', borderBottom: '1px solid var(--border)',
                  cursor: 'pointer',
                  color: 'var(--text-primary)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>{FILE_ICONS[f.key] || '\uD83D\uDCC4'}</span>
                  <span style={{ fontWeight: 500, fontSize: 13 }}>{f.label}</span>
                </div>
                <div style={{
                  fontSize: 11, color: 'var(--text-muted)', marginTop: 4,
                }}>
                  {f.description}
                </div>
              </button>
            ))}
            {listError && (
              <div style={{ padding: '16px', fontSize: 12, color: 'var(--danger)' }}>
                {listError}
              </div>
            )}
          </div>

          {/* Editor area */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {!selectedKey ? (
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontSize: 14,
              }}>
                Select a file to edit
              </div>
            ) : loading ? (
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontSize: 13,
              }}>
                Loading…
              </div>
            ) : loadError ? (
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--danger)', fontSize: 13,
              }}>
                {loadError}
              </div>
            ) : (
              <>
                <textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  disabled={saving}
                  spellCheck={false}
                  style={{
                    flex: 1, padding: '16px 20px',
                    background: 'var(--bg-primary)',
                    color: 'var(--text-primary)',
                    border: 'none', outline: 'none', resize: 'none',
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: 13, lineHeight: 1.6,
                  }}
                />
                <div style={{
                  display: 'flex', alignItems: 'center',
                  padding: '10px 20px',
                  borderTop: '1px solid var(--border)',
                  gap: 10,
                }}>
                  {hasChanges && (
                    <span style={{ fontSize: 11, color: 'var(--warning)' }}>
                      Unsaved changes
                    </span>
                  )}
                  <span style={{ flex: 1 }} />
                  {saveMsg && (
                    <span style={{
                      fontSize: 12,
                      color: saveMsg === 'Saved' ? 'var(--success)' : 'var(--danger)',
                    }}>
                      {saveMsg}
                    </span>
                  )}
                  <button
                    onClick={handleSave}
                    disabled={!hasChanges || saving}
                    style={{
                      padding: '6px 16px',
                      background: hasChanges ? 'var(--accent)' : 'var(--bg-tertiary)',
                      color: hasChanges ? 'white' : 'var(--text-muted)',
                      border: 'none', borderRadius: 'var(--radius-md)',
                      fontSize: 12, fontWeight: 600, cursor: hasChanges ? 'pointer' : 'default',
                    }}
                  >
                    {saving ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
