import { useRef, type ReactNode } from 'react';
import { useFocusTrap } from '../hooks/useFocusTrap';
import './SettingsDialog.css';

export function SettingsDialog({ children, onClose, covered = false, returnFocus }: {
  children: ReactNode;
  onClose: () => void;
  covered?: boolean;
  returnFocus?: HTMLElement | null;
}) {
  // Keep the original trigger while a child panel covers Settings.
  const trigger = useRef(returnFocus ?? (typeof document === 'undefined' ? null : document.activeElement as HTMLElement | null));
  const ref = useFocusTrap<HTMLDivElement>(!covered, trigger.current);
  return <div className="settings-overlay" inert={covered} aria-hidden={covered || undefined}
    onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div ref={ref} className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title"
      onKeyDown={event => {
        if (event.key === 'Escape' && !event.defaultPrevented) {
          event.preventDefault();
          event.stopPropagation();
          onClose();
        }
      }}>
      <header className="settings-dialog-header">
        <div><h1 id="settings-title">Settings</h1><p>Connections, preferences, and usage limits.</p></div>
        <button type="button" className="toolbar-icon" aria-label="Close settings" onClick={onClose}>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m4 4 10 10M14 4 4 14" /></svg>
        </button>
      </header>
      <div className="settings-dialog-body">{children}</div>
    </div>
  </div>;
}
