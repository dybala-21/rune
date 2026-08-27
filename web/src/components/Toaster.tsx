import { useEffect, useState } from 'react';
import { subscribe, dismissToast, type Toast } from '../utils/toast';

// Bottom-center stack of transient notifications. Fed by the toast bus, so any
// module can raise one. Click to dismiss early; each auto-expires on its ttl.

const GLYPH: Record<Toast['kind'], string> = {
  info: '◆', success: '✓', error: '✗',
};

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  useEffect(() => subscribe(setToasts), []);

  if (!toasts.length) return null;
  return (
    <div className="toaster" role="status" aria-live="polite">
      {toasts.map(t => (
        <button
          key={t.id}
          className={`toast toast-${t.kind}`}
          onClick={() => dismissToast(t.id)}
          title="Dismiss"
        >
          <span className="toast-glyph" aria-hidden="true">{GLYPH[t.kind]}</span>
          <span className="toast-msg">{t.message}</span>
        </button>
      ))}
    </div>
  );
}
