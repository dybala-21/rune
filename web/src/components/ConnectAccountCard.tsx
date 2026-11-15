import { useRef, useState } from 'react';
import { CONNECT_COPY } from '../shared/ui-copy';

/**
 * Asks the owner to connect an account (Gmail, Calendar) for the task that is waiting.
 *
 * The button opens their approval page, where they sign in to the account; this app never sees
 * it. The run continues by itself once the account is connected, and the card closes.
 */
export function ConnectAccountCard({
  url,
  label,
  detail,
  onDecline,
}: {
  url: string;
  label: string;
  detail: string;
  onDecline: () => Promise<void>;
}) {
  const [opened, setOpened] = useState(false);
  const [declining, setDeclining] = useState(false);
  const inFlight = useRef(false);
  const usable = url.startsWith('https://');

  const decline = async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setDeclining(true);
    try {
      await onDecline();
    } catch {
      // The hook reports the error. Unlock the card to try again.
      setDeclining(false);
    } finally {
      inFlight.current = false;
    }
  };

  return (
    <div className="slide-up" style={{
      margin: '8px 0',
      padding: '14px 16px',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-secondary)',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      position: 'relative',
      zIndex: 20,
    }}>
      <div style={{ fontWeight: 600, fontSize: 14 }}>{label}</div>
      <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        {detail} {opened ? CONNECT_COPY.waiting : CONNECT_COPY.hint}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        {usable && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setOpened(true)}
            style={{
              flex: 1,
              padding: '10px 16px',
              background: 'var(--accent)',
              color: 'var(--bg-primary)',
              borderRadius: 'var(--radius-md)',
              fontWeight: 600,
              fontSize: 14,
              textAlign: 'center',
              textDecoration: 'none',
            }}
          >
            {opened ? CONNECT_COPY.again : label}
          </a>
        )}
        <button
          onClick={decline}
          disabled={declining}
          style={{
            padding: '10px 16px',
            background: 'transparent',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            fontWeight: 600,
            fontSize: 13,
            opacity: declining ? 0.5 : 1,
            cursor: declining ? 'default' : 'pointer',
          }}
        >
          {CONNECT_COPY.notNow}
        </button>
      </div>
    </div>
  );
}
