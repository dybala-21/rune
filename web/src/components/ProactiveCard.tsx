import { useState } from 'react';
import { respondToSuggestion } from '../api';
import type { ProactiveSuggestion } from '../types';
import { proactiveStatus } from '../utils/proactive';
import { SparkIcon } from './icons';

interface ProactiveCardProps {
  suggestion: ProactiveSuggestion;
}

export function ProactiveCard({ suggestion }: ProactiveCardProps) {
  const [acknowledged, setAcknowledged] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const state = suggestion.state;
  // Keep a just-accepted response while an earlier feed request finishes.
  const response = proactiveStatus(state && acknowledged && ['pending', 'unconfirmed'].includes(state.response)
    ? { ...state, response: acknowledged } : state);
  const output = state?.result.output || state?.result.error || '';
  const isIntervene = suggestion.intensity === 'intervene';
  const accentColor = isIntervene ? '#61AFEF' : '#56B6C2';
  const timeAgo = formatTimeAgo(suggestion.timestamp);

  async function respond(value: 'accept' | 'dismiss') {
    if (busy || response) return;
    setBusy(true);
    setError(undefined);
    try {
      await respondToSuggestion(suggestion.id, value);
      setAcknowledged(value === 'accept' ? 'accepted' : 'dismissed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not record your response');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="slide-up" style={{
      padding: '10px 0',
    }}>
      <div style={{
        borderLeft: `3px solid ${accentColor}`,
        borderRadius: 'var(--radius-md)',
        background: 'var(--bg-secondary)',
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <span style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            fontSize: 13,
            fontWeight: 600,
            color: accentColor,
          }}>
            <SparkIcon size={12} />
            rune
          </span>
          <span style={{
            fontSize: 11,
            color: 'var(--text-muted)',
          }}>
            {timeAgo}
          </span>
        </div>
        {suggestion.headline && <strong style={{ fontSize: 14 }}>{suggestion.headline}</strong>}

        <div style={{
          fontSize: 14,
          lineHeight: 1.6,
          color: 'var(--text-primary)',
        }}>
          {suggestion.body}
        </div>
        <div role="status" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {response || <div style={{ display: 'flex', gap: 8 }}>
            <button disabled={busy} onClick={() => void respond('accept')}>Run once</button>
            <button disabled={busy} onClick={() => void respond('dismiss')}>Dismiss</button>
          </div>}
          {error && <p role="alert">{error}</p>}
          {output && <p style={{ whiteSpace: 'pre-wrap', color: 'var(--text-primary)' }}>{output}</p>}
        </div>
      </div>
    </div>
  );
}

function formatTimeAgo(ts: number): string {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
