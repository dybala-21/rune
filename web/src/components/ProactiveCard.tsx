/**
 * ProactiveCard — displays a proactive suggestion from RUNE.
 *
 * Conversational style — no buttons. The user responds naturally
 * in the chat input, and the agent interprets the response.
 */

import type { ProactiveSuggestion } from '../types';

interface ProactiveCardProps {
  suggestion: ProactiveSuggestion;
}

export function ProactiveCard({ suggestion }: ProactiveCardProps) {
  const isNudge = suggestion.intensity === 'nudge';
  const isIntervene = suggestion.intensity === 'intervene';
  const accentColor = isIntervene ? '#61AFEF' : '#56B6C2';
  const timeAgo = formatTimeAgo(suggestion.timestamp);

  if (isNudge) {
    return (
      <div className="slide-up" style={{
        padding: '6px 12px',
        fontSize: 13,
        color: 'var(--text-muted)',
        fontStyle: 'italic',
        opacity: 0.7,
      }}>
        💬 rune: {suggestion.body}
      </div>
    );
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
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <span style={{
            fontSize: 13,
            fontWeight: 600,
            color: accentColor,
          }}>
            💬 rune
          </span>
          <span style={{
            fontSize: 11,
            color: 'var(--text-muted)',
          }}>
            {timeAgo}
          </span>
        </div>

        {/* Body */}
        <div style={{
          fontSize: 14,
          lineHeight: 1.6,
          color: 'var(--text-primary)',
        }}>
          {suggestion.body}
        </div>
      </div>
    </div>
  );
}

function formatTimeAgo(ts: number): string {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return '방금';
  if (diff < 3600) return `${Math.floor(diff / 60)}분 전`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`;
  return `${Math.floor(diff / 86400)}일 전`;
}
