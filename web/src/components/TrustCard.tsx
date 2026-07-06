import { useState } from 'react';
import type { TrustInfo } from '../types';

interface TrustCardProps {
  trust: TrustInfo;
  /** Re-run the last request on the escalation model. */
  onEscalate?: () => void;
}

function testsPassed(counts: Record<string, number>): number {
  return counts.pass ?? counts.passed ?? 0;
}

/**
 * The verify-or-fail-honestly verdict. RUNE's differentiator: it doesn't claim
 * "done" without evidence — verified runs show what backed the call, and runs
 * it couldn't verify say so plainly with the next step, instead of pretending.
 */
export function TrustCard({ trust, onEscalate }: TrustCardProps) {
  const [showEvidence, setShowEvidence] = useState(false);
  const gate = trust.evidenceGate;
  const passes = gate ? testsPassed(gate.verdictCounts) : 0;
  const accent = trust.verified ? 'var(--success)' : 'var(--warning)';
  const bg = trust.verified ? 'var(--success-subtle)' : 'var(--warning-subtle, var(--bg-secondary))';

  return (
    <div style={{
      alignSelf: 'flex-start',
      maxWidth: 560,
      margin: '2px 0',
      border: `1px solid ${accent}`,
      borderLeft: `3px solid ${accent}`,
      borderRadius: 'var(--radius-md)',
      background: bg,
      padding: '10px 13px',
      fontSize: 13,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span aria-hidden="true" style={{ fontSize: 14 }}>{trust.verified ? '✓' : '⚠'}</span>
        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
          {trust.verified ? 'Verified — done' : "Not marking this done"}
        </span>
        {gate?.hasCheck && (
          <span style={{
            marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-muted)',
          }}>
            {passes > 0 ? `${passes} check${passes > 1 ? 's' : ''} passed` : gate.lastVerdict}
          </span>
        )}
      </div>

      {trust.verified ? (
        gate?.hasCheck && (
          <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Checked against your project's tests before reporting done.{' '}
            {gate.lastEvidence && (
              <button
                type="button"
                onClick={() => setShowEvidence(s => !s)}
                style={{
                  background: 'none', border: 'none', color: 'var(--accent)',
                  cursor: 'pointer', padding: 0, fontSize: 12,
                }}
              >
                {showEvidence ? 'hide evidence' : 'show evidence'}
              </button>
            )}
          </div>
        )
      ) : (
        <div style={{ marginTop: 6, fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
          {trust.honestNote || "I couldn't verify this result, so I won't claim it's done."}
          {trust.escalationHint && (
            <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--text-muted)' }}>{trust.escalationHint}</span>
              {onEscalate && (
                <button
                  type="button"
                  onClick={onEscalate}
                  style={{
                    background: 'var(--accent)', color: '#0A1319', border: 'none',
                    borderRadius: 'var(--radius-sm)', padding: '5px 12px',
                    fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Retry on a stronger model
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {showEvidence && gate?.lastEvidence && (
        <pre style={{
          marginTop: 8, padding: '8px 10px', background: 'var(--bg-primary)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
          fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.5,
          whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text-secondary)',
          maxHeight: 160, overflow: 'auto',
        }}>{gate.lastEvidence}</pre>
      )}
    </div>
  );
}
