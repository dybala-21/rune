import { useEffect, useState } from 'react';
import { fetchEscalationStatus, setEscalation, type EscalationStatus } from '../api';
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
  const [esc, setEsc] = useState<EscalationStatus | null>(null);
  // Cloud retry asks for one confirm first — that click is the moment code
  // leaves the machine, so it shouldn't fire on a single tap.
  const [confirmCloud, setConfirmCloud] = useState(false);
  // Only the not-verified card needs the ladder; fetch lazily then.
  useEffect(() => {
    if (trust.verified) return;
    let live = true;
    fetchEscalationStatus().then(s => live && setEsc(s)).catch(() => {});
    return () => { live = false; };
  }, [trust.verified]);
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
          <div style={{ marginTop: 8 }}>
            {esc?.enabled ? (
              esc.isCloud && confirmCloud ? (
                <div style={{
                  display: 'flex', flexDirection: 'column', gap: 8,
                  padding: '9px 11px', borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--warning)', background: 'var(--warning-subtle, var(--bg-secondary))',
                }}>
                  <span style={{ color: 'var(--text-primary)', fontSize: 12 }}>
                    Send this run’s code to <b>{esc.model || esc.provider}</b> in the cloud?
                    It leaves your machine and the provider may retain it briefly
                    (~30 days) for abuse monitoring.
                  </span>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      type="button"
                      onClick={() => { setConfirmCloud(false); onEscalate?.(); }}
                      style={{
                        background: 'var(--warning)', color: '#0A1319', border: 'none',
                        borderRadius: 'var(--radius-sm)', padding: '5px 12px',
                        fontSize: 12, fontWeight: 600, cursor: 'pointer',
                      }}
                    >
                      Send and retry
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmCloud(false)}
                      style={{
                        background: 'none', color: 'var(--text-muted)',
                        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                        padding: '5px 12px', fontSize: 12, cursor: 'pointer',
                      }}
                    >
                      Keep it local
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  {onEscalate && (
                    <button
                      type="button"
                      onClick={() => { if (esc.isCloud) setConfirmCloud(true); else onEscalate(); }}
                      style={{
                        background: 'var(--accent)', color: '#0A1319', border: 'none',
                        borderRadius: 'var(--radius-sm)', padding: '5px 12px',
                        fontSize: 12, fontWeight: 600, cursor: 'pointer',
                      }}
                    >
                      Retry on {esc.model || esc.provider}
                    </button>
                  )}
                  <span style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>
                    {esc.isCloud
                      ? '↑ a stronger cloud model — sends this run’s code off your machine'
                      : '↑ a stronger local model — stays on your machine'}
                  </span>
                </div>
              )
            ) : esc && !esc.enabled && esc.suggestion ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={async () => {
                    // Adopt the suggested local model for this session, then retry.
                    await setEscalation('ollama', esc.suggestion!).catch(() => {});
                    onEscalate?.();
                  }}
                  style={{
                    background: 'var(--accent)', color: '#0A1319', border: 'none',
                    borderRadius: 'var(--radius-sm)', padding: '5px 12px',
                    fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Retry on {esc.suggestion}
                </button>
                <span style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>
                  ↑ a larger model you already have — stays on your machine
                </span>
              </div>
            ) : esc && !esc.enabled ? (
              <span style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>
                Want a stronger model to retry? Set{' '}
                <code style={{ color: 'var(--accent)' }}>llm.escalationProvider</code>{' '}
                in Settings, then this becomes a one-click retry.
              </span>
            ) : null}
          </div>
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
