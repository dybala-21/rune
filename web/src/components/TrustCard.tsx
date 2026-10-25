import { useEffect, useState } from 'react';
import { fetchEscalationStatus, setEscalation, type EscalationStatus } from '../api';
import type { TrustInfo } from '../types';
import { RunEvidence } from './RunEvidence';
import { checkSummary, describeTrust, trustColors } from '../utils/trust';

interface TrustCardProps {
  trust: TrustInfo;
  onRetry?: () => void;
  /** Re-run the last request on the escalation model. */
  onEscalate?: () => void;
}

/** Show the result and any unresolved checks. */
export function TrustCard({ trust, onRetry, onEscalate }: TrustCardProps) {
  const [esc, setEsc] = useState<EscalationStatus | null>(null);
  // Confirm before sending this run to a cloud model.
  const [confirmCloud, setConfirmCloud] = useState(false);
  const view = describeTrust(trust);
  const hasEscalationAction = Boolean(onEscalate);
  useEffect(() => {
    setEsc(null);
    setConfirmCloud(false);
    if (!view.canEscalate || !hasEscalationAction) return;
    let live = true;
    fetchEscalationStatus().then(s => live && setEsc(s)).catch(error => {
      console.warn('Could not load retry options', error);
    });
    return () => { live = false; };
  }, [view.canEscalate, trust.reason, hasEscalationAction]);
  const gate = trust.evidenceGate;
  const { accent, background } = trustColors(view.tone);
  if (!view.showCard) return <RunEvidence trust={trust} />;

  return (
    <div style={{
      alignSelf: 'flex-start',
      maxWidth: 560,
      minWidth: 0,
      width: '100%',
      overflowWrap: 'anywhere',
      margin: '2px 0',
      border: `1px solid ${accent}`,
      borderLeft: `3px solid ${accent}`,
      borderRadius: 'var(--radius-md)',
      background,
      padding: '10px 13px',
      fontSize: 13,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <span aria-hidden="true" style={{ fontSize: 14, flexShrink: 0 }}>{view.glyph}</span>
        <span style={{ fontWeight: 600, color: 'var(--text-primary)', minWidth: 0 }}>
          {view.title}
        </span>
        {gate?.hasCheck && (
          <span style={{
            marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-muted)',
          }}>
            {checkSummary(trust)}
          </span>
        )}
      </div>

      <div style={{ marginTop: 6, fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
        {view.note}{' '}
        {view.canEscalate && onEscalate && (
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
                        background: 'var(--warning)', color: 'var(--text-on-accent)', border: 'none',
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
                        background: 'var(--accent)', color: 'var(--text-on-accent)', border: 'none',
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
                    try {
                      await setEscalation('ollama', esc.suggestion!);
                      onEscalate?.();
                    } catch (error) {
                      console.warn('Could not set the retry model', error);
                    }
                  }}
                  style={{
                    background: 'var(--accent)', color: 'var(--text-on-accent)', border: 'none',
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
        )}
      </div>

      {onRetry && <button type="button" className="msg-action-btn" onClick={onRetry} style={{ marginTop: 8 }}>Retry request</button>}
      <RunEvidence trust={trust} />
    </div>
  );
}
