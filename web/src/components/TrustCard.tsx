import { useEffect, useState } from 'react';
import { fetchEscalationStatus, setEscalation, type EscalationStatus } from '../api';
import type { TrustInfo } from '../types';
import { checkEvidence, checkSummary, describeTrust, trustColors } from '../utils/trust';

interface TrustCardProps {
  trust: TrustInfo;
  /** Re-run the last request on the escalation model. */
  onEscalate?: () => void;
}

/** Show the checks supporting a result and any remaining verification gaps. */
export function TrustCard({ trust, onEscalate }: TrustCardProps) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [esc, setEsc] = useState<EscalationStatus | null>(null);
  // Cloud retry asks for one confirm first — that click is the moment code
  // leaves the machine, so it shouldn't fire on a single tap.
  const [confirmCloud, setConfirmCloud] = useState(false);
  const view = describeTrust(trust);
  useEffect(() => {
    setEsc(null);
    setConfirmCloud(false);
    if (!view.canEscalate) return;
    let live = true;
    fetchEscalationStatus().then(s => live && setEsc(s)).catch(error => {
      console.warn('Could not load retry options', error);
    });
    return () => { live = false; };
  }, [view.canEscalate, trust.reason]);
  const gate = trust.evidenceGate;
  const evidence = checkEvidence(trust);
  const { accent, background } = trustColors(view.tone);
  if (!view.showCard) return null;

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
        {evidence && (
          <button
            type="button"
            onClick={() => setShowEvidence(s => !s)}
            style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', padding: 0, fontSize: 12 }}
          >
            {showEvidence ? 'hide evidence' : 'show evidence'}
          </button>
        )}
        {view.canEscalate && (
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
                    try {
                      await setEscalation('ollama', esc.suggestion!);
                      onEscalate?.();
                    } catch (error) {
                      console.warn('Could not set the retry model', error);
                    }
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
        )}
      </div>

      {trust.artifactReceipts?.map(receipt => (
        <div key={receipt.revision} style={{ marginTop: 8, fontSize: 12, lineHeight: 1.6 }}>
          <div style={{ fontWeight: 600 }}>Document checks</div>
          <div>{receipt.artifacts.map(a => a.path.split(/[\\/]/).pop()).join(', ')}</div>
          <div>Saved content: {receipt.checks.native_content === 'pass' ? 'passed' : 'not confirmed'}</div>
          <div>Source totals: {receipt.checks.source_metrics === 'pass' ? 'passed' : 'not confirmed'}</div>
          <div>Layout: {receipt.checks.visual_layout === 'pass' ? 'checked' : 'not checked'}</div>
          <div>Full task review: {receipt.checks.task_acceptance === 'pass' ? 'passed' : 'not performed'}</div>
        </div>
      ))}

      {showEvidence && evidence && (
        <pre style={{
          marginTop: 8, padding: '8px 10px', background: 'var(--bg-primary)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
          fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.5,
          whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text-secondary)',
          maxHeight: 160, overflow: 'auto',
        }}>{evidence}</pre>
      )}
    </div>
  );
}
