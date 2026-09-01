import { useEffect, useRef, useState } from 'react';
import { fetchModels, setActiveModel } from '../api';
import { toast } from '../utils/toast';
import { ProviderMark } from './ProviderMark';

interface ActiveModel { provider: string; model: string; source: string; }

// The status-bar model chip, now a button. Clicking opens a two-step picker:
// provider first, then that provider's models — so switching is intuitive
// without needing to know the exact provider:model string.
export function ModelPicker({ active }: { active: ActiveModel }) {
  const [open, setOpen] = useState(false);
  const [providers, setProviders] = useState<Record<string, string[]>>({});
  const [step, setStep] = useState<'provider' | 'model'>('provider');
  const [selProvider, setSelProvider] = useState('');
  const [override, setOverride] = useState<{ provider: string; model: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    if (!Object.keys(providers).length) {
      fetchModels().then(setProviders).catch(() => toast.error("Couldn't load models"));
    }
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const shown: ActiveModel = override ? { ...override, source: 'active' } : active;

  const toggle = () => {
    setStep('provider');
    setSelProvider('');
    setOpen(o => !o);
  };

  const pickModel = async (provider: string, model: string) => {
    setBusy(true);
    try {
      await setActiveModel(provider, model);
      setOverride({ provider, model });
      toast.success(`Model → ${provider}:${model}`);
      setOpen(false);
    } catch {
      toast.error("Couldn't switch model");
    } finally {
      setBusy(false);
    }
  };

  const providerNames = Object.keys(providers);

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={toggle}
        title={`Model (${shown.source}): ${shown.provider}:${shown.model} — click to change`}
        aria-haspopup="menu"
        aria-expanded={open}
        style={{
          display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px',
          background: open ? 'var(--bg-hover)' : 'var(--bg-tertiary)',
          border: '1px solid var(--border)', borderRadius: 999,
          color: 'var(--text-secondary)', fontSize: 11, cursor: 'pointer',
          minWidth: 0, maxWidth: 320,
        }}
      >
        <ProviderMark provider={shown.provider} size={14} />
        <span style={{
          fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap',
          overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          {shown.model}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 9 }}>▾</span>
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute', top: 'calc(100% + 6px)', right: 0, zIndex: 100,
            minWidth: 240, maxWidth: 320, maxHeight: 360, overflowY: 'auto',
            background: 'var(--bg-surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg)',
            padding: 4,
          }}
        >
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '6px 8px',
            fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase',
            color: 'var(--text-muted)',
          }}>
            {step === 'model' && (
              <button
                onClick={() => setStep('provider')}
                aria-label="Back to providers"
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--text-secondary)', padding: 0, fontSize: 12,
                }}
              >‹</button>
            )}
            {step === 'model' && <ProviderMark provider={selProvider} size={14} />}
            <span>{step === 'provider' ? 'Provider' : selProvider}</span>
          </div>

          {step === 'provider' ? (
            providerNames.length === 0 ? (
              <div style={{ padding: '8px', color: 'var(--text-muted)', fontSize: 12 }}>Loading…</div>
            ) : (
              providerNames.map(p => (
                <button
                  key={p}
                  onClick={() => { setSelProvider(p); setStep('model'); }}
                  style={rowBtn(p === shown.provider)}
                >
                  <ProviderMark provider={p} />
                  <span style={{ flex: 1, textAlign: 'left' }}>{p}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                    {providers[p].length} ›
                  </span>
                </button>
              ))
            )
          ) : (
            (providers[selProvider] ?? []).map(m => {
              const isCurrent = selProvider === shown.provider && m === shown.model;
              return (
                <button
                  key={m}
                  disabled={busy}
                  onClick={() => pickModel(selProvider, m)}
                  style={rowBtn(isCurrent)}
                >
                  <span style={{
                    flex: 1, textAlign: 'left', fontFamily: 'var(--font-mono)',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>{m}</span>
                  {isCurrent && <span style={{ color: 'var(--accent)', fontSize: 11 }}>✓</span>}
                </button>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

function rowBtn(active: boolean): React.CSSProperties {
  return {
    display: 'flex', alignItems: 'center', gap: 8, width: '100%',
    padding: '7px 10px', background: active ? 'var(--accent-subtle)' : 'none',
    border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
    color: active ? 'var(--accent)' : 'var(--text-primary)', fontSize: 12.5,
  };
}
