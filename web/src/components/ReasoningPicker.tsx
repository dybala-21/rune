import { useState } from 'react';
import { setReasoningEffort } from '../api';
import { toast } from '../utils/toast';

type Effort = 'low' | 'medium' | 'high';
const LEVELS: Effort[] = ['low', 'medium', 'high'];
const LABEL: Record<Effort, string> = { low: 'Low', medium: 'Med', high: 'High' };

// Reasoning depth for reasoning-capable models. Shown only when the active
// model accepts a reasoning_effort. Low favors speed/cost, High thinks more.
export function ReasoningPicker({ effort }: { effort: Effort | null }) {
  // null (provider default) shows as Medium — GPT-5.6's own default.
  const [current, setCurrent] = useState<Effort>(effort ?? 'medium');
  const [busy, setBusy] = useState(false);

  const pick = async (e: Effort) => {
    if (e === current || busy) return;
    const prev = current;
    setCurrent(e);
    setBusy(true);
    try {
      await setReasoningEffort(e);
    } catch {
      setCurrent(prev);
      toast.error("Couldn't set reasoning depth");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      title="Reasoning depth — Low is faster/cheaper, High thinks more"
      style={{
        display: 'flex', alignItems: 'center', gap: 0,
        border: '1px solid var(--border)', borderRadius: 999,
        background: 'var(--bg-tertiary)', overflow: 'hidden',
      }}
    >
      <span style={{ color: 'var(--text-muted)', fontSize: 10, padding: '0 6px 0 9px' }}>
        Reason
      </span>
      {LEVELS.map(e => {
        const on = e === current;
        return (
          <button
            key={e}
            onClick={() => pick(e)}
            aria-pressed={on}
            style={{
              padding: '4px 9px', fontSize: 11, cursor: 'pointer', border: 'none',
              background: on ? 'var(--accent)' : 'transparent',
              color: on ? 'white' : 'var(--text-secondary)',
              fontWeight: on ? 600 : 400,
            }}
          >
            {LABEL[e]}
          </button>
        );
      })}
    </div>
  );
}
