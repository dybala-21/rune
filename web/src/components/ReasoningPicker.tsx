import { useEffect, useState } from 'react';
import { setReasoningEffort, type ReasoningEffort } from '../api';
import { toast } from '../utils/toast';
import { PickerPopover, SelectedCheck } from './PickerPopover';

const LABEL: Record<ReasoningEffort, string> = {
  none: 'None', minimal: 'Minimal', low: 'Low', medium: 'Medium',
  high: 'High', xhigh: 'XHigh', max: 'Max',
};

const DETAIL: Record<ReasoningEffort, string> = {
  none: 'Without additional reasoning',
  minimal: 'A brief reasoning pass',
  low: 'Light reasoning for quick tasks',
  medium: 'A balance of depth and speed',
  high: 'More room to work through a problem',
  xhigh: 'Deeper reasoning for complex tasks',
  max: 'The highest available reasoning effort',
};

function EffortBars({ level }: { level: ReasoningEffort | null }) {
  const strength = level ? { none: 0, minimal: 1, low: 1, medium: 2, high: 3, xhigh: 4, max: 5 }[level] : 0;
  return <span className="effort-bars" aria-hidden="true">
    {[1, 2, 3, 4, 5].map(n => <i key={n} data-filled={n <= strength} style={{ height: 4 + n * 2 }} />)}
  </span>;
}

export function ReasoningPicker({ effort, options, model, budgets = {} }: {
  effort: ReasoningEffort | null;
  options: ReasoningEffort[];
  model: { provider: string; model: string };
  budgets?: Partial<Record<ReasoningEffort, number>>;
}) {
  const [current, setCurrent] = useState<ReasoningEffort | null>(effort);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setCurrent(effort); }, [effort]);

  const pick = async (value: '' | ReasoningEffort, close: () => void) => {
    if (busy) return;
    setBusy(true);
    try {
      const result = await setReasoningEffort(value, model);
      setCurrent(result.reasoningEffort);
      close();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Couldn't set reasoning depth");
    } finally {
      setBusy(false);
    }
  };

  const selected = current && options.includes(current) ? current : null;
  const choices: Array<ReasoningEffort | null> = [null, ...options];
  return <PickerPopover
    label={`Reasoning depth: ${selected ? LABEL[selected] : 'Default'}`}
    className="reasoning-trigger" busy={busy}
    trigger={<><EffortBars level={selected} /><span>{selected ? LABEL[selected] : 'Default'}</span></>}
  >
    {close => <>
      <div className="picker-heading">
        <strong>Reasoning depth</strong>
        <span>{Object.keys(budgets).length ? 'Thinking token budget' : 'Effort for this model'}</span>
      </div>
      {choices.map(level => <button
        key={level ?? 'default'} type="button" role="menuitemradio" tabIndex={-1}
        aria-checked={selected === level} disabled={busy} className="picker-option"
        onClick={() => void pick(level ?? '', close)}
      >
        <span className="picker-option-icon"><EffortBars level={level} /></span>
        <span className="picker-option-copy">
          <strong>{level ? LABEL[level] : 'Default'}</strong>
          <small>{level && budgets[level] !== undefined
            ? `${budgets[level]?.toLocaleString()} thinking tokens`
            : level ? DETAIL[level] : 'Use the model’s default depth'}</small>
        </span>
        {selected === level && <SelectedCheck />}
      </button>)}
      <div className="picker-footer">Saved for this model. Higher levels can take longer.</div>
    </>}
  </PickerPopover>;
}
