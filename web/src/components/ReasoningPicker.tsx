import type { ReasoningEffort } from '../api';
import { SelectedCheck } from './PickerPopover';

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

export function reasoningLabel(effort: ReasoningEffort | null) {
  return effort ? LABEL[effort] : 'Auto';
}

function EffortBars({ level }: { level: ReasoningEffort | null }) {
  const strength = level ? { none: 0, minimal: 1, low: 1, medium: 2, high: 3, xhigh: 4, max: 5 }[level] : 0;
  return <span className="effort-bars" aria-hidden="true">
    {[1, 2, 3, 4, 5].map(n => <i key={n} data-filled={n <= strength} style={{ height: 4 + n * 2 }} />)}
  </span>;
}

export function ReasoningPicker({ effort, options, budgets = {}, disabled, onPick }: {
  effort: ReasoningEffort | null;
  options: ReasoningEffort[];
  budgets?: Partial<Record<ReasoningEffort, number>>;
  disabled: boolean;
  onPick: (effort: ReasoningEffort | null) => void;
}) {
  const selected = effort && options.includes(effort) ? effort : null;
  const choices: Array<ReasoningEffort | null> = [null, ...options];
  return <>
    {choices.map(level => <button
      key={level ?? 'default'} type="button" role="menuitemradio" tabIndex={-1}
      aria-checked={selected === level} disabled={disabled} className="picker-option"
      onClick={() => onPick(level)}
    >
      <span className="picker-option-icon"><EffortBars level={level} /></span>
      <span className="picker-option-copy">
        <strong>{reasoningLabel(level)}</strong>
        <small>{level && budgets[level] !== undefined
          ? `${budgets[level]?.toLocaleString()} thinking tokens`
          : level ? DETAIL[level] : 'Use this model’s default'}</small>
      </span>
      {selected === level && <SelectedCheck />}
    </button>)}
    <div className="picker-footer">Saved for this model. Deeper reasoning can take longer.</div>
  </>;
}
