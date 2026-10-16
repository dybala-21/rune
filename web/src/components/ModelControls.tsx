import type { ConfigInfo } from '../api';
import { ModelPicker } from './ModelPicker';
import { ReasoningPicker } from './ReasoningPicker';

export function ModelControls({ config }: { config: ConfigInfo | null }) {
  const model = config?.activeModel;
  if (!model) return null;
  return <div className="composer-models" aria-label="Model settings">
    <ModelPicker active={model} />
    {config.reasoningSupported && !!config.reasoningOptions?.length && <ReasoningPicker
      key={`${model.provider}/${model.model}`} model={model}
      effort={config.reasoningEffort ?? null} options={config.reasoningOptions}
      budgets={config.reasoningBudgets} />}
  </div>;
}
