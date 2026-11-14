import type { ConfigInfo, ModelConnectionAction } from '../api';
import { ModelPicker, type ModelMenuRequest } from './ModelPicker';
import './ModelControls.css';

export function ModelControls({ config, openRequest, onConfigure, onModelChange }: {
  config: ConfigInfo | null;
  openRequest?: ModelMenuRequest;
  onConfigure?: (action: ModelConnectionAction) => void;
  onModelChange?: (model: { provider: string; model: string }) => void;
}) {
  if (!config?.activeModel) return null;
  return <div className="composer-models" aria-label="Model settings">
    <ModelPicker config={config} openRequest={openRequest} onConfigure={onConfigure} onModelChange={onModelChange} />

  </div>;
}
