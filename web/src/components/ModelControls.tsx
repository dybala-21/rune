import type { ConfigInfo, ModelConnectionAction } from '../api';
import { ModelPicker, type ModelMenuRequest } from './ModelPicker';
import './ModelControls.css';

export function ModelControls({ config, openRequest, onConfigure }: {
  config: ConfigInfo | null;
  openRequest?: ModelMenuRequest;
  onConfigure?: (action: ModelConnectionAction) => void;
}) {
  if (!config?.activeModel) return null;
  return <div className="composer-models" aria-label="Model settings">
    <ModelPicker config={config} openRequest={openRequest} onConfigure={onConfigure} />

  </div>;
}
