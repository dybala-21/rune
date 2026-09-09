import { useEffect, useState } from 'react';
import { fetchModels, setActiveModel } from '../api';
import { toast } from '../utils/toast';
import { ProviderMark } from './ProviderMark';
import { PickerPopover, SelectedCheck } from './PickerPopover';

interface ActiveModel { provider: string; model: string; source: string; }

export function ModelPicker({ active }: { active: ActiveModel }) {
  const [providers, setProviders] = useState<Record<string, string[]>>({});
  const [selProvider, setSelProvider] = useState(active.provider);
  const [override, setOverride] = useState<ActiveModel | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  useEffect(() => { setOverride(null); }, [active.provider, active.model]);
  const shown = override ?? active;

  const loadModels = () => {
    setLoadFailed(false);
    fetchModels().then(setProviders).catch(() => {
      setLoadFailed(true);
      toast.error("Couldn't load models");
    });
  };
  const onOpen = () => {
    setSelProvider(shown.provider);
    if (!Object.keys(providers).length) loadModels();
  };
  const pickModel = async (provider: string, model: string, close: () => void) => {
    if (busy) return;
    setBusy(true);
    try {
      await setActiveModel(provider, model);
      setOverride({ provider, model, source: 'active' });
      close();
    } catch {
      toast.error("Couldn't switch model");
    } finally {
      setBusy(false);
    }
  };

  return <PickerPopover
    label={`Model: ${shown.model}`} className="model-trigger" busy={busy}
    onOpen={onOpen} menuKey={`${selProvider}/${Object.keys(providers).length}`}
    trigger={<><ProviderMark provider={shown.provider} size={16} /><span className="model-name">{shown.model}</span></>}
  >
    {close => <>
      <div className="picker-heading">
        <strong>Choose a model</strong>
        <span>For your next conversation turn</span>
      </div>
      {selProvider && <button type="button" role="menuitem" tabIndex={-1}
        className="picker-back" onClick={() => setSelProvider('')} disabled={busy}>
        <span aria-hidden="true">←</span><ProviderMark provider={selProvider} size={14} />
        {selProvider}<span className="picker-back-hint">All providers</span>
      </button>}
      {loadFailed ? <button type="button" role="menuitem" tabIndex={-1} className="picker-option" onClick={loadModels}>
        Couldn’t load models. Try again
      </button> : !Object.keys(providers).length ? <div className="picker-loading" role="status">Loading models…</div>
        : !selProvider ? Object.entries(providers).map(([provider, models]) => <button
          key={provider} type="button" role="menuitem" tabIndex={-1} className="picker-option"
          onClick={() => setSelProvider(provider)}
        >
          <span className="picker-option-icon"><ProviderMark provider={provider} size={18} /></span>
          <span className="picker-option-copy"><strong>{provider}</strong></span>
          <span className="picker-count">{models.length}</span><span aria-hidden="true">›</span>
        </button>) : (providers[selProvider] ?? []).map(model => {
          const selected = shown.provider === selProvider && shown.model === model;
          return <button key={model} type="button" role="menuitemradio" tabIndex={-1}
            aria-checked={selected} disabled={busy} className="picker-option model-option"
            onClick={() => void pickModel(selProvider, model, close)}>
            <span className="picker-option-copy"><strong>{model}</strong></span>
            {selected && <SelectedCheck />}
          </button>;
        })}
    </>}
  </PickerPopover>;
}
