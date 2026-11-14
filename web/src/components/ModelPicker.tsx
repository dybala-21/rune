import { useEffect, useRef, useState } from 'react';
import { fetchModels, setActiveModel, setReasoningEffort, type ConfigInfo, type ReasoningEffort, type ModelConnectionAction } from '../api';
import { toast } from '../utils/toast';
import { ProviderMark } from './ProviderMark';
import { PickerPopover, SelectedCheck } from './PickerPopover';
import { ReasoningPicker, reasoningLabel } from './ReasoningPicker';

type ActiveModel = ConfigInfo['activeModel'];
const sameModel = (a: ActiveModel, b: ActiveModel) => a.provider === b.provider && a.model === b.model;

export interface ModelMenuRequest { provider: string; id: number }

interface Props {
  config: ConfigInfo;
  openRequest?: ModelMenuRequest;
  onConfigure?: (action: ModelConnectionAction) => void;
}

const RECENT_KEY = 'rune:recent-models';
function readRecent(): Array<{ provider: string; model: string }> {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]');
    return Array.isArray(saved) ? saved.filter(item => item && typeof item.provider === 'string' && typeof item.model === 'string').slice(0, 5) : [];
  } catch { return []; }
}

export function ModelPicker({ config, openRequest, onConfigure }: Props) {
  const active = config.activeModel;
  const [query, setQuery] = useState('');
  const [recent, setRecent] = useState(readRecent);
  const [providers, setProviders] = useState<Record<string, string[]>>({});
  const [selProvider, setSelProvider] = useState(active.provider);
  const [view, setView] = useState<'models' | 'reasoning'>('models');
  const [override, setOverride] = useState<ActiveModel | null>(null);
  const [effortOverride, setEffortOverride] = useState<{ model: ActiveModel; effort: ReasoningEffort | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const requestPending = useRef(false);
  const requestedMenu = useRef(0);
  const catalogRequest = useRef(0);
  useEffect(() => () => { catalogRequest.current++; }, []);
  useEffect(() => { setOverride(null); setView('models'); }, [active.provider, active.model]);
  useEffect(() => { setEffortOverride(null); }, [active.provider, active.model, config.reasoningEffort]);
  const shown = override ?? active;
  const settingsReady = sameModel(shown, active);
  const options = config.reasoningOptions ?? [];
  const hasReasoning = settingsReady && config.reasoningSupported && options.length > 0;
  const effort = effortOverride && sameModel(effortOverride.model, active) ? effortOverride.effort : config.reasoningEffort;
  const selectedEffort = effort && options.includes(effort) ? effort : null;

  const loadModels = () => {
    setLoadFailed(false);
    const request = ++catalogRequest.current;
    fetchModels().then(next => { if (request === catalogRequest.current) setProviders(next); }).catch(() => {
      if (request !== catalogRequest.current) return;
      setLoadFailed(true);
      toast.error("Couldn't load models");
    });
  };
  const onOpen = () => {
    setView('models');
    setQuery('');
    const requested = openRequest && openRequest.id !== requestedMenu.current;
    setSelProvider(requested ? openRequest.provider : shown.provider);
    if (openRequest) requestedMenu.current = openRequest.id;
    loadModels();
  };
  const pickModel = async (provider: string, model: string, close: () => void) => {
    if (requestPending.current) return;
    if (shown.provider === provider && shown.model === model) {
      close();
      return;
    }
    requestPending.current = true;
    setBusy(true);
    try {
      const result = await setActiveModel(provider, model);
      setOverride({ ...result, source: 'active' });
      const next = [{ provider, model }, ...recent.filter(item => item.provider !== provider || item.model !== model)].slice(0, 5);
      setRecent(next);
      try { localStorage.setItem(RECENT_KEY, JSON.stringify(next)); } catch { /* Browsing still works without local storage. */ }
      close();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Couldn't switch model");
    } finally {
      requestPending.current = false;
      setBusy(false);
    }
  };
  const pickEffort = async (value: ReasoningEffort | null, close: () => void) => {
    // Capabilities belong to config's model, which can lag behind the selected model.
    if (requestPending.current || !hasReasoning || value !== null && !options.includes(value)) return;
    requestPending.current = true;
    setBusy(true);
    try {
      const result = await setReasoningEffort(value ?? '', { provider: shown.provider, model: shown.model });
      setEffortOverride({ model: shown, effort: result.reasoningEffort });
      close();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Couldn't set reasoning depth");
    } finally {
      requestPending.current = false;
      setBusy(false);
    }
  };

  const search = query.trim().toLowerCase();
  const connection = config.modelConnection?.provider === shown.provider && config.modelConnection.model === shown.model ? config.modelConnection : null;
  const matches = Object.entries(providers).flatMap(([provider, models]) => models
    .filter(model => `${provider} ${model}`.toLowerCase().includes(search))
    .map(model => ({ provider, model })));
  const recents = recent.filter(item => providers[item.provider]?.includes(item.model)
    && !(item.provider === shown.provider && item.model === shown.model));
  const modelOption = (provider: string, model: string, close: () => void, showProvider = false) => {
    const selected = shown.provider === provider && shown.model === model;
    return <button key={`${provider}/${model}`} type="button" role="menuitemradio" tabIndex={-1}
      aria-checked={selected} disabled={busy} className="picker-option model-option"
      onClick={() => void pickModel(provider, model, close)}>
      <span className="picker-option-copy"><strong>{model}</strong>{showProvider && <small>{provider}</small>}</span>
      {selected && <SelectedCheck />}
    </button>;
  };

  return <><PickerPopover
    label={`Model: ${shown.model}${hasReasoning ? `, reasoning: ${reasoningLabel(selectedEffort)}` : ''}`}
    className="model-trigger" busy={busy}
    onOpen={onOpen} openRequest={openRequest?.id} menuKey={`${view}/${selProvider}/${Object.keys(providers).length}/${settingsReady}`}
    trigger={<><ProviderMark provider={shown.provider} size={16} /><span className="model-name">{shown.model}</span>
      {hasReasoning && <span className="model-effort">{reasoningLabel(selectedEffort)}</span>}</>}
  >
    {close => view === 'reasoning' ? <>
      <button type="button" role="menuitem" tabIndex={-1} className="picker-back"
        onClick={() => setView('models')} disabled={busy}>
        <span aria-hidden="true">←</span>Model settings
      </button>
      <div className="picker-heading">
        <strong>Reasoning</strong><span>{shown.model}</span>
      </div>
      {hasReasoning ? <ReasoningPicker effort={selectedEffort} options={options}
        budgets={config.reasoningBudgets} disabled={busy}
        onPick={value => void pickEffort(value, close)} />
        : <div className="picker-loading" role="status">{settingsReady ? 'This model has no reasoning options.' : 'Updating model settings…'}</div>}
    </> : <>
      <div className="picker-heading">
        <strong>Model settings</strong><span>Applies to your next message</span>
      </div>
      <div className="model-search">
        <input type="search" data-picker-search aria-label="Search all models" placeholder="Search all models…"
          value={query} onChange={event => setQuery(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter' && search) {
              event.preventDefault(); event.stopPropagation();
              if (matches[0] && !busy && !loadFailed) void pickModel(matches[0].provider, matches[0].model, close);
            }
          }} />
        {query && <button type="button" aria-label="Clear model search" onClick={event => { event.currentTarget.parentElement?.querySelector('input')?.focus(); setQuery(''); }}>×</button>}
      </div>
      {connection && !search && <div className="model-connection" data-state={connection.state}>
        <strong>{connection.state === 'needs_setup' ? 'Setup needed' : connection.state === 'configured' ? 'Configured · access not checked' : 'Connection not checked'}</strong>
        <p>{connection.reason}</p>
        {connection.action && onConfigure && <button type="button" role="menuitem" tabIndex={-1}
          onClick={() => { close(); onConfigure(connection.action!); }}>{connection.action.label}</button>}
      </div>}
      {!search && (hasReasoning || !settingsReady) && <button type="button" role="menuitem" tabIndex={-1}
        className="picker-option model-reasoning-link" disabled={busy || !settingsReady}
        onClick={() => setView('reasoning')}>
        <span className="picker-option-copy"><strong>Reasoning</strong></span>
        <span className="model-reasoning-value">{settingsReady ? reasoningLabel(selectedEffort) : 'Updating…'}</span>
        <span aria-hidden="true">›</span>
      </button>}
      {!search && recents.length > 0 && <div className="model-recent">
        <div className="model-list-label">Recent models</div>
        {recents.map(item => modelOption(item.provider, item.model, close, true))}
      </div>}
      {selProvider && !search && <button type="button" role="menuitem" tabIndex={-1}
        className="picker-back" onClick={() => setSelProvider('')} disabled={busy}>
        <span aria-hidden="true">←</span><ProviderMark provider={selProvider} size={14} />
        {selProvider}<span className="picker-back-hint">All providers</span>
      </button>}
      {loadFailed ? <button type="button" role="menuitem" tabIndex={-1} className="picker-option" onClick={loadModels} disabled={busy}>
        Couldn’t load models. Try again
      </button> : !Object.keys(providers).length ? <div className="picker-loading" role="status">Loading models…</div>
        : search ? matches.length ? matches.map(item => modelOption(item.provider, item.model, close, true))
          : <p className="picker-loading" role="status">No models match “{query.trim()}”.</p>
        : !selProvider ? Object.entries(providers).map(([provider, models]) => <button
          key={provider} type="button" role="menuitem" tabIndex={-1} className="picker-option"
          onClick={() => setSelProvider(provider)} disabled={busy}
        >
          <span className="picker-option-icon"><ProviderMark provider={provider} size={18} /></span>
          <span className="picker-option-copy"><strong>{provider}</strong></span>
          <span className="picker-count">{models.length}</span><span aria-hidden="true">›</span>
        </button>) : (providers[selProvider] ?? []).length
          ? (providers[selProvider] ?? []).filter(model => !recents.some(item => item.provider === selProvider && item.model === model))
            .map(model => modelOption(selProvider, model, close))
          : <p className="picker-loading" role="status">No models listed for this provider. Check its connection in Settings.</p>}
    </>}
  </PickerPopover>
    {connection?.state === 'needs_setup' && <div className="model-setup-notice" role="status">
      <span>Model needs setup.</span>
      {connection.action && onConfigure && <button type="button" title={connection.reason}
        onClick={() => onConfigure(connection.action!)}>{connection.action.label}</button>}
    </div>}
  </>;
}
