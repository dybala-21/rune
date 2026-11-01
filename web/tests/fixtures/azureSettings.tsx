import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AzureSettings } from '../../src/components/AzureSettings';
import { fetchConfig, type ConfigInfo } from '../../src/api';
import '../../src/styles/globals.css';

const initial: ConfigInfo['azure'] = {
  endpoint: 'https://fixture.services.ai.azure.com/openai/v1', endpointKey: 'AZURE_API_BASE', endpointScope: 'project',
  hasKey: true, keyName: 'AZURE_API_KEY', keyScope: 'project', revision: 'initial', credentialRevision: 'credentials',
  deployments: { daily: 'Kimi-K2.7-Code', reasoner: 'gpt-6-sol' }, suggestedModels: ['Kimi-K2.7-Code', 'DeepSeek-V4-Pro'], pricing: null,
};
function Fixture() {
  const [connection, setConnection] = useState(initial!);
  const [active, setActive] = useState<ConfigInfo['activeModel']>({ provider: 'azure', model: 'daily', source: 'active' });
  const load = async () => { const next = await fetchConfig(); setConnection(next.azure!); setActive(next.activeModel); };
  return <div className="fixture-scroll" style={{ width: 'min(400px, 100%)', height: '100vh', overflowY: 'auto', margin: 'auto' }}>
    <AzureSettings connection={connection} activeModel={active} onSaved={load} />
    <button id="poll" onClick={load}>Simulate settings refresh</button>
  </div>;
}
createRoot(document.getElementById('root')!).render(<Fixture />);
