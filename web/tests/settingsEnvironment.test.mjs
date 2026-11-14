import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, AzureSettings, LocalModelSettings;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ AzureSettings } = await server.ssrLoadModule('/src/components/AzureSettings.tsx'));
  ({ LocalModelSettings } = await server.ssrLoadModule('/src/components/LocalModelSettings.tsx'));
});
after(async () => { await server?.close(); });
const render = (component, props) => renderToStaticMarkup(createElement(component, { onSaved: async () => {}, ...props }));
const azure = { revision: '1', credentialRevision: '1', endpoint: '', endpointKey: 'AZURE_API_BASE',
  endpointScope: null, hasKey: false, keyName: 'AZURE_API_KEY', keyScope: null, deployments: {}, suggestedModels: [] };
const servers = { ollama: { baseUrl: 'http://localhost:11434', envKey: 'OLLAMA_API_BASE', scope: null },
  endpoints: { gpu: { baseUrl: 'https://models.example.com/v1', models: [], visionModels: [], toolMode: 'native',
    keyName: 'RUNE_ENDPOINT_GPU_API_KEY', hasKey: false, keyScope: null, selfHosted: false } } };

test('unavailable Azure setup shows the server reason without editable or paid actions', () => {
  const html = render(AzureSettings, { connection: { ...azure, unavailableReason: 'Azure setup is managed elsewhere.' } });
  assert.match(html, /Azure setup is managed elsewhere/);
  assert.doesNotMatch(html, /<input|<button|<select/);
  for (const unavailableReason of [null, undefined]) {
    const editable = render(AzureSettings, { connection: { ...azure, unavailableReason } });
    assert.match(editable, /id="azure-endpoint"/);
    assert.match(editable, /Connect and fetch models/);
  }
});

test('custom endpoint policy changes guidance while keeping server and key settings', () => {
  const hosted = render(LocalModelSettings, { state: { ...servers, endpointsPublicHttpsOnly: true } });
  assert.match(hosted, /Custom endpoints must use public HTTPS/);
  assert.match(hosted, /placeholder="https:\/\/models.example.com\/v1"/);
  assert.match(hosted, /id="endpoint-key-0"/);
  assert.match(hosted, />Add endpoint</);
  assert.match(hosted, /connected Rune engine/);
  for (const endpointsPublicHttpsOnly of [false, undefined]) {
    const html = render(LocalModelSettings, { state: { ...servers, endpointsPublicHttpsOnly } });
    assert.doesNotMatch(html, /Custom endpoints must use public HTTPS/);
    assert.match(html, /id="ollama-base"/);
    assert.match(html, />Add endpoint</);
  }
});
