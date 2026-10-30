import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, DecisionRoutingSettings, parseMicrosoftDecision, decisionStatusText;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ DecisionRoutingSettings, parseMicrosoftDecision, decisionStatusText } = await server.ssrLoadModule('/src/components/DecisionRoutingSettings.tsx'));
});
after(async () => { await server?.close(); });

function routing(backend = 'connected', status = 'disabled') {
  return { backend, status, effectiveBackend: backend, microsoft: { endpoint: '', deployment: '' },
    providers: { jev: { hasKey: true, keyScope: 'user' }, microsoft: { hasKey: false, keyScope: null } } };
}

test('off and both decision providers are labeled radio choices', () => {
  const html = renderToStaticMarkup(createElement(DecisionRoutingSettings, { routing: routing(), onSaved: async () => {} }));
  assert.equal((html.match(/type="radio"/g) ?? []).length, 3);
  assert.match(html.match(/<input[^>]+value="connected"[^>]*>/)?.[0] ?? '', /\bchecked=/);
  assert.match(html, />Jev</);
  assert.match(html, />Microsoft</);
  assert.doesNotMatch(html, /type="password"/);
  assert.match(html, /Off. Your current model handles task routing./);
});

test('Microsoft setup uses its own saved key state and explains the deployment name', () => {
  const html = renderToStaticMarkup(createElement(DecisionRoutingSettings, { routing: routing('microsoft', 'missing_key'), onSaved: async () => {} }));
  assert.match(html, /Azure endpoint/);
  assert.match(html, /Deployment name/);
  assert.match(html, /Required/);
  assert.match(html, /type="password"/);
  assert.doesNotMatch(html, /Leave blank to keep the saved key/);
  assert.match(html, /deployment name, which may differ from the model name/);
});

test('Azure URLs are normalized without accepting redirects or credential-bearing URLs', () => {
  assert.deepEqual(parseMicrosoftDecision('https://test.services.ai.azure.com/providers/microsoft/v1/systemone', 'Decision-1'),
    { endpoint: 'https://test.services.ai.azure.com', deployment: 'Decision-1' });
  for (const endpoint of ['http://test.services.ai.azure.com', 'https://user:pass@test.services.ai.azure.com',
    'https://test.services.ai.azure.com.evil.test', 'https://test.services.ai.azure.com?key=secret', 'https://localhost']) {
    assert.throws(() => parseMicrosoftDecision(endpoint, 'Decision-1'));
  }
  assert.throws(() => parseMicrosoftDecision('https://test.services.ai.azure.com', ''));
});

test('paused and unavailable connections do not look active', () => {
  assert.match(decisionStatusText(routing('microsoft', 'local')), /Paused.*local/);
  assert.match(decisionStatusText(routing('microsoft', 'automatic_model')), /Paused.*automatic/);
  assert.match(decisionStatusText(routing('microsoft', 'cooldown')), /Using your current model/);
  assert.match(decisionStatusText(routing('microsoft', 'unverified')), /checked on your next request/);
});
