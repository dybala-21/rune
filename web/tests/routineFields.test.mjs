import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, fields;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  fields = await server.ssrLoadModule('/src/components/RoutineFields.tsx');
});
after(async () => { await server?.close(); });

test('routine totals allow blank and zero but reject invalid amounts', () => {
  const policy = fields.defaultRoutinePolicy();
  fields.validateRoutinePolicy(policy);
  fields.validateRoutinePolicy({ ...policy, total_tokens: 0, total_model_requests: 0, total_cost_usd: 0 });
  for (const patch of [{ total_tokens: -1 }, { total_model_requests: 1.5 }, { total_cost_usd: NaN }, { total_cost_usd: Infinity }]) {
    assert.throws(() => fields.validateRoutinePolicy({ ...policy, ...patch }));
  }
});

test('task history labels partial usage and displays the budget stop', () => {
  const html = renderToStaticMarkup(createElement(fields.RoutineHistory, {
    job: { id: 'job', recentRuns: [], budget: { used: { modelRequests: 2, tokens: 200, costUsd: .02,
      requestsKnown: true, tokensKnown: false, costKnown: false }, blocked: true, reason: 'Total limit reached' } },
    onRefresh: async () => {},
  }));
  assert.ok(html.includes('2 model requests'));
  assert.ok(html.includes('not a complete total'));
  assert.ok(html.includes('Total limit reached'));
});
