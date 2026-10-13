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

test('approval history shows the exact action and separate approve and dismiss controls', () => {
  const html = renderToStaticMarkup(createElement(fields.RoutineHistory, {
    job: { id: 'job', enabled: true, recentRuns: [{ id: 'cron:job:1', started_at: 1, blocked: true,
      result: { status: 'needs_approval', pending_approval: { reason: 'Send prepared reply',
        action: { tool: 'web_fetch', params: { method: 'POST', url: 'https://mail.example.test/send', body: 'hello' } } } } }] },
    onRefresh: async () => {},
  }));
  for (const text of ['Approval needed', 'Approve once and continue', 'Dismiss and pause task', 'https://mail.example.test/send', 'remaining task budget']) assert.ok(html.includes(text));
  assert.ok(!html.includes('Record review and clear block'));
});

test('expired screen references offer dismissal instead of approving an old click', () => {
  const html = renderToStaticMarkup(createElement(fields.RoutineHistory, {
    job: { id: 'job', enabled: true, recentRuns: [{ id: 'cron:job:1', started_at: 1, blocked: true,
      result: { status: 'needs_approval', pending_approval: { reason: 'Click', action: { tool: 'browser_act', params: { selector: 'e1' } } } } }] },
    onRefresh: async () => {},
  }));
  assert.ok(html.includes('fresh interactive review'));
  assert.ok(!html.includes('Approve once and continue'));
});

test('a running task does not offer manual recovery controls', () => {
  const html = renderToStaticMarkup(createElement(fields.RoutineHistory, {
    job: { id: 'job', enabled: false, recentRuns: [{ id: 'run', started_at: 1, blocked: true, state: 'running', result: null }] },
    onRefresh: async () => {},
  }));
  assert.ok(html.includes('Task running'));
  assert.ok(!html.includes('Record review and clear block'));
});

test('a stopped worker is labelled as requiring recovery', () => {
  const html = renderToStaticMarkup(createElement(fields.RoutineHistory, {
    job: { id: 'job', enabled: false, recentRuns: [{ id: 'run', started_at: 1, blocked: true, state: 'recovery_required', result: null }] },
    onRefresh: async () => {},
  }));
  assert.ok(html.includes('recovery required'));
  assert.ok(html.includes('Record review and clear block'));
  assert.ok(!html.includes('Task running'));
});
