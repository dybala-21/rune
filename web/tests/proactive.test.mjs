import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, withProactive, withoutProactive, ProactiveCard;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ withProactive, withoutProactive } = await server.ssrLoadModule('/src/utils/proactive.ts'));
  ({ ProactiveCard } = await server.ssrLoadModule('/src/components/ProactiveCard.tsx'));
});
after(async () => { await server?.close(); });

const item = (id, overrides = {}) => ({
  id, title: `Proposal ${id}`, description: 'Inspect the report', confidence: .7,
  createdAt: '2026-10-03T03:00:00Z', response: 'pending', executionStatus: null, result: {}, ...overrides,
});

test('offline proposals restore once beside chat, using server state instead of a stale draft', () => {
  const chat = { id: 'chat', role: 'user', content: 'Keep this', timestamp: Date.parse('2026-10-03T02:00:00Z') };
  const stale = { id: 'old-card', role: 'system', content: 'Old', timestamp: 0, suggestion: { id: 'report' } };
  const result = item('report', { response: 'accepted', executionStatus: 'completed', result: { output: '2450' } });
  const restored = withProactive([chat, stale], [item('offline'), result, result]);
  assert.equal(restored.length, 3);
  assert.equal(restored[0], chat);
  assert.equal(restored.find(message => message.suggestion?.id === 'report').suggestion.state.result.output, '2450');
  assert.deepEqual(withProactive(restored, [item('offline'), result]), restored);
  assert.deepEqual(withoutProactive(restored), [chat]);
});

test('a dismissed or expired card never becomes actionable after a fresh render', () => {
  for (const response of ['dismissed', 'expired']) {
    const [message] = withProactive([], [item('report', { response })]);
    const html = renderToStaticMarkup(createElement(ProactiveCard, { suggestion: message.suggestion }));
    assert.doesNotMatch(html, /Run once|<button/);
    assert.match(html, new RegExp(response[0].toUpperCase() + response.slice(1)));
  }
});

test('queued and completed cards restore without offering another execution', () => {
  for (const executionStatus of [null, 'completed', 'success', 'needs_approval']) {
    const [message] = withProactive([], [item('report', { response: 'accepted', executionStatus, result: { output: '2450' } })]);
    const html = renderToStaticMarkup(createElement(ProactiveCard, { suggestion: message.suggestion }));
    assert.doesNotMatch(html, /Run once|<button/);
    assert.match(html, /2450/);
    if (executionStatus === null) assert.match(html, /Queued for the background worker/);
    if (executionStatus === 'success') assert.match(html, /Verified/);
  }
});

test('an authoritative pending proposal can be accepted or dismissed', () => {
  const [message] = withProactive([], [item('report')]);
  const html = renderToStaticMarkup(createElement(ProactiveCard, { suggestion: message.suggestion }));
  assert.match(html, /Run once/);
  assert.match(html, /Dismiss/);
  assert.deepEqual(withProactive([message], []), []);
});

test('refreshing the feed does not reorder chat when restored timestamps differ', () => {
  const messages = [
    { id: 'question', role: 'user', content: 'Question', timestamp: 30 },
    { id: 'answer', role: 'assistant', content: 'Answer', timestamp: 20 },
  ];
  assert.deepEqual(withoutProactive(withProactive(messages, [item('offline')])), messages);
});
