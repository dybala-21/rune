import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, ToolCallCard, ToolActivity, ToolBody, ToolOutput, ChatPanel, presentTool;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ ToolCallCard } = await server.ssrLoadModule('/src/components/ToolCallCard.tsx'));
  ({ ToolActivity, ToolBody } = await server.ssrLoadModule('/src/components/ToolActivity.tsx'));
  ({ ToolOutput } = await server.ssrLoadModule('/src/components/ToolDetails.tsx'));
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ presentTool } = await server.ssrLoadModule('/src/utils/toolActivity.ts'));
});
after(async () => { await server?.close(); });

const call = (toolName, args = {}, extra = {}) => ({ id: 'a', toolName, args, result: 'Original output', success: true, timestamp: 1, run: 1, ...extra });
const render = (component, props) => renderToStaticMarkup(createElement(component, props));
const card = (toolCall, active = false) => render(ToolCallCard, { toolCall, active });

test('file and document summaries identify their targets without rendering payloads', () => {
  for (const toolName of ['file_read', 'file_write', 'file_edit', 'document_create']) {
    const html = card(call(toolName, { path: '/workspace/분기 보고서.xlsx', content: 'PRIVATE DOCUMENT CONTENT', replace: 'PRIVATE REPLACEMENT' }));
    assert.match(html, /분기 보고서.xlsx/);
    assert.doesNotMatch(html, /PRIVATE|Original output|<pre|<details[^>]* open/);
  }
  assert.equal(presentTool(call('file_read', { path: 'C:\\work\\report.csv' })).target, 'report.csv');
  assert.equal(presentTool(call('file_search', { path: '/src', pattern: 'invoice' })).target, 'invoice');
});

test('page and app actions show a target, never the text entered into a field', () => {
  const page = call('web_fetch', { url: 'https://example.org/news?id=1' });
  assert.match(card(page), /Read web page|example.org/);
  assert.match(render(ToolBody, { tool: page }), /href="https:\/\/example.org\/news\?id=1"/);
  for (const url of ['javascript:alert(1)', 'https://user:secret@example.org']) {
    assert.doesNotMatch(render(ToolBody, { tool: call('web_fetch', { url }) }), /href=|secret|javascript:/);
  }
  const browser = card(call('browser_act', { action: 'type', selector: '#message', value: 'PRIVATE INPUT' }));
  assert.match(browser, /Enter text|#message/);
  assert.doesNotMatch(browser, /PRIVATE INPUT/);
  assert.match(card(call('desktop_act', { action: 'key', key: 's', modifiers: ['command'] })), /Press key|command \+ s/);
});

test('failed checks, unknown results, interrupted runs and live calls stay distinct', () => {
  const base = call('bash_execute', { command: 'npm test' });
  assert.match(card({ ...base, success: false }), /Failed/);
  const checked = card({ ...base, checkStatus: 'fail' });
  assert.match(checked, /Checks failed/);
  assert.doesNotMatch(checked, /Succeeded/);
  assert.match(card({ ...base, checkStatus: 'inconclusive' }), /Checks unconfirmed/);
  assert.match(card({ ...base, success: undefined }), /Status unavailable/);
  const pending = { ...base, result: undefined, success: undefined };
  assert.match(card(pending, true), /Running/);
  const stopped = card(pending);
  assert.match(stopped, /Interrupted/);
  assert.doesNotMatch(stopped, /Running|spinner/);
  assert.match(card({ ...pending, completedAt: 2 }, true), /Status unavailable/);
  assert.doesNotMatch(render(ToolBody, { tool: { ...pending, success: false }, active: true }), /Waiting for the tool/);
  assert.match(card(call('desktop_connect', {}, { result: undefined, success: undefined }), true), /Waiting for you/);
});

test('group totals retain failures and interruptions instead of promoting every result to success', () => {
  const tools = [call('file_read'), call('file_read', {}, { id: 'b', success: false }),
    call('file_read', {}, { id: 'c', result: undefined, success: undefined })];
  const html = render(ToolActivity, { tools });
  assert.match(html, /1 failed · 1 interrupted · 1 succeeded/);
  assert.doesNotMatch(html, /3 succeeded|3\/3|verified/);
});

test('long output is bounded, available on demand, and rendered as text', () => {
  const html = render(ToolOutput, { text: '<script>bad()</script>\n' + 'x'.repeat(20_000) + 'OUTPUT TAIL' });
  assert.match(html, /Show full output|&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>|OUTPUT TAIL/);
  assert.ok(html.length < 10_000);
  assert.match(render(ToolBody, { tool: call('file_read', {}, { outputTruncated: true }) }), /Only part of the output was saved/);
});

const chatProps = { conversationKey: 'test', messages: [], thinkingBlocks: [], isRunning: false,
  activitySummary: null, delegateEvents: [], compactionEvents: [], currentStepInfo: null,
  pendingQuestion: null, pendingApproval: null, onRespondQuestion: async () => {}, onRespondApproval: async () => {} };

test('grouping preserves run boundaries and leaves every question actionable', () => {
  const tools = [1, 2, 3].map(i => call('file_read', { path: `file${i}` }, { id: String(i), timestamp: i }));
  const count = html => (html.match(/class="tool-activity tool-state-/g) ?? []).length;
  assert.equal(count(render(ChatPanel, { ...chatProps, toolCalls: tools })), 1);
  assert.equal(count(render(ChatPanel, { ...chatProps, toolCalls: tools.map((t, i) => ({ ...t, runId: String(i) })) })), 3);
  const questions = [1, 2, 3].map(i => call(i === 3 ? 'ask.user' : 'ask_user', { question: `Question ${i}`, options: [{ label: `Option ${i}` }] }, { id: String(i), callId: String(i), timestamp: i, result: undefined, success: undefined }));
  const html = render(ChatPanel, { ...chatProps, toolCalls: questions, pendingQuestion: { id: 'q', callId: '3', question: 'Question 3', options: [{ label: 'Option 3' }] } });
  assert.equal(count(html), 0);
  assert.match(html, /Question 1/);
  assert.match(html, /Question 2/);
  assert.match(html, /Question 3/);
  assert.doesNotMatch(html, /<button[^>]*disabled[^>]*>\s*(?:<[^>]+>)*Option 3/);
});

test('an earlier run without a result does not spin when a later run is active', () => {
  const tools = [call('file_read', { path: 'old.txt' }, { result: undefined, runId: 'old' }),
    call('bash_execute', { command: 'npm test' }, { id: 'b', result: undefined, success: undefined, timestamp: 2, runId: 'new' })];
  const html = render(ChatPanel, { ...chatProps, isRunning: true, toolCalls: tools });
  assert.match(html, /Interrupted/);
  assert.match(html, /Running/);
});

test('extension tools receive the same disclosure without exposing arbitrary arguments', () => {
  const html = card(call('plugin_custom_action', { token: 'PRIVATE TOKEN', body: 'PRIVATE BODY' }));
  assert.match(html, /Plugin custom action/);
  assert.doesNotMatch(html, /PRIVATE/);
});
