import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, SearchActivity, SearchResults, ChatPanel, searchPreview, sourceUrl;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ SearchActivity, SearchResults } = await server.ssrLoadModule('/src/components/SearchActivity.tsx'));
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ searchPreview, sourceUrl } = await server.ssrLoadModule('/src/utils/searchActivity.ts'));
});
after(async () => { await server?.close(); });

const tool = (id, overrides = {}) => ({ id, toolName: 'web_search', args: { query: '공식 공지사항' }, timestamp: 1, run: 1,
  result: 'Original output', success: true, searchResults: { count: 1, items: [
    { title: '서버 점검 안내', url: 'https://example.org/news', description: 'Latest <strong>notice</strong>', date: 'October 24, 2026' },
  ] }, ...overrides });
const render = (component, props) => renderToStaticMarkup(createElement(component, props));

test('collapsed searches show queries and counts without a source list or raw logs', () => {
  const html = render(SearchActivity, { tools: [tool('a')] });
  assert.match(html, /Web search/);
  assert.match(html, /공식 공지사항/);
  assert.match(html, /1 result/);
  assert.doesNotMatch(html, /example.org|web_search|Original output|verified|<pre|<details[^>]* open/);
});

test('source links are safe and search markup stays text', () => {
  const html = render(SearchResults, { tool: tool('a', { searchResults: { count: 3, items: [
    { title: '<img src=x onerror=alert(1)>Title', url: 'https://example.org/a', description: '<strong>Excerpt</strong> &amp; more' },
    { title: 'Bad', url: 'javascript:alert(1)' },
    { title: 'Credentials', url: 'https://user:password@example.org/' },
  ] } }) });
  assert.match(html, /href="https:\/\/example.org\/a"/);
  assert.match(html, /rel="noopener noreferrer"/);
  assert.match(html, /Excerpt &amp; more/);
  assert.doesNotMatch(html, /<img|onerror|javascript:|user:password/);
  assert.equal(sourceUrl('data:text/html,bad'), null);
});

test('mixed results keep failure, pending and interrupted states visible', () => {
  const calls = [tool('a'), tool('b', { result: 'Timed out', success: false, searchResults: undefined }),
    tool('c', { result: undefined, success: undefined, searchResults: undefined })];
  const active = render(SearchActivity, { tools: calls, active: true });
  assert.match(active, /Searching the web/);
  assert.match(active, /1 failed/);
  assert.doesNotMatch(active, /3\/3|3 results|done/);
  const stopped = render(SearchActivity, { tools: calls });
  assert.match(stopped, /1 interrupted/);
  assert.doesNotMatch(stopped, /Searching the web|search-symbol-active/);
  assert.match(render(SearchActivity, { tools: [tool('u', { success: undefined })] }), /Status unavailable/);
});

test('an empty successful search is distinct from a failed search', () => {
  assert.match(render(SearchActivity, { tools: [tool('a', { searchResults: { count: 0, items: [] } })] }), /No search results/);
  assert.match(render(SearchActivity, { tools: [tool('a', { success: false, searchResults: undefined })] }), /Search failed/);
});

test('legacy numbered results remain usable without inventing a count for truncated output', () => {
  const result = 'Search excerpts are indexed snapshots, not live readings.\n\n1. Notice\n   https://example.org/news\n   Page date reported by search: October 24, 2026 (not necessarily the date of the facts)\n   The <strong>notice</strong>.\n';
  const call = tool('a', { result, searchResults: undefined });
  const preview = searchPreview(call);
  assert.equal(preview.count, 1);
  assert.equal(preview.items[0].description, 'The notice.');
  assert.equal(preview.items[0].date, 'October 24, 2026');
  assert.equal(searchPreview({ ...call, outputTruncated: true }).count, undefined);
  assert.equal(searchPreview({ ...call, result: result.replace('1. Notice', '7. Notice') }).count, undefined);
  assert.deepEqual(searchPreview({ ...call, result: 'Old unstructured response' }), { items: [], count: undefined });
});

test('search grouping preserves messages and run boundaries', () => {
  const props = { conversationKey: 'test', messages: [], thinkingBlocks: [], isRunning: false,
    activitySummary: null, delegateEvents: [], compactionEvents: [], currentStepInfo: null,
    pendingQuestion: null, pendingApproval: null, onRespondQuestion: async () => {}, onRespondApproval: async () => {} };
  const calls = [tool('a'), tool('b', { toolName: 'web.search', timestamp: 2 })];
  const count = html => (html.match(/class="search-activity"/g) ?? []).length;
  assert.equal(count(render(ChatPanel, { ...props, toolCalls: calls })), 1);
  assert.equal(count(render(ChatPanel, { ...props, toolCalls: [calls[0], { ...calls[1], run: 2 }] })), 2);
  assert.equal(count(render(ChatPanel, { ...props, toolCalls: calls,
    messages: [{ id: 'm', role: 'assistant', content: 'Next source.', timestamp: 1.5 }] })), 2);
});
