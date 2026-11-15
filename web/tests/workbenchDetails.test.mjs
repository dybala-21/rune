import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, ChatPanel, WorkbenchPanel, ProgressPane, shouldOpenWorkbench, preferredWorkbenchTab, desktopAttention;
before(async () => {
  server = await createServer({
    configFile: false, esbuild: { jsx: 'automatic' },
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ WorkbenchPanel } = await server.ssrLoadModule('/src/components/WorkbenchPanel.tsx'));
  ({ ProgressPane } = await server.ssrLoadModule('/src/components/ProgressPane.tsx'));
  ({ shouldOpenWorkbench, preferredWorkbenchTab, desktopAttention } = await server.ssrLoadModule('/src/utils/workbench.ts'));
});
after(async () => { await server?.close(); });

const summary = { success: true, totalToolCalls: 15, filesRead: 1, filesWritten: 2,
  bashExecutions: 3, webSearches: 4, browserActions: 5, totalDurationMs: 62000 };
const toolCalls = [{ id: 'search', toolName: 'web_search', args: { query: 'Release notes' },
  timestamp: 1, result: 'Found a source', success: true }];
const panel = props => renderToStaticMarkup(createElement(WorkbenchPanel, {
  sessionId: 'details', toolCalls, isRunning: false, activitySummary: summary, onClose() {}, ...props,
}));

test('Progress keeps every reported count and duration in collapsed Run details, including history', () => {
  for (const historical of [false, true]) {
    const html = panel({ historical });
    assert.match(html, /<details class="workbench-run-details"><summary>Run details<\/summary>/);
    for (const [label, value] of [['Tool calls', 15], ['Files read', 1], ['File writes', 2],
      ['Shell commands', 3], ['Web lookups', 4], ['Browser actions', 5], ['Duration', '1m2s']]) {
      assert.ok(html.includes(`<dt>${label}</dt><dd>${value}</dd>`), `${label} remains available`);
    }
  }
  assert.doesNotMatch(panel({ isRunning: true }), /workbench-run-details/);
});

test('aggregate counts do not replace the existing verification or failure result', () => {
  const html = panel({ trust: { reason: 'error: timeout', completionStatus: 'failed' } });
  const [progress, details] = html.split('<details class="workbench-run-details">');
  assert.match(progress, /Run failed/);
  assert.ok(details);
  assert.doesNotMatch(details, /Verified|Checks passed|Completed|var\(--success\)/);
});

test('chat keeps its answer and inline activity without a duplicate totals card', () => {
  const html = renderToStaticMarkup(createElement(ChatPanel, {
    conversationKey: 'details', messages: [{ id: 'answer', role: 'assistant', content: 'The release is ready.', timestamp: 2 }],
    toolCalls, thinkingBlocks: [], isRunning: false, activitySummary: summary,
    delegateEvents: [], compactionEvents: [], currentStepInfo: null, pendingQuestion: null, pendingApproval: null,
    onRespondQuestion: async () => {}, onRespondApproval: async () => {},
  }));
  assert.match(html, /The release is ready\./);
  assert.match(html, /Web search/);
  assert.doesNotMatch(html, />Tools<|>Web<|1m2s|Run details/);
});


test('lookups stay in chat while code, documents and Computer still open their work panel', () => {
  const lookups = ['web_search', 'web_fetch', 'file_read'].map((toolName, i) => ({
    id: String(i), toolName, args: {}, timestamp: i,
  }));
  assert.equal(shouldOpenWorkbench(lookups, []), false);
  assert.equal(preferredWorkbenchTab(lookups, []), 'progress');
  for (const [toolName, tab] of [['file_edit', 'diff'], ['bash_execute', 'activity'],
    ['document_create', 'outputs'], ['browser_navigate', 'computer'], ['desktop_observe', 'computer']]) {
    const calls = [...lookups, { id: 'work', toolName, args: {}, timestamp: 4 }];
    assert.equal(shouldOpenWorkbench(calls, []), true, toolName);
    assert.equal(preferredWorkbenchTab(calls, []), tab, toolName);
  }
  assert.equal(shouldOpenWorkbench(lookups, [{ path: 'src/app.ts' }]), true);
});

const progress = calls => renderToStaticMarkup(createElement(ProgressPane, {
  toolCalls: calls, mode: 'research', isRunning: false, currentStep: null,
  trust: null, activitySummary: null, orchestration: null,
}));

test('recorded activity never implies that missing results or unconfirmed checks succeeded', () => {
  const base = toolCalls[0];
  for (const extra of [{ result: undefined, success: undefined }, { success: undefined },
    { checkStatus: 'inconclusive' }, { checkStatus: 'fail' }]) {
    const html = progress([{ ...base, ...extra }]);
    assert.match(html, /aria-label="Activity recorded"/);
    assert.doesNotMatch(html, /aria-label="Succeeded"/);
  }
  assert.match(progress([base]), /aria-label="Succeeded"/);
});

test('source details preserve complete queries and safe page links without displaying URL paths', () => {
  const query = 'Compare annual releases with their compatibility and support windows';
  const url = 'https://example.org/releases/%EC%83%88%EB%A1%9C%EC%9A%B4?version=1';
  const html = progress([
    { ...toolCalls[0], args: { query } },
    { ...toolCalls[0], id: 'page', toolName: 'web_fetch', args: { url } },
    { ...toolCalls[0], id: 'unsafe', toolName: 'web_fetch', args: { url: 'javascript:alert(1)' } },
    { ...toolCalls[0], id: 'credentials', toolName: 'web_fetch', args: { url: 'https://user:secret@example.org/' } },
  ]);
  assert.match(html, /<details(?![^>]*\bopen\b)[^>]*><summary[^>]*>Sources · 2<\/summary>/);
  assert.ok(html.includes(`href="${url}"`));
  assert.ok(html.includes(`aria-label="${url}"`));
  assert.ok(html.includes(query));
  const visible = html.replace(/<[^>]*>/g, '');
  assert.match(visible, /example\.org/);
  assert.doesNotMatch(visible, /%EC%|javascript:|secret/);
});


test('desktop attention excludes completed calls and pending calls from another run', () => {
  const pending = { id: 'connect', toolName: 'desktop_connect', args: {}, timestamp: 1, runId: 'previous' };
  assert.equal(desktopAttention([pending], true), 'connection');
  assert.equal(desktopAttention([pending], true, 'previous'), 'connection');
  assert.equal(desktopAttention([pending], true, 'current'), null);
  assert.equal(desktopAttention([{ ...pending, completedAt: 2 }], true, 'previous'), null);
  assert.equal(desktopAttention([{ ...pending, result: 'Connected' }], true, 'previous'), null);
  assert.equal(desktopAttention([pending], false, 'previous'), null);
  const action = { ...pending, id: 'action', toolName: 'desktop_act', runId: 'current' };
  assert.equal(desktopAttention([pending, action], true, 'current'), 'action');
});
