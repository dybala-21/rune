import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, ChatPanel, WorkbenchPanel;
before(async () => {
  server = await createServer({
    configFile: false, esbuild: { jsx: 'automatic' },
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ WorkbenchPanel } = await server.ssrLoadModule('/src/components/WorkbenchPanel.tsx'));
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
