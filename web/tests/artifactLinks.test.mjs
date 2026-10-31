import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, artifactHref, Markdown;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ artifactHref } = await server.ssrLoadModule('/src/utils/artifactLinks.ts'));
  ({ Markdown } = await server.ssrLoadModule('/src/components/Markdown.tsx'));
});
after(async () => { await server?.close(); });

test('artifact links carry the displayed conversation and exact Unicode path', () => {
  const href = artifactHref('office/검토 보고서.pdf', 'saved-office');
  const url = new URL(href, 'http://localhost');
  assert.equal(url.pathname, '/api/v1/files/download');
  assert.equal(url.searchParams.get('sessionId'), 'saved-office');
  assert.equal(url.searchParams.get('path'), 'office/검토 보고서.pdf');
  const markup = renderToStaticMarkup(createElement(Markdown, { content: '[PDF](office/검토 보고서.pdf)', sessionId: 'saved-office' }));
  assert.ok(markup.includes('/api/v1/files/download?'));
  assert.ok(markup.includes('sessionId=saved-office'));
});

test('source URLs remain external, executable schemes and sessionless files do not link', () => {
  assert.equal(artifactHref('https://kospilab.com/'), 'https://kospilab.com/');
  assert.equal(artifactHref('javascript:alert(1)', 'office'), null);
  assert.equal(artifactHref('data:text/html,x', 'office'), null);
  assert.equal(artifactHref('//example.com', 'office'), null);
  assert.equal(artifactHref('report.pdf'), null);
});

test('formatted file links remain clickable in lists and table cells', () => {
  for (const content of ['- **[Word](office/briefing.docx)**: 제목 수정', '| 파일 |\n|---|\n| *[PDF](office/briefing.pdf)* |']) {
    const markup = renderToStaticMarkup(createElement(Markdown, { content, sessionId: 'office' }));
    assert.ok(markup.includes('<a href="/api/v1/files/download?'));
    assert.ok(!markup.includes('[Word]'));
    assert.ok(!markup.includes('[PDF]'));
  }
  const code = renderToStaticMarkup(createElement(Markdown, { content: '`[PDF](office/briefing.pdf)`', sessionId: 'office' }));
  assert.ok(!code.includes('<a '));
});


test('office outputs open as files while changes remain available', async () => {
  const { WorkbenchPanel } = await server.ssrLoadModule('/src/components/WorkbenchPanel.tsx');
  const { shouldOpenWorkbench, preferredWorkbenchTab } = await server.ssrLoadModule('/src/utils/workbench.ts');
  const calls = [{ id: 'doc', toolName: 'document_create', args: { path: 'office/보고서.docx' }, result: 'Created', success: true, timestamp: 1 }];
  assert.equal(shouldOpenWorkbench(calls, []), true);
  assert.equal(preferredWorkbenchTab(calls, []), 'outputs');
  const html = renderToStaticMarkup(createElement(WorkbenchPanel, { sessionId: 'office', toolCalls: calls,
    fileChanges: [{ id: 'c', path: 'helper.py', kind: 'modified', patch: '+x' }],
    isRunning: false, activitySummary: null, onClose() {} }));
  assert.match(html, /Files from this task/);
  assert.match(html, /sessionId=office/);
  assert.match(html, />Changes</);
  assert.doesNotMatch(html, /Checks passed|Content verified/);
});

test('failed, interrupted and unknown document writes do not become download links', async () => {
  const { workbenchOutputs } = await server.ssrLoadModule('/src/utils/workbench.ts');
  const { OutputsPane } = await server.ssrLoadModule('/src/components/OutputsPane.tsx');
  for (const state of [{ success: false, result: 'error' }, {}, { result: 'unknown' }]) {
    const calls = [{ id: 'doc', toolName: 'document_create', args: { path: 'report.pdf' }, timestamp: 1, ...state }];
    const html = renderToStaticMarkup(createElement(OutputsPane, { outputs: workbenchOutputs(calls, 'office'), isRunning: false }));
    assert.doesNotMatch(html, /href=|Creating|Created/);
  }
  const html = renderToStaticMarkup(createElement(OutputsPane, { isRunning: false, outputs: [
    { path: 'javascript:alert(1)', state: 'created', sessionId: 'office' },
    { path: 'https://example.com/unsafe', state: 'created', sessionId: 'office' },
  ] }));
  assert.doesNotMatch(html, /href=/);
});


test('bundle files keep their session and unpublished downloads remain unconfirmed', async () => {
  const { workbenchOutputs } = await server.ssrLoadModule('/src/utils/workbench.ts');
  const outputs = workbenchOutputs([], 'office', { artifactReceipts: [
    { kind: 'document_bundle', artifacts: [{ path: 'bundle/report.docx', sha256: 'hash' }], checks: { native_content: 'pass' } },
    { scope: 'download', path: '/Desktop/report.pdf', sessionId: 'native', verified: false },
  ] });
  assert.deepEqual(outputs, [
    { path: 'bundle/report.docx', sessionId: 'office', state: 'created' },
    { path: '/Desktop/report.pdf', sessionId: 'native', state: 'unknown' },
  ]);
});
