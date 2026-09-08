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
