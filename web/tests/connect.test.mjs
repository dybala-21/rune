import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, Card;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ ConnectAccountCard: Card } = await server.ssrLoadModule('/src/components/ConnectAccountCard.tsx'));
});
after(async () => { await server?.close(); });

const render = (url = 'https://approve.example.com/connect/gmail-read') => renderToStaticMarkup(createElement(Card, {
  url, label: 'Connect Gmail', detail: "Lets Rune read your mail. It can't send any.", onDecline: async () => {},
}));

test('one button opens the approval page to connect the account, in a new tab', () => {
  const html = render();
  assert.match(html, /<a href="https:\/\/approve\.example\.com\/connect\/gmail-read" target="_blank" rel="noopener noreferrer"[^>]*>Connect Gmail<\/a>/);
  assert.match(html, /read your mail/);
  assert.match(html, /carries on by itself/);
  assert.match(html, /<button[^>]*>Not now<\/button>/);
  assert.doesNotMatch(html, /Approval Required|Allow Once/);
});

test('a link that is not https is never offered', () => {
  const html = render('javascript:alert(1)');
  assert.doesNotMatch(html, /<a /);
  assert.match(html, /Not now/);
});
