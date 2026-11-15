import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, Card;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ LoginCard: Card } = await server.ssrLoadModule('/src/components/LoginCard.tsx'));
});
after(async () => { await server?.close(); });

const render = (keepUrl = 'https://approve.example.com/keep/shop.example') => renderToStaticMarkup(createElement(Card, {
  sessionId: 'conv-1', site: 'shop.example', keepUrl, onDone: async () => {}, onDecline: async () => {},
}));

test('the owner signs in in the card, then keeps it on the approval page in a new tab', () => {
  const html = render();
  assert.match(html, /Sign in to shop\.example/);
  assert.match(html, /never sees your password/);
  assert.match(html, /<a href="https:\/\/approve\.example\.com\/keep\/shop\.example" target="_blank" rel="noopener noreferrer"[^>]*>Keep me signed in<\/a>/);
  assert.doesNotMatch(html, />Done<\/button>/, 'kept with the passkey, or not at all');
  assert.match(html, />Not now<\/button>/);
  assert.match(html, /Opening the page/, 'until the browser view arrives');
});

test('a site Rune already keeps needs only Done, and a link that is not https is never offered', () => {
  for (const url of ['', 'javascript:alert(1)']) {
    const html = render(url);
    assert.doesNotMatch(html, /<a /);
    assert.match(html, />Done<\/button>/);
  }
});
