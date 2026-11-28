import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, notice, InboxNoticeSwitch;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  notice = await server.ssrLoadModule('/src/utils/inboxNotice.ts');
  ({ InboxNoticeSwitch } = await server.ssrLoadModule('/src/components/InboxNoticeSwitch.tsx'));
});
after(async () => { await server?.close(); delete globalThis.Notification; });

const bytes = text => new TextEncoder().encode(text).length;

test('a kept message shows as its first line over the rest on one line', () => {
  const message = 'Price watch\nA price you asked me to watch went down.\n\nAirPods Pro 3 (www.apple.com)\n399,000 KRW → 369,000 KRW';
  assert.deepEqual(notice.noticeText(message), {
    title: 'Price watch',
    body: 'A price you asked me to watch went down. AirPods Pro 3 (www.apple.com) 399,000 KRW → 369,000 KRW',
  });
  assert.deepEqual(notice.noticeText('Morning briefing'), { title: 'Morning briefing', body: '' });
});

test('a long message is cut between characters, within what macOS keeps', () => {
  const { title, body } = notice.noticeText(`가격 알림 ${'가'.repeat(40)}\n${'내려간 가격 '.repeat(60)}`);
  assert.ok(bytes(title) + bytes(body) <= 250, `${bytes(title) + bytes(body)} bytes`);
  assert.ok(title.endsWith('…') && body.endsWith('…'));
  assert.ok(!body.includes('�'));
  const emoji = notice.noticeText(`t\n${'𝄞'.repeat(200)}`).body;
  assert.ok([...emoji.slice(0, -1)].every(char => char === '𝄞'), 'no half of a two-unit character');
});

test('the switch asks from a click, and says when notifications are on or blocked', () => {
  const render = () => renderToStaticMarkup(createElement(InboxNoticeSwitch));
  delete globalThis.Notification;
  assert.equal(render(), '', 'no notifications here: nothing to switch');
  globalThis.Notification = { permission: 'default' };
  assert.match(render(), /<button[^>]*>Notify me on this computer<\/button>/);
  globalThis.Notification = { permission: 'granted' };
  assert.match(render(), />Notifications on</);
  globalThis.Notification = { permission: 'denied' };
  assert.match(render(), /blocked for this page/);
});
