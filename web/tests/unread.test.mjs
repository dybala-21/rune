import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

let server, unreadModule;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  unreadModule = await server.ssrLoadModule('/src/utils/unread.ts');
});
after(async () => { await server?.close(); });

function memoryStorage(initial = {}) {
  const values = { ...initial };
  return { getItem: key => values[key] ?? null, setItem: (key, value) => { values[key] = value; }, values };
}

test('messages RUNE kept in "From RUNE" since it was last opened are new, and only there', () => {
  const { INBOX_ID, loadSeen, markSeen, unread } = unreadModule;
  const storage = memoryStorage();
  let seen = loadSeen(storage);
  assert.equal(unread(seen, INBOX_ID, 1), 1, 'never opened: its message is new');
  seen = markSeen(seen, INBOX_ID, 1, storage);
  assert.equal(unread(seen, INBOX_ID, 1), 0);
  assert.equal(unread(seen, INBOX_ID, 3), 2, 'two came after it was opened');
  assert.equal(unread(loadSeen(storage), INBOX_ID, 3), 2, 'kept across reloads');
  assert.equal(unread(seen, 'web-chat-1', 9), 0, "the owner's own chats were read as they happened");
});

test('storage that cannot be read or written leaves the list working', () => {
  const { INBOX_ID, loadSeen, markSeen, unread } = unreadModule;
  const broken = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } };
  assert.deepEqual(loadSeen(broken), {});
  assert.deepEqual(loadSeen(memoryStorage({ 'rune:web:seen-messages': 'not json' })), {});
  assert.deepEqual(loadSeen(memoryStorage({ 'rune:web:seen-messages': '[1]' })), {});
  const seen = markSeen({}, INBOX_ID, 2, broken);
  assert.equal(unread(seen, INBOX_ID, 2), 0, 'still counted as read for this page');
});

test('"From RUNE" missing from a read list counts as none, so its first message is new', () => {
  const { INBOX_ID, inboxMessages } = unreadModule;
  const chat = id => ({ id, turnCount: 4 });
  assert.equal(inboxMessages([chat('web-1')], false, false), 0, 'read, and nothing from RUNE yet');
  assert.equal(inboxMessages([{ id: INBOX_ID, turnCount: 2 }, chat('web-1')], false, false), 2);
  assert.equal(inboxMessages([], true, false), undefined, 'still loading');
  assert.equal(inboxMessages([chat('web-1')], false, true), undefined, 'a search shows only what matches');
  const pages = Array.from({ length: 51 }, (_, i) => chat(`web-${i}`));
  assert.equal(inboxMessages(pages, false, false), undefined, 'older pages are not read again');
});
