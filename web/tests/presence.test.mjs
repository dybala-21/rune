import assert from 'node:assert/strict';
import { before, after, beforeEach, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

let server, presence;
const page = { hidden: false };
let sent, announced;

before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  presence = await server.ssrLoadModule('/src/utils/presence.ts');
});
after(async () => { await server?.close(); });

beforeEach(() => {
  page.hidden = false;
  sent = [];
  announced = [];
  presence.resetPresenceForTests({
    page,
    send: async (path, init) => { sent.push([path, init.method]); return answer('running'); },
    announce: status => announced.push(status),
  });
});

const answer = status => new Response(null, { headers: status ? { 'x-rune-machine': status } : {} });

test('nothing changes against a local daemon', () => {
  presence.noteResponse(answer(null));
  assert.deepEqual(presence.presenceHeaders(1_000_000), {});
  presence.noteInput(1_000_000);
  assert.deepEqual(sent, [], 'no presence pings to a daemon that does not know them');
  assert.equal(presence.machineStatus(), null);
});

test('behind the gateway, requests nobody caused are marked background', () => {
  presence.noteResponse(answer('running'));
  assert.deepEqual(presence.presenceHeaders(1_000_000), { 'X-Rune-Background': '1' });
  presence.noteInput(1_000_000);
  assert.deepEqual(presence.presenceHeaders(1_000_000 + presence.RECENT_INPUT_MS - 1), {}, 'right after input it counts');
  assert.deepEqual(presence.presenceHeaders(1_000_000 + presence.RECENT_INPUT_MS), { 'X-Rune-Background': '1' });
});

test('a hidden page is never someone using it', () => {
  presence.noteResponse(answer('running'));
  presence.noteInput(1_000_000);
  page.hidden = true;
  assert.deepEqual(presence.presenceHeaders(1_000_001), { 'X-Rune-Background': '1' });
});

test('input sends a presence ping at most once a minute', async () => {
  presence.noteResponse(answer('suspended'));
  presence.noteInput(1_000_000);
  presence.noteInput(1_000_000 + 1_000);
  presence.noteInput(1_000_000 + presence.PRESENCE_EVERY_MS - 1);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.deepEqual(sent, [['/cloud/presence', 'POST']]);
  presence.noteInput(1_000_000 + presence.PRESENCE_EVERY_MS);
  assert.equal(sent.length, 2);
});

test('the machine status is announced when it changes', async () => {
  presence.noteResponse(answer('suspended'));
  presence.noteResponse(answer('suspended'));
  presence.noteInput(1_000_000);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.deepEqual(announced, ['suspended', 'running'], 'the ping woke it');
  assert.equal(presence.machineStatus(), 'running');
});

test('a failed ping is retried on the next input', async () => {
  presence.noteResponse(answer('suspended'));
  presence.resetPresenceForTests({ send: async (path) => { sent.push([path]); throw new TypeError('offline'); } });
  presence.noteResponse(answer('suspended'));
  presence.noteInput(1_000_000);
  await new Promise(resolve => setTimeout(resolve, 0));
  presence.noteInput(1_000_000 + 1);
  assert.equal(sent.length, 2);
});

test('no ping while the page is hidden', () => {
  presence.noteResponse(answer('running'));
  page.hidden = true;
  presence.noteInput(1_000_000);
  assert.deepEqual(sent, []);
});
