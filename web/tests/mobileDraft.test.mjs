import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

let server, appendImportedText, installMobileDraft;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ appendImportedText, installMobileDraft } = await server.ssrLoadModule('/src/utils/mobileDraft.ts'));
});
after(async () => { await server?.close(); });

test('native import preserves the existing draft and treats code as literal text', () => {
  const host = {};
  let draft = 'Summarize this conversation';
  const dispose = installMobileDraft(host, () => draft, value => { draft = value; });
  assert.equal(host.runeImportText('민수: hello 🐱\n</script>${alert(1)}'), true);
  assert.equal(draft, 'Summarize this conversation\n\n민수: hello 🐱\n</script>${alert(1)}');
  assert.equal(host.runeImportText({ text: 'wrong protocol' }), false);
  dispose();
  assert.equal(host.runeImportText, undefined);
});

test('empty, binary, and oversized imports never modify the composer', () => {
  for (const value of ['', ' \n', null, 12, 'x\0y', 'x'.repeat(100_001)]) {
    assert.equal(appendImportedText('', value), null);
  }
  assert.equal(appendImportedText('x'.repeat(99_999), 'y'), null);
  assert.equal(appendImportedText('', 'x'.repeat(100_000)).length, 100_000);
});

test('old composer cleanup does not remove the next conversation receiver', () => {
  const host = {};
  const old = installMobileDraft(host, () => '', () => assert.fail('old composer was used'));
  let draft = '';
  const current = installMobileDraft(host, () => draft, value => { draft = value; });
  old();
  assert.equal(host.runeImportText('new conversation'), true);
  assert.equal(draft, 'new conversation');
  current();
});
