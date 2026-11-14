import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const sources = Object.fromEntries(['sessionContext', 'composerStore'].map(name => [name,
  ts.transpileModule(fs.readFileSync(new URL(`../src/utils/${name}.ts`, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText]));
const alice = 'a'.repeat(64), bob = 'b'.repeat(64);
function client(values = new Map()) {
  let now = 1_000_000, blocked = false;
  const storage = {
    getItem(key) { if (blocked) throw new Error('storage blocked'); return values.get(key) ?? null; },
    setItem(key, value) { if (blocked) throw new Error('storage blocked'); values.set(key, value); },
    removeItem(key) { if (blocked) throw new Error('storage blocked'); values.delete(key); },
  };
  const modules = {};
  for (const name of ['sessionContext', 'composerStore']) {
    const exports = {};
    vm.runInNewContext(sources[name], { exports, require: () => modules.sessionContext,
      sessionStorage: storage, Date: { now: () => now } });
    modules[name] = exports;
  }
  return { ...modules.sessionContext, ...modules.composerStore, values,
    advance(ms) { now += ms; }, block() { blocked = true; } };
}

test('unknown ownership cannot read or write drafts; only local keeps legacy keys', () => {
  const c = client();
  assert.equal(c.storageKey('history'), null);
  assert.equal(c.accountScope(), null);
  c.saveComposerDraft('chat', 'secret', false);
  assert.equal(c.values.size, 0);
  assert.equal(c.bindStorageScope(undefined), false);
  assert.equal(c.readComposerDraft('chat'), null);
  const local = client();
  local.bindStorageScope('local');
  assert.equal(local.storageKey('history'), 'history');
  assert.equal(local.accountScope(), null);
  assert.equal(local.isLocalSession(), true);
  for (const invalid of ['alice', '../bob', 'a'.repeat(63), {}, 1]) {
    const other = client();
    other.bindStorageScope(invalid);
    assert.equal(other.storageKey('history'), null);
  }
});

test('same-tab sign-in restores text only for the same authenticated account', () => {
  const first = client();
  first.bindStorageScope(alice);
  assert.equal(first.accountScope(), alice);
  assert.equal(first.isLocalSession(), false);
  first.saveComposerDraft('chat', 'Finish the quarterly report', true);
  const reload = client(first.values);
  reload.bindStorageScope(alice);
  assert.equal(reload.readComposerDraft('chat').text, 'Finish the quarterly report');
  assert.equal(reload.readComposerDraft('chat').hadAttachments, true);
  assert.equal(reload.readComposerDraft('other-chat'), null);
  const switched = client(first.values);
  switched.bindStorageScope(bob);
  assert.equal(switched.readComposerDraft('chat'), null);
  assert.equal(switched.storageKey('history'), `history:${bob}`);
});

test('an account change cannot redirect old callbacks into the new account', () => {
  const c = client();
  assert.equal(c.bindStorageScope(alice), false);
  assert.equal(c.bindStorageScope(alice), false);
  assert.equal(c.bindStorageScope(bob), true);
  c.saveComposerDraft('chat', 'Old account draft', false);
  assert.equal(c.storageKey('history'), `history:${alice}`);
  const next = client(c.values);
  next.bindStorageScope(bob);
  assert.equal(next.readComposerDraft('chat'), null);
});

test('drafts expire, clear after send, and stay bounded across conversations', () => {
  const c = client();
  c.bindStorageScope(alice);
  c.saveComposerDraft('first', 'old', false);
  c.advance(24 * 60 * 60 * 1000);
  assert.equal(c.readComposerDraft('first'), null);
  for (let i = 0; i < 10; i++) c.saveComposerDraft(`chat-${i}`, `draft-${i}`, false);
  assert.equal(c.readComposerDraft('chat-0'), null);
  assert.equal(c.readComposerDraft('chat-2').text, 'draft-2');
  c.saveComposerDraft('chat-9', '', false);
  assert.equal(c.readComposerDraft('chat-9'), null);
  c.saveComposerDraft('oversized', 'x'.repeat(100_001), false);
  assert.equal(c.readComposerDraft('oversized'), null);
});

test('malformed data and unavailable storage never block the composer', () => {
  const c = client();
  c.bindStorageScope(alice);
  c.values.set(c.storageKey('rune.composer.v1'), '{broken');
  assert.equal(c.readComposerDraft('chat'), null);
  c.saveComposerDraft('chat', 'valid', false);
  assert.equal(c.readComposerDraft('chat').text, 'valid');
  c.block();
  assert.equal(c.readComposerDraft('chat'), null);
  assert.doesNotThrow(() => c.saveComposerDraft('chat', 'updated', false));
});
