import assert from 'node:assert/strict';
import fs from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/memoryView.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const view = await import(`data:text/javascript,${encodeURIComponent(source)}`);

test('a fact from a connector names the connector first, then the conversation', () => {
  const chips = view.originChips({ kind: '', connectors: ['gmail'], conversations: ['c1'] }, 'learned');
  assert.deepEqual(chips, [
    { text: 'from gmail', tone: 'connector' },
    { text: 'from a conversation', tone: 'conversation' },
  ]);
});

test('a fact with no recorded origin says so instead of guessing', () => {
  assert.deepEqual(view.originChips(undefined, 'learned'), [{ text: 'origin not recorded', tone: 'unknown' }]);
  assert.deepEqual(view.originChips(null, 'memory'), [{ text: 'written by you', tone: 'you' }]);
  assert.deepEqual(view.originChips({ kind: 'rule', connectors: [], conversations: [] }, 'learned'),
    [{ text: 'learned from how tasks ran', tone: 'learned' }]);
});

test('lines stay under their section in the order sections first appear', () => {
  const line = (section, text) => ({ section, line: text, key: '', origin: null });
  const groups = view.bySection([line('Notes', 'a'), line('Preferences', 'b'), line('Notes', 'c')]);
  assert.deepEqual(groups.map(g => [g.section, g.lines.map(l => l.line)]), [['Notes', ['a', 'c']], ['Preferences', ['b']]]);
});

test('forgetting a connector reports what went, or that nothing was there', () => {
  assert.equal(view.forgottenSummary({ facts: 2, episodes: 1, dailyEntries: 0 }), 'Forgot 2 facts, 1 task record.');
  assert.equal(view.forgottenSummary({ facts: 0, episodes: 0, dailyEntries: 0 }), 'Nothing was remembered from it.');
  assert.equal(view.hiddenReason('connector_forgotten:gmail'), 'forgotten with gmail');
});
