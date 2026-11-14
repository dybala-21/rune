import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/utils/modelConnection.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const exports = {};
vm.runInNewContext(source, { exports });
const { getModelSetupConnection } = exports;

test('only known missing setup for the exact active model blocks sending', () => {
  const activeModel = { provider: 'openai', model: 'shared', source: 'active' };
  const connection = { state: 'needs_setup', provider: 'openai', model: 'shared', reason: 'An API key is needed.', action: null };
  assert.equal(getModelSetupConnection({ activeModel, modelConnection: connection }), connection);
  for (const modelConnection of [undefined, { ...connection, state: 'configured' }, { ...connection, state: 'unknown' },
    { ...connection, provider: 'azure' }, { ...connection, model: 'previous-model' }]) {
    assert.equal(getModelSetupConnection({ activeModel, modelConnection }), null);
  }
  assert.equal(getModelSetupConnection(null), null);
  assert.equal(getModelSetupConnection(undefined), null);
});
