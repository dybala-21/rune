import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as jsx from 'react/jsx-runtime';
import { renderToStaticMarkup } from 'react-dom/server';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/LocalModelSettings.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const nodes = node => Array.isArray(node) ? node.flatMap(nodes) : node?.props ? [node, ...nodes(node.props.children)] : [];
const text = node => typeof node === 'string' ? node : Array.isArray(node) ? node.map(text).join('')
  : node?.props ? text(node.props.children) : '';
const initialState = () => ({ ollama: { baseUrl: 'http://localhost:11434', envKey: 'OLLAMA_API_BASE', scope: null }, endpoints: {} });
const endpoint = () => ({ baseUrl: 'https://models.example.com/v1', models: ['example'], visionModels: [],
  toolMode: 'native', keyName: 'RUNE_ENDPOINT_GPU_API_KEY', hasKey: false, keyScope: null, selfHosted: false });

function mount(initial = initialState(), mutationResult = { activation: { state: 'next_run', reason: '' } }) {
  const slots = [], effects = [], calls = [];
  let cursor = 0, first = true, state = initial, failRefresh = false;
  const useState = value => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = value;
    return [slots[index], next => { slots[index] = typeof next === 'function' ? next(slots[index]) : next; }];
  };
  const modules = {
    'react/jsx-runtime': jsx,
    react: { useState, useRef: value => useState({ current: value })[0],
      useEffect: effect => { if (first) effects.push(effect); } },
    '../api': {
      async patchConfig(patch) {
        calls.push(['config', patch]);
        state = { ...state, endpoints: Object.fromEntries(Object.entries(patch.endpoints).map(([name, entry]) =>
          [name, { ...endpoint(), ...entry }])) };
      },
      async setEnvVar(key, value, scope) { calls.push(['env', key, value, scope]); return mutationResult; },
      async testEndpoint(name) {
        calls.push(['test', name]);
        return { reachable: true, models: ['example'], tools: 'native', detail: '' };
      },
    },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, Error, URL, require: name => modules[name] ?? {} });
  const render = () => {
    cursor = 0;
    const tree = exports.LocalModelSettings({ state, onSaved: async () => {
      calls.push(['refresh']);
      if (failRefresh) throw new Error('Unavailable');
    } });
    if (first) { first = false; effects.forEach(effect => effect()); }
    return tree;
  };
  const find = predicate => {
    const node = nodes(render()).find(predicate);
    assert.ok(node, 'expected control is present');
    return node;
  };
  render();
  return {
    calls, render,
    html: () => renderToStaticMarkup(render()),
    button: label => find(node => node.type === 'button' && text(node).trim() === label).props,
    field: id => find(node => node.props.id === id).props,
    change: (id, value) => find(node => node.props.id === id).props.onChange({ target: { value } }),
    failRefresh() { failRefresh = true; },
  };
}

test('Ollama-only saves describe the next step without offering a nonexistent test', async () => {
  const app = mount();
  app.change('ollama-base', 'http://localhost:11435');
  await app.button('Save').onClick();
  assert.match(app.html(), /Saved\. Applies to new runs/);
  assert.match(app.html(), /Open the model picker/);
  assert.match(app.html(), /Ollama connection has not been tested/);
  assert.doesNotMatch(app.html(), /Use Test|>Test</);
  assert.deepEqual(app.calls.map(call => call[0]), ['config', 'env', 'refresh']);
});

test('older engines leave activation unknown and refresh failures retain saved feedback', async () => {
  const app = mount(initialState(), {});
  app.change('ollama-base', 'http://localhost:11435');
  app.failRefresh();
  await app.button('Save').onClick();
  assert.match(app.html(), /Saved\. Activation timing is unknown/);
  assert.match(app.html(), /Settings were saved, but the updated settings could not be loaded/);
  assert.doesNotMatch(app.html(), /Restart/);
});

test('endpoint tests describe usage and only run after an explicit click', async () => {
  const app = mount({ ...initialState(), endpoints: { gpu: endpoint() } });
  assert.match(app.html(), /Test lists models and sends one short model request\. Usage charges may apply/);
  assert.equal(app.button('Test')['aria-describedby'], 'model-server-test-help');
  assert.equal(app.calls.length, 0);
  app.change('endpoint-models-0', 'example,other');
  assert.equal(app.button('Test').disabled, true);
  await app.button('Save').onClick();
  assert.match(app.html(), /Use Test to check an endpoint/);
  assert.deepEqual(app.calls.map(call => call[0]), ['config', 'refresh']);
  await app.button('Test').onClick();
  assert.equal(app.calls.at(-1)[0], 'test');
  assert.match(app.html(), /Tool calls work/);
  app.change('endpoint-url-0', 'https://other.example.com/v1');
  assert.doesNotMatch(app.html(), /Tool calls work/);
});

test('endpoint key saves honor restart metadata and clear the entered key', async () => {
  const app = mount({ ...initialState(), endpoints: { gpu: endpoint() } },
    { activation: { state: 'restart_required', reason: 'The running engine caches this setting.' } });
  app.change('endpoint-key-0', 'test-value');
  await app.button('Save').onClick();
  assert.match(app.html(), /Restart the connected Rune engine/);
  assert.equal(app.field('endpoint-key-0').value, '');
  assert.doesNotMatch(app.html(), /test-value/);
  assert.deepEqual(app.calls.map(call => call[0]), ['config', 'env', 'refresh']);
});

test('empty Ollama edits cannot be reported as saved and inherited values stay read-only', async () => {
  const app = mount();
  app.change('ollama-base', '');
  await app.button('Save').onClick();
  assert.match(app.html(), /Enter an http\(s\) URL for the Ollama server/);
  assert.equal(app.calls.length, 0);
  const inherited = mount({ ...initialState(), ollama: { ...initialState().ollama, scope: 'process' } });
  assert.equal(inherited.field('ollama-base').disabled, true);
  assert.match(inherited.html(), /address is inherited/);
});
