import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as jsx from 'react/jsx-runtime';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { test } from 'node:test';

const compile = name => ts.transpileModule(fs.readFileSync(new URL(`../src/components/${name}.tsx`, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const load = (source, modules) => {
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name] });
  return exports;
};
const reasoning = load(compile('ReasoningPicker'), {
  'react/jsx-runtime': jsx, './PickerPopover': { SelectedCheck: () => null },
});
const source = compile('ModelPicker');
const flush = () => new Promise(resolve => setImmediate(resolve));
const initial = {
  activeModel: { provider: 'first', model: 'shared', source: 'active' },
  reasoningSupported: true, reasoningEffort: 'high', reasoningOptions: ['low', 'high'],
};

function mount() {
  const slots = [], dependencies = [], pendingEffects = [], calls = [], errors = [];
  let cursor = 0, config = initial, rejectNext = false, catalogs = 0, fullTree;
  const useState = initial => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [slots[index], value => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  };
  const modules = {
    'react/jsx-runtime': jsx,
    react: {
      useState, useRef: value => useState({ current: value })[0],
      useEffect: (effect, deps) => {
        const index = cursor++;
        if (!dependencies[index] || deps.some((value, i) => value !== dependencies[index][i])) pendingEffects.push(effect);
        dependencies[index] = deps;
      },
    },
    '../api': {
      fetchModels: async () => { catalogs++; return { first: ['shared', 'other'], second: ['shared'] }; },
      setActiveModel: async (provider, model) => { calls.push({ kind: 'model', provider, model }); return { provider, model }; },
      setReasoningEffort: async (effort, model) => {
        calls.push({ kind: 'reasoning', effort, provider: model.provider, model: model.model });
        if (rejectNext) { rejectNext = false; throw new Error('Unavailable'); }
        return { reasoningEffort: effort || null };
      },
    },
    '../utils/toast': { toast: { error: message => errors.push(message) } },
    './ProviderMark': { ProviderMark: 'provider' },
    './PickerPopover': { PickerPopover: 'popover', SelectedCheck: 'check' },
    './ReasoningPicker': { ...reasoning, ReasoningPicker: 'reasoning' },
  };
  const { ModelPicker } = load(source, modules);
  const render = () => {
    cursor = 0;
    const tree = ModelPicker({ config, onConfigure() {} });
    fullTree = tree;
    pendingEffects.splice(0).forEach(effect => effect());
    return tree.props.children[0];
  };
  const nodes = node => Array.isArray(node) ? node.flatMap(nodes)
    : node?.props ? [node, ...nodes(node.props.children)] : [];
  const text = node => typeof node === 'string' ? node : Array.isArray(node) ? node.map(text).join('')
    : node?.props ? text(node.props.children) : '';
  const menu = () => nodes(render().props.children(() => {}));
  const find = label => {
    const node = menu().find(node => node.type === 'button' && text(node).includes(label));
    assert.ok(node, `${label} is available`);
    return node.props;
  };
  render();
  return {
    calls, errors, render, find,
    catalogs: () => catalogs,
    notice() { render(); return text(nodes(fullTree).find(node => node.props.className === 'model-setup-notice')); },
    search(value) { menu().find(node => node.type === 'input').props.onChange({ target: { value } }); },
    choices: () => menu().filter(node => node.props.role === 'menuitemradio').map(text),
    async open() { render().props.onOpen(); await flush(); },
    options() { return menu().find(node => node.type === 'reasoning')?.props; },
    update(next) { config = next; render(); },
    rejectReasoning() { rejectNext = true; },
  };
}

test('reasoning waits for capabilities of the selected provider and model', async () => {
  const app = mount();
  await app.open();
  app.find('All providers').onClick();
  app.find('second').onClick();
  app.find('shared').onClick();
  await flush();
  assert.equal(app.find('Reasoning').disabled, true);
  assert.doesNotMatch(app.render().props.label, /reasoning: High/);
  app.update({ ...initial, activeModel: { provider: 'second', model: 'shared', source: 'active' },
    reasoningEffort: null, reasoningOptions: ['medium'], reasoningBudgets: { medium: 4096 } });
  app.find('Reasoning').onClick();
  const options = app.options();
  assert.deepEqual(options.options, ['medium']);
  assert.equal(options.budgets.medium, 4096);
  options.onPick('high');
  await flush();
  assert.equal(app.calls.filter(call => call.kind === 'reasoning').length, 0);
  options.onPick('medium');
  await flush();
  assert.deepEqual(app.calls.at(-1), { kind: 'reasoning', effort: 'medium', provider: 'second', model: 'shared' });
  assert.match(app.render().props.label, /reasoning: Medium/);
});

test('failed reasoning saves preserve the selection and Auto clears only the active model', async () => {
  const app = mount();
  await app.open();
  app.find('Reasoning').onClick();
  app.rejectReasoning();
  app.options().onPick('low');
  await flush();
  assert.match(app.render().props.label, /reasoning: High/);
  assert.equal(app.errors.length, 1);
  app.options().onPick(null);
  await flush();
  assert.deepEqual(app.calls.at(-1), { kind: 'reasoning', effort: '', provider: 'first', model: 'shared' });
  assert.match(app.render().props.label, /reasoning: Auto/);
});

test('unsupported models omit reasoning and the composer retains one picker', () => {
  const app = mount();
  app.update({ ...initial, reasoningSupported: false, reasoningOptions: [] });
  assert.doesNotMatch(app.render().props.label, /reasoning/);
  const { ModelControls } = load(compile('ModelControls'), {
    'react/jsx-runtime': jsx, './ModelControls.css': {},
    './ModelPicker': { ModelPicker: () => createElement('button', { 'aria-label': 'Model settings' }) },
  });
  const html = renderToStaticMarkup(createElement(ModelControls, { config: initial }));
  assert.equal((html.match(/<button/g) ?? []).length, 1);
});

test('reasoning keeps Auto, None and model token budgets distinct', () => {
  const html = renderToStaticMarkup(createElement(reasoning.ReasoningPicker, {
    effort: 'none', options: ['none', 'high'], budgets: { high: 8192 }, disabled: false, onPick() {},
  }));
  assert.match(html, /Auto/);
  assert.match(html, /Use this model’s default/);
  assert.match(html, /Without additional reasoning/);
  assert.match(html, /8,192 thinking tokens/);
  assert.equal((html.match(/aria-checked="true"/g) ?? []).length, 1);
  assert.doesNotMatch(html, /aria-haspopup/);
});


test('selecting the current model does not save it again', async () => {
  const app = mount();
  await app.open();
  app.find('shared').onClick();
  await flush();
  assert.equal(app.calls.length, 0);
  app.find('other').onClick();
  await flush();
  assert.deepEqual(app.calls, [{ kind: 'model', provider: 'first', model: 'other' }]);
});


test('model search spans providers and filters locally without another catalog request', async () => {
  const app = mount();
  await app.open();
  app.search('second shared');
  assert.deepEqual(app.choices(), ['sharedsecond']);
  assert.equal(app.catalogs(), 1);
  app.search('missing-model');
  assert.deepEqual(app.choices(), []);
  assert.equal(app.catalogs(), 1);
  app.search('second shared');
  app.find('shared').onClick();
  await flush();
  assert.deepEqual(app.calls.at(-1), { kind: 'model', provider: 'second', model: 'shared' });
});

test('setup notice follows the displayed model while a selection is awaiting config', async () => {
  const app = mount();
  const connection = { state: 'needs_setup', provider: 'first', model: 'shared', reason: 'An API key is needed.',
    action: { target: 'environment', label: 'Add API key', key: 'TEST_KEY' } };
  app.update({ ...initial, modelConnection: connection });
  assert.match(app.notice(), /Model needs setup/);
  await app.open();
  app.find('other').onClick();
  await flush();
  assert.equal(app.notice(), '', 'old provider setup must not accompany the new model');
  app.update({ ...initial, activeModel: { provider: 'first', model: 'other', source: 'active' },
    modelConnection: { ...connection, model: 'other', state: 'configured' } });
  assert.equal(app.notice(), '');
});
