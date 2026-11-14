import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as jsx from 'react/jsx-runtime';
import { renderToStaticMarkup } from 'react-dom/server';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/EnvPanel.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const flush = () => new Promise(resolve => setImmediate(resolve));
const text = node => typeof node === 'string' ? node : Array.isArray(node) ? node.map(text).join('')
  : node?.props ? text(node.props.children) : '';
const nodes = node => Array.isArray(node) ? node.flatMap(nodes) : node?.props ? [node, ...nodes(node.props.children)] : [];
const variable = (key, scope = 'user', category = 'llm') => ({ key, scope, category, maskedValue: 'te****st', isSecret: true });

function mount(props = {}, initial = [], mutationResult) {
  const slots = [], effects = [], calls = [], traps = [];
  let cursor = 0, first = true, closed = 0, variables = initial, failDelete = false, failList = false, failSave = false;
  const trigger = { isConnected: true, focus() {} };
  const useState = initialValue => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = initialValue;
    return [slots[index], value => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  };
  const modules = {
    'react/jsx-runtime': jsx,
    react: { useState, useRef: value => useState({ current: value })[0], useCallback: fn => fn,
      useEffect: effect => { if (first) effects.push(effect); } },
    '../hooks/useFocusTrap': { useFocusTrap: (active, restoreTo) => { traps.push({ active, restoreTo }); return { current: null }; } },
    '../api': {
      async fetchEnvVars() {
        calls.push(['list']);
        if (failList) throw new Error('Could not refresh the key list');
        return { variables, paths: { user: '/user/.env', project: '/project/.env' } };
      },
      async setEnvVar(key, value, scope) {
        calls.push(['set', key, value, scope]);
        if (failSave) throw new Error('Could not write the key');
        variables = [variable(key, scope)];
        return mutationResult;
      },
      async unsetEnvVar(key, scope) {
        calls.push(['unset', key, scope]);
        if (failDelete) throw new Error('Could not remove the saved key');
        variables = variables.filter(item => item.key !== key);
        return mutationResult;
      },
    },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, Error, require: name => modules[name] ?? {},
    document: { activeElement: trigger }, requestAnimationFrame: fn => fn() });
  const render = () => {
    cursor = 0;
    traps.length = 0;
    const tree = exports.EnvPanel({ ...props, onClose: () => closed++ });
    if (first) { first = false; effects.forEach(effect => effect()); }
    return tree;
  };
  const find = predicate => {
    const found = nodes(render()).find(predicate);
    assert.ok(found, 'expected control is present');
    return found;
  };
  render();
  return {
    render, find, calls, traps, trigger,
    get closed() { return closed; },
    html: () => renderToStaticMarkup(render()),
    component: name => find(node => node.type?.name === name).props,
    click: label => find(node => node.type === 'button' && text(node).trim() === label).props.onClick({ currentTarget: trigger }),
    escape() {
      const event = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; } };
      render().props.onKeyDown(event);
      assert.equal(event.defaultPrevented, true);
      assert.equal(event.stopped, true);
    },
    failDelete() { failDelete = true; },
    failList() { failList = true; },
    failSave() { failSave = true; },
  };
}

test('provider entry filters only LLM settings and keeps the full environment view available', async () => {
  const saved = [variable('OPENAI_API_KEY'), variable('OLLAMA_HOST'), variable('TELEGRAM_TOKEN', 'project', 'telegram')];
  const app = mount({ initialMode: 'providers' }, saved);
  await flush();
  assert.match(app.html(), /OLLAMA_HOST/);
  assert.doesNotMatch(app.html(), /TELEGRAM_TOKEN/);
  app.click('All environment variables');
  assert.match(app.html(), /TELEGRAM_TOKEN/);
  const legacy = mount({}, saved);
  await flush();
  assert.match(legacy.html(), /TELEGRAM_TOKEN/);
  assert.equal(app.calls.length, 1);
});

test('provider shortcuts open blank values and save only through the existing environment API', async () => {
  for (const [label, key] of [['OpenAI', 'OPENAI_API_KEY'], ['Anthropic', 'ANTHROPIC_API_KEY'], ['Google Gemini', 'GEMINI_API_KEY'], ['xAI', 'XAI_API_KEY']]) {
    const app = mount({ initialMode: 'providers' });
    await flush();
    app.click(label + 'Add key');
    let editor = app.component('EnvEditor');
    assert.equal(editor.envKey, key);
    assert.equal(editor.value, '');
    assert.equal(app.calls.length, 1);
    editor.onValueChange('test-value');
    editor = app.component('EnvEditor');
    await editor.onSave();
    assert.deepEqual(app.calls, [['list'], ['set', key, 'test-value', 'user'], ['list']]);
    assert.equal(app.component('EnvDetailView').envVar.key, key);
  }
});

test('inherited provider keys are read-only and never point to the project file', async () => {
  for (const managed of [false, true]) {
    const app = mount({ initialMode: 'providers', managed }, [variable('OPENAI_API_KEY', 'process')]);
    await flush();
    app.click('OpenAIInherited key');
    const detail = app.component('EnvDetailView');
    const html = renderToStaticMarkup(app.find(node => node.type?.name === 'EnvDetailView'));
    assert.match(html, managed ? /Hosted server environment/ : /Rune process environment/);
    assert.match(html, /read-only/);
    assert.doesNotMatch(html, /\/project\/\.env|>Edit<|>Remove<|choose Edit/);
    detail.onEdit();
    assert.equal(app.component('EnvDetailView').envVar.scope, 'process');
    assert.equal(app.calls.length, 1);
  }
});

test('Google key aliases open their saved value instead of creating a duplicate', async () => {
  const app = mount({ initialMode: 'providers' }, [variable('GOOGLE_API_KEY')]);
  await flush();
  app.click('Google GeminiSaved key');
  assert.equal(app.component('EnvDetailView').envVar.key, 'GOOGLE_API_KEY');
});

test('Escape dismisses only the removal dialog and preserves its return-focus target', async () => {
  const app = mount({ initialMode: 'providers' }, [variable('OPENAI_API_KEY')]);
  await flush();
  app.click('OpenAISaved key');
  app.component('EnvDetailView').onDelete();
  assert.equal(app.find(node => node.props.role === 'dialog').props.inert, true);
  app.find(node => node.props.role === 'alertdialog');
  assert.deepEqual(app.traps.map(trap => trap.active), [false, true]);
  assert.equal(app.traps[1].restoreTo, app.trigger);
  app.escape();
  assert.equal(app.closed, 0);
  assert.equal(app.find(node => node.props.role === 'dialog').props.inert, false);
  assert.equal(nodes(app.render()).some(node => node.props.role === 'alertdialog'), false);
  app.escape();
  assert.equal(app.closed, 1);
  assert.equal(app.calls.length, 1);
});

test('a failed deletion stays visible inside the confirmation dialog', async () => {
  const app = mount({ initialMode: 'providers' }, [variable('OPENAI_API_KEY')]);
  await flush();
  app.click('OpenAISaved key');
  app.component('EnvDetailView').onDelete();
  app.failDelete();
  await app.click('Remove');
  assert.equal(app.find(node => node.props.role === 'alertdialog').props['aria-modal'], 'true');
  assert.match(text(app.find(node => node.props.role === 'alert')), /Could not remove/);
  assert.equal(app.closed, 0);
});

test('a manually entered inherited key cannot be overwritten through Add variable', async () => {
  const app = mount({}, [variable('OPENAI_API_KEY', 'process')]);
  await flush();
  app.click('Add variable');
  let editor = app.component('EnvEditor');
  editor.onKeyChange('OPENAI_API_KEY');
  editor.onValueChange('test-value');
  editor = app.component('EnvEditor');
  await editor.onSave();
  assert.equal(app.calls.length, 1);
  assert.match(app.html(), /This value is inherited/);
});

test('Escape cancels an unsaved key without closing the panel or retaining the value', async () => {
  const app = mount({ initialMode: 'providers' });
  await flush();
  app.click('OpenAIAdd key');
  app.component('EnvEditor').onValueChange('test-value');
  app.escape();
  assert.equal(app.closed, 0);
  app.click('OpenAIAdd key');
  assert.equal(app.component('EnvEditor').value, '');
  assert.equal(app.calls.length, 1);
});


test('saved activation states stay distinct and choosing a model is an explicit provider handoff', async () => {
  for (const [state, expected] of [
    ['next_run', /Applies to new runs/],
    ['restart_required', /Restart Rune’s engine/],
    ['unknown', /Activation timing is unknown/],
  ]) {
    const chosen = [];
    const app = mount({ initialMode: 'providers', onChooseModel: provider => chosen.push(provider) }, [],
      { activation: { state, reason: 'Activation detail' }, provider: 'openai' });
    await flush();
    app.click('OpenAIAdd key');
    app.component('EnvEditor').onValueChange('test-value');
    await app.component('EnvEditor').onSave();
    assert.match(app.html(), expected);
    assert.match(app.html(), /OPENAI_API_KEY saved/);
    assert.match(app.html(), /Activation detail/);
    assert.doesNotMatch(app.html(), /test-value/);
    if (state !== 'restart_required') assert.doesNotMatch(app.html(), /Restart/);
    assert.deepEqual(chosen, []);
    assert.equal(app.calls.length, 3);
    app.click('Choose model');
    assert.deepEqual(chosen, ['openai']);
    assert.equal(app.calls.length, 3);
  }
});

test('older engines report unknown activation and preserve the Gemini provider alias', async () => {
  const chosen = [];
  const app = mount({ onChooseModel: provider => chosen.push(provider), initialKey: 'GOOGLE_API_KEY' },
    [variable('GOOGLE_API_KEY')]);
  await flush();
  app.component('EnvDetailView').onEdit();
  app.component('EnvEditor').onValueChange('test-value');
  await app.component('EnvEditor').onSave();
  assert.match(app.html(), /Activation timing is unknown/);
  assert.doesNotMatch(app.html(), /Restart/);
  app.click('Choose model');
  assert.deepEqual(chosen, ['gemini']);
});

test('targeted setup opens a blank key form or an inherited key detail without a write', async () => {
  const missing = mount({ initialKey: 'ANTHROPIC_API_KEY' });
  await flush();
  assert.equal(missing.component('EnvEditor').envKey, 'ANTHROPIC_API_KEY');
  assert.equal(missing.component('EnvEditor').value, '');
  assert.deepEqual(missing.calls, [['list']]);
  const inherited = mount({ initialKey: 'GEMINI_API_KEY' }, [variable('GOOGLE_API_KEY', 'process')]);
  await flush();
  assert.equal(inherited.component('EnvDetailView').envVar.key, 'GOOGLE_API_KEY');
  assert.match(inherited.html(), /read-only/);
  assert.deepEqual(inherited.calls, [['list']]);
});

test('a saved key stays saved if refreshing its list fails', async () => {
  const app = mount({ initialMode: 'providers' }, [], { activation: { state: 'next_run', reason: '' } });
  await flush();
  app.click('OpenAIAdd key');
  app.component('EnvEditor').onValueChange('test-value');
  app.failList();
  await app.component('EnvEditor').onSave();
  assert.match(app.html(), /OPENAI_API_KEY saved/);
  assert.match(app.html(), /Could not refresh the key list/);
  assert.match(app.html(), /The saved variables could not be loaded/);
  assert.equal(app.find(node => node.type === 'button' && text(node).trim() === 'OpenAIAdd key').props.disabled, true);
  app.click('OpenAIAdd key');
  assert.equal(app.calls.length, 3);
  assert.doesNotMatch(app.html(), /test-value/);
});

test('a failed save keeps its draft and never offers the model handoff', async () => {
  const app = mount({ initialMode: 'providers', onChooseModel() {} });
  await flush();
  app.click('OpenAIAdd key');
  app.component('EnvEditor').onValueChange('test-value');
  app.failSave();
  await app.component('EnvEditor').onSave();
  assert.equal(app.component('EnvEditor').value, 'test-value');
  assert.match(text(app.find(node => node.props.role === 'alert')), /Could not write/);
  assert.doesNotMatch(app.html(), /Choose model|OPENAI_API_KEY saved/);
});

test('removing a key reports its activation and does not offer a model handoff', async () => {
  const app = mount({ initialKey: 'OPENAI_API_KEY', onChooseModel() {} }, [variable('OPENAI_API_KEY')],
    { activation: { state: 'next_run', reason: '' } });
  await flush();
  app.component('EnvDetailView').onDelete();
  await app.click('Remove');
  assert.match(app.html(), /OPENAI_API_KEY removed/);
  assert.match(app.html(), /Applies to new runs/);
  assert.doesNotMatch(app.html(), /Choose model|Restart/);
});

test('backdrop clicks keep an unsaved key editor open', async () => {
  const app = mount({ initialKey: 'OPENAI_API_KEY' });
  await flush();
  app.component('EnvEditor').onValueChange('test-value');
  const backdrop = {};
  app.render().props.onClick({ target: backdrop, currentTarget: backdrop });
  assert.equal(app.closed, 0);
  assert.equal(app.component('EnvEditor').value, 'test-value');
});


test('hosted workspace keys save in user settings and never offer a project scope', async () => {
  const app = mount({ managed: true, initialKey: 'OPENAI_API_KEY' });
  await flush();
  const editor = app.component('EnvEditor');
  assert.equal(editor.managed, true);
  const html = renderToStaticMarkup(app.find(node => node.type?.name === 'EnvEditor'));
  assert.match(html, /This workspace engine uses user settings/);
  assert.doesNotMatch(html, /\/project\/\.env|>project</);
  editor.onScopeChange('project');
  editor.onValueChange('test-value');
  await app.component('EnvEditor').onSave();
  assert.deepEqual(app.calls, [['list'], ['set', 'OPENAI_API_KEY', 'test-value', 'user'], ['list']]);
});
