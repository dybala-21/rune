import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as jsx from 'react/jsx-runtime';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/SettingsSidebar.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const flush = () => new Promise(resolve => setImmediate(resolve));

function mount(initial = {}, props = {}, variables = []) {
  const slots = [], effects = [], patches = [];
  let cursor = 0, first = true, holdPatch;
  let config = { proactiveEnabled: false, advisorEnabled: false, memoryTuning: {
    preset: 'balanced', policyMode: 'balanced', semanticLimit: 5, semanticMinScore: 0.3,
    uncertainSemanticLimit: 8, uncertainSemanticMinScore: 0.2, maxEpisodes: 10, contextMaxChars: 8000,
  }, ...initial };
  const useState = initial => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = initial;
    return [slots[index], value => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
  };
  const api = {
    fetchSkills: async () => ({ skills: [] }), fetchEnvVars: async () => ({ variables }),
    fetchChannels: async () => ({ channels: [] }), fetchCronJobs: async () => ({ jobs: [] }),
    fetchConfig: async () => structuredClone(config),
    patchConfig: async changes => {
      patches.push(structuredClone(changes));
      if (holdPatch) await holdPatch;
      config = { ...config, ...changes, memoryTuning: { ...config.memoryTuning, ...changes.memoryTuning } };
    },
  };
  const modules = {
    'react/jsx-runtime': jsx,
    react: { useState, useRef: value => useState({ current: value })[0], useCallback: fn => fn,
      useEffect: effect => { if (first) effects.push(effect); } },
    '../api': api, '../utils/toast': { toast: { error: message => assert.fail(message) } },
    './SettingsControls': { SettingsSection: 'section', SettingsItem: 'item', MemoryField: 'memory-field' },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name] ?? {},
    setInterval: () => 0, clearInterval() {} });
  const render = () => {
    cursor = 0;
    const tree = exports.SettingsSidebar(props);
    if (first) { first = false; effects.forEach(effect => effect()); }
    return tree;
  };
  function nodes(node) {
    if (Array.isArray(node)) return node.flatMap(nodes);
    return node?.props ? [node, ...nodes(node.props.children)] : [];
  }
  const text = node => typeof node === 'string' ? node
    : Array.isArray(node) ? node.map(text).join('') : node?.props ? text(node.props.children) : '';
  const find = predicate => {
    const node = nodes(render()).find(predicate);
    assert.ok(node, 'settings control is present');
    return node.props;
  };
  render();
  return {
    patches,
    control: find,
    contains: predicate => nodes(render()).some(predicate),
    content: () => text(render()),
    groupFor: title => text(nodes(render()).find(node => node.type === 'details'
      && nodes(node).some(child => child.props.title === title))?.props.children[0]),
    field: () => find(node => node.props.label === 'Semantic limit (1–20)'),
    toggle: label => find(node => node.props.role === 'switch' && text(node).includes(label)).onClick(),
    save: () => find(node => node.type === 'button' && text(node) === 'Save memory settings').onClick(),
    setRemoteLimit(value) { config.memoryTuning.semanticLimit = value; },
    pausePatch() { let release; holdPatch = new Promise(resolve => { release = resolve; }); return release; },
  };
}

for (const label of ['Suggestions', 'Advisor']) {
  test(`${label} preserves unsaved memory edits and refreshes again after saving`, async () => {
    const app = mount();
    await flush();
    app.field().onChange('9');
    await app.toggle(label);
    assert.equal(app.field().value, '9');
    await app.save();
    app.setRemoteLimit(7);
    await app.toggle(label);
    assert.equal(app.field().value, '7');
  });

  test(`${label} preserves memory edits made while its request is pending`, async () => {
    const app = mount();
    await flush();
    const release = app.pausePatch();
    const pending = app.toggle(label);
    app.field().onChange('9');
    release();
    await pending;
    assert.equal(app.field().value, '9');
  });
}


test('server-managed execution is read-only without removing provider key access', async () => {
  let opened = '';
  const app = mount({ executionEnvironment: { managed: true, backend: 'container', image: 'python:3.13-slim', allowNetwork: false } }, {
    onOpenProviderKeys: () => { opened = 'providers'; }, onOpenEnvPanel: () => { opened = 'environment'; },
  });
  await flush();
  assert.equal(app.contains(node => node.props.id === 'command-environment'), false);
  assert.match(app.content(), /Isolated container/);
  assert.match(app.content(), /Managed by the server operator/);
  app.control(node => node.props.className === 'settings-key-link').onClick();
  assert.equal(opened, 'providers');
  app.control(node => node.props.title === 'Environment variables').onClick();
  assert.equal(opened, 'environment');
});

test('execution selection requires an explicit editable environment', async () => {
  const editable = mount({ executionEnvironment: { managed: false, backend: 'local', allowNetwork: true } });
  const unknown = mount({ executionEnvironment: { backend: 'local', allowNetwork: true } });
  await flush();
  assert.equal(editable.control(node => node.props.id === 'command-environment').value, 'local');
  assert.match(editable.content(), /Connected engine/);
  assert.doesNotMatch(editable.content(), /Local computer/);
  assert.equal(unknown.contains(node => node.props.id === 'command-environment'), false);
  assert.match(unknown.content(), /Execution settings are not available/);
});

test('environment settings stay accessible under Advanced without showing internal values', async () => {
  let opened = false;
  const variables = [
    { key: 'RUNE_BROKER_HOME', maskedValue: '/srv/internal-broker', scope: 'process', category: 'other' },
    { key: 'RUNE_CLOUD_WORKER', maskedValue: '/srv/internal-worker', scope: 'process', category: 'other' },
  ];
  const app = mount({}, { onOpenEnvPanel: () => { opened = true; } }, variables);
  await flush();
  const link = app.control(node => node.props.title === 'Environment variables');
  assert.equal(link.subtitle, '2 variables');
  assert.match(app.groupFor('Environment variables'), /^Advanced/);
  for (const variable of variables) {
    assert.equal(app.contains(node => node.props.label === variable.key || node.props.detail === variable.maskedValue), false);
    assert.equal(app.content().includes(variable.key), false);
    assert.equal(app.content().includes(variable.maskedValue), false);
  }
  link.onClick();
  assert.equal(opened, true);
});

test('task, tool and memory headers keep their actions without duplicate entry rows', async () => {
  const opened = [];
  const app = mount({}, {
    onOpenCronPanel: () => opened.push('tasks'),
    onOpenMcpPanel: () => opened.push('tools'),
    onOpenMarkdownPanel: () => opened.push('memory'),
  });
  await flush();
  for (const title of ['Scheduled tasks', 'Connected tools', 'Memory and instructions']) {
    const section = app.control(node => node.props.title === title);
    assert.equal(section.children, undefined);
    section.onClick();
  }
  assert.deepEqual(opened, ['tasks', 'tools', 'memory']);
  for (const label of ['Manage schedules', 'Manage connections', 'Edit files']) {
    assert.equal(app.contains(node => node.props.label === label), false);
  }
  const unavailable = mount();
  await flush();
  assert.equal(unavailable.contains(node => node.props.title === 'Memory and instructions'), false);
});


test('memory tuning saves to hosted user settings while local projects keep project scope', async () => {
  for (const managed of [true, false, undefined]) {
    const app = mount({ executionEnvironment: { managed, backend: 'container', allowNetwork: false } });
    await flush();
    if (managed) {
      assert.match(app.content(), /Saves to this workspace’s user settings/);
      assert.doesNotMatch(app.content(), /Saved for this project in \.rune\/\.env/);
    } else {
      assert.match(app.content(), /Saved for this project in \.rune\/\.env/);
    }
    app.field().onChange('9');
    await app.save();
    assert.equal(app.patches.at(-1).memoryTuning.scope, managed ? 'user' : 'project');
    assert.equal(app.patches.at(-1).memoryTuning.semanticLimit, 9);
    assert.equal(app.field().value, '9');
  }
});
