import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import * as jsx from 'react/jsx-runtime';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/PickerPopover.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function mount() {
  const state = [], deps = [], cleanups = [], effects = [], listeners = new Map();
  let cursor = 0, inert = false;
  const document = { body: {}, activeElement: null,
    addEventListener: (type, listener) => listeners.set(type, listener),
    removeEventListener: type => listeners.delete(type),
  };
  const anchor = {
    closest: () => inert ? {} : null,
    contains: target => target === anchor,
    focus: () => { document.activeElement = anchor; },
    getBoundingClientRect: () => ({ top: 500, bottom: 530, left: 20 }),
  };
  const panel = {
    contains: target => target === panel,
    focus: () => { document.activeElement = panel; },
    querySelectorAll: () => [], querySelector: () => null,
  };
  const useState = initial => {
    const i = cursor++;
    if (!(i in state)) state[i] = initial;
    return [state[i], value => { state[i] = typeof value === 'function' ? value(state[i]) : value; }];
  };
  const effect = (fn, next) => {
    const i = cursor++;
    if (!next || !deps[i] || next.some((value, index) => value !== deps[i][index])) {
      effects.push(() => { cleanups[i]?.(); cleanups[i] = fn(); });
    }
    deps[i] = next;
  };
  const modules = {
    'react/jsx-runtime': jsx, 'react-dom': { createPortal: node => node },
    react: { useState, useRef: value => useState({ current: value })[0], useId: () => 'test-picker',
      useEffect: effect, useLayoutEffect: effect },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name], document,
    window: { innerWidth: 1000, innerHeight: 700, addEventListener() {}, removeEventListener() {} },
  });
  const render = () => {
    cursor = 0;
    const tree = exports.PickerPopover({ label: 'Models', trigger: 'Model', children: () => 'Choices' });
    const [button, content] = tree.props.children;
    button.props.ref.current = anchor;
    if (content) content.props.ref.current = panel;
    effects.splice(0).forEach(run => run());
    return { button: button.props, menu: content?.props };
  };
  return {
    render, anchor, document,
    open() { render().button.onClick(); return render(); },
    focusOutside() { const target = {}; document.activeElement = target; listeners.get('focusin')({ target }); return target; },
    cover() { inert = true; render(); },
  };
}

function key(value, extra = {}) {
  return { key: value, prevented: false, stopped: false,
    preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...extra };
}

test('moving focus outside closes the menu without stealing the new focus', () => {
  const picker = mount();
  picker.open();
  const target = picker.focusOutside();
  assert.equal(picker.render().menu, undefined);
  assert.equal(picker.document.activeElement, target);
});

test('covering the composer with an overlay removes its portal menu', () => {
  const picker = mount();
  picker.open();
  picker.cover();
  assert.equal(picker.render().menu, undefined);
});

test('the command shortcut closes the menu and leaves a stable return target', () => {
  const picker = mount();
  const menu = picker.open().menu;
  const event = key('k', { metaKey: true });
  menu.onKeyDown(event);
  assert.equal(event.prevented, false);
  assert.equal(event.stopped, false);
  assert.equal(picker.document.activeElement, picker.anchor);
  assert.equal(picker.render().menu, undefined);
});

test('menu activation and escape do not reach background command handlers', () => {
  const picker = mount();
  const { button, menu } = picker.open();
  for (const target of [button, menu]) {
    for (const value of ['Enter', ' ']) {
      const event = key(value);
      target.onKeyDown(event);
      assert.equal(event.stopped, true);
      assert.equal(event.prevented, false, 'native button activation remains available');
    }
  }
  const arrow = key('ArrowDown');
  button.onKeyDown(arrow);
  assert.equal(arrow.stopped, true);
  const escape = key('Escape');
  menu.onKeyDown(escape);
  assert.equal(escape.stopped, true);
  assert.equal(escape.prevented, true);
  assert.equal(picker.render().menu, undefined);
});


test('search input keeps text-navigation keys while suppressing background shortcuts', () => {
  const picker = mount();
  const menu = picker.open().menu;
  for (const value of ['Home', 'End', 'ArrowLeft', 'ArrowRight']) {
    const event = key(value, { target: { tagName: 'INPUT' } });
    menu.onKeyDown(event);
    assert.equal(event.prevented, false);
    assert.equal(event.stopped, true);
  }
});
