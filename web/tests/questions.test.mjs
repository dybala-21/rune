import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, Card;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ InlineQuestionCard: Card } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
});
after(async () => { await server?.close(); });

const options = [{ label: '개념 설명' }, { label: '시스템 확인', description: '현재 연결 확인' }];
const tool = { id: 'row-1', callId: 'call-1', toolName: 'ask_user', timestamp: 0, args: { question: '어떤 정보?', options } };
const pending = { id: 'q1', callId: 'call-1', question: '어떤 정보?', options };
const render = (toolCall = tool, pendingQuestion = pending) => renderToStaticMarkup(createElement(Card, { toolCall, pendingQuestion, onRespond: async () => {} }));

test('a delivered question enables its options and free text input', () => {
  const html = render();
  assert.match(html, /현재 연결 확인/);
  assert.match(html, /placeholder="Type your answer/);
  assert.doesNotMatch(html, /Answered/);
  assert.equal((html.match(/<button[^>]*disabled/g) ?? []).length, 1); // empty free-text submit only
});

test('transport failure is an error, never an answered question', () => {
  const html = render({ ...tool, result: 'Failed to get user response: Encoding objects of type AskUserParams is unsupported', success: false }, null);
  assert.match(html, />Failed</);
  assert.match(html, /Error:/);
  assert.doesNotMatch(html, /Answered|Answer:/);
  assert.equal((html.match(/<button[^>]*disabled/g) ?? []).length, 2);
});

test('only a successful user response is labelled answered', () => {
  assert.match(render({ ...tool, result: 'User responded: "시스템 확인"', success: true }, null), /Answered/);
  assert.doesNotMatch(render({ ...tool, result: 'User did not provide an answer (skipped).', success: true }, null), /Answered/);
});

test('a question cannot borrow another tool call’s pending response', () => {
  const html = render(tool, { ...pending, callId: 'other-call', question: '다른 질문' });
  assert.match(html, /어떤 정보/);
  assert.doesNotMatch(html, /다른 질문/);
  assert.equal((html.match(/<button[^>]*disabled/g) ?? []).length, 2);
});
