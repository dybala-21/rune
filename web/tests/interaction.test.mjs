import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, latestTurn, retryAttachments, attachmentMime, ChatPanel, InputArea, retryRequestId, updateDelivery;
before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ retryRequestId, updateDelivery } = await server.ssrLoadModule('/src/utils/messageDelivery.ts'));
  ({ latestTurn, retryAttachments } = await server.ssrLoadModule('/src/utils/retry.ts'));
  ({ attachmentMime } = await server.ssrLoadModule('/src/utils/attachments.ts'));
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ InputArea } = await server.ssrLoadModule('/src/components/InputArea.tsx'));
});
after(async () => { await server?.close(); });
const message = (id, role, extra = {}) => ({ id, role, content: id, timestamp: Number(id), ...extra });
const chat = messages => renderToStaticMarkup(createElement(ChatPanel, {
  conversationKey: 'test', messages, toolCalls: [], thinkingBlocks: [], isRunning: false,
  onRegenerate() {}, activitySummary: null, delegateEvents: [], compactionEvents: [],
  currentStepInfo: null, pendingQuestion: null, pendingApproval: null,
  onRespondQuestion() {}, onRespondApproval() {},
}));

test('a failed new request owns retry; the previous answer does not', () => {
  const messages = [message('1', 'user'), message('2', 'assistant'), message('3', 'user'),
    message('4', 'system', { trust: { reason: 'error: timeout', completionStatus: 'failed' } })];
  assert.equal(latestTurn(messages).user.id, '3');
  assert.equal(latestTurn(messages).retryAnchor.id, '4');
  assert.equal(latestTurn(messages).assistant, undefined);
  const html = chat(messages);
  assert.match(html, /Retry request/);
  assert.doesNotMatch(html, /Regenerate/);
});

test('a failed send has a retry action even without a trust card', () => {
  const html = chat([message('1', 'user'), message('2', 'system', { level: 'error', content: 'Failed to send' })]);
  assert.match(html, /Retry request/);
});

test('successful latest answer keeps regenerate and stale errors cannot take its turn', () => {
  const messages = [message('1', 'user'), message('2', 'system', { level: 'error' }),
    message('3', 'user'), message('4', 'assistant')];
  assert.equal(latestTurn(messages).retryAnchor.id, '4');
  assert.match(chat(messages), /Regenerate/);
});

test('retry reuses all attachment bytes or rejects missing saved content', () => {
  const user = message('1', 'user', { attachments: [
    { name: 'table.csv', mimeType: 'text/csv', dataUrl: 'data:text/csv;base64,YSwxCg==' },
    { name: 'image.png', mimeType: 'image/png', dataUrl: 'data:image/png;base64,aW1hZ2U=' },
  ] });
  assert.deepEqual(retryAttachments(user).map(a => a.data), ['YSwxCg==', 'aW1hZ2U=']);
  delete user.attachments[1].dataUrl;
  assert.throws(() => retryAttachments(user), /Reattach image.png/);
});

test('running composer permits drafts, attachments and stop, but never send', () => {
  const html = renderToStaticMarkup(createElement(InputArea, { isRunning: true, disabled: false, onSend() {}, onAbort() {} }));
  assert.doesNotMatch(html, /readonly|aria-label="Send message"/i);
  assert.match(html, /aria-label="Stop generation"/);
  assert.match(html, /aria-label="Attach file"/);
  assert.match(html, /Send your draft when it finishes/);
});

test('office files are recognized when the browser omits their MIME type', () => {
  assert.equal(attachmentMime({ name: '매출.CSV', type: '' }), 'text/csv');
  assert.match(attachmentMime({ name: '계획.docx', type: '' }), /wordprocessingml/);
  assert.equal(attachmentMime({ name: 'program.exe', type: '' }), undefined);
});


test('saved attachment references survive retry without embedding the original file', () => {
  const attachments = [{ name: 'notes.txt', mimeType: 'text/plain', ref: 'owned-file' }];
  assert.deepEqual(retryAttachments(message('1', 'user', { attachments })), attachments);
  const user = message('1', 'user', { content: 'read', requestId: 'request', delivery: 'unknown', attachments });
  assert.equal(retryRequestId(user, 'read', attachments), 'request');
  assert.equal(retryRequestId(user, 'changed', attachments), undefined);
  assert.equal(retryRequestId({ ...user, delivery: 'accepted' }, 'read', attachments), undefined);
  assert.equal(retryRequestId(user, 'read', [{ ...attachments[0], ref: 'another-file' }]), undefined);
});


test('a late receipt removes only the delivery warning for its own request', () => {
  const messages = [message('user', 'user', { requestId: 'accepted', delivery: 'unknown' }),
    message('delivery-accepted', 'system', { level: 'error' }),
    message('delivery-other', 'system', { level: 'error' })];
  const restored = updateDelivery(messages, 'accepted', 'accepted');
  assert.deepEqual(restored.map(m => m.id), ['user', 'delivery-other']);
  assert.equal(restored[0].delivery, 'accepted');
});
