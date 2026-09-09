import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const root = fileURLToPath(new URL('..', import.meta.url));
const home = mkdtempSync(resolve(tmpdir(), 'rune-trust-contract-'));
const gate = last_verdict => ({ has_check: true, last_verdict, verdict_counts: { pass: 1 }, last_evidence: 'check output' });
const cases = [
  { name: 'table differences override passing command checks', trace: { mech_check: 'pass', table_acceptance: { required: true, status: 'fail', contracts: [], results: [], unverified: [] } }, title: 'Checks failed', tone: 'warning', ok: false, card: true },
  { name: 'web lookup completed without a checker', trace: {}, title: 'Completed', tone: 'neutral', ok: true, card: false },
  { name: 'ordinary chat', trace: { verification: { required: false, status: 'unverified' } }, title: 'Completed', tone: 'neutral', ok: true, card: false },
  { name: 'task check passed', trace: { evidence_gate: gate('pass') }, title: 'Checks passed', tone: 'success', ok: true, card: true },
  { name: 'passing check without historical counts', trace: { evidence_gate: { has_check: true, last_verdict: 'pass' } }, title: 'Checks passed', tone: 'success', ok: true, card: true },
  { name: 'latest check failed after an earlier pass', trace: { evidence_gate: gate('fail') }, title: 'Checks failed', tone: 'warning', ok: false, card: true },
  { name: 'checker could not produce a verdict', trace: { evidence_gate: gate('skip') }, title: 'Verification inconclusive', tone: 'neutral', ok: true, card: true },
  { name: 'check was not applicable', trace: { evidence_gate: { has_check: false, last_verdict: 'skip' } }, title: 'Completed', tone: 'neutral', ok: true, card: false },
  { name: 'code edited after an earlier pass', trace: { verification: { required: true, status: 'unverified' }, mech_check: 'pass', tests_passed_after_edit: false }, title: 'Verification required', tone: 'warning', ok: false, card: true },
  { name: 'fresh tests passed', trace: { verification: { required: true, status: 'pass' }, tests_passed_after_edit: true }, title: 'Tests passing', tone: 'success', ok: true, card: true },
  { name: 'tests passed but task check failed', trace: { verification: { required: true, status: 'pass' }, tests_passed_after_edit: true, evidence_gate: gate('fail') }, title: 'Checks failed', tone: 'warning', ok: false, card: true },
  { name: 'tool limit reached after passing checks', trace: { tool_budget_exhausted: true, evidence_gate: gate('pass') }, title: 'Incomplete', tone: 'warning', ok: false, card: true },
  { name: 'token budget ended a web task', trace: { reason: 'token_budget_exhausted' }, title: 'Incomplete', tone: 'warning', ok: false, card: true },
  { name: 'repeated gate blocks allow escalation', trace: { reason: 'max_gate_blocked' }, title: 'Incomplete', tone: 'warning', ok: false, card: true, escalate: true },
  { name: 'advisor abort allows escalation', trace: { reason: 'advisor_abort' }, title: 'Incomplete', tone: 'warning', ok: false, card: true, escalate: true },
  { name: 'unresolved output checks do not imply a missing response or a saved file', trace: { reason: 'completed_gate_warnings' }, title: 'Review needed', tone: 'warning', ok: false, card: true },
  { name: 'unresolved requirement includes the concrete blocker', trace: { reason: 'completed_gate_warnings', completion_check: { name: 'Output requirements', detail: 'Comparison table is missing.' } }, title: 'Review needed', tone: 'warning', ok: false, card: true },
  { name: 'failed check keeps its warning under a general output warning', trace: { reason: 'completed_gate_warnings', evidence_gate: gate('fail') }, title: 'Checks failed', tone: 'warning', ok: false, card: true },
  { name: 'tool budget takes precedence over output warning', trace: { reason: 'completed_gate_warnings', tool_budget_exhausted: true }, title: 'Incomplete', tone: 'warning', ok: false, card: true },
  { name: 'runtime failure is not a failed check', trace: { reason: 'error: connection lost' }, title: 'Run failed', tone: 'danger', ok: false, card: true },
  { name: 'cancellation after passing checks', trace: { reason: 'cancelled', evidence_gate: gate('pass') }, title: 'Stopped', tone: 'neutral', ok: false, card: true },
  { name: 'missing completion status', trace: { reason: '' }, title: 'Status unavailable', tone: 'neutral', ok: null, card: false },
  { name: 'document checks do not attest the entire task', trace: { artifact_receipts: [{
    kind: 'document_bundle', revision: 'r1', source_sha256: 'abc', artifacts: [{ path: '/report.xlsx', sha256: 'def' }],
    checks: { native_content: 'pass', source_metrics: 'pass', visual_layout: 'not_performed', task_acceptance: 'not_performed' },
  }] }, title: 'Completed', tone: 'neutral', ok: true, card: true },
  { name: 'cancelled run retains published document checks', trace: { reason: 'cancelled', artifact_receipts: [{
    kind: 'document_bundle', revision: 'r2', source_sha256: 'abc', artifacts: [{ path: '/published.xlsx', sha256: 'def' }],
    checks: { native_content: 'pass', source_metrics: 'pass', visual_layout: 'not_performed', task_acceptance: 'not_performed' },
  }] }, title: 'Stopped', tone: 'neutral', ok: false, card: true },
];

let server, describeTrust, checkEvidence, computeRunVerdict, TrustCard, ProgressPane, WorkbenchPanel, payloads, abortedMessage, upsertRunMessage;
before(async () => {
  const result = spawnSync(process.env.RUNE_TEST_PYTHON || resolve(root, '../.venv/bin/python'), ['-c', `
import json, sys
from types import SimpleNamespace
from rune.api.server import build_trust_payload
print(json.dumps([build_trust_payload(SimpleNamespace(**trace)) for trace in json.load(sys.stdin)]))
`], {
    cwd: resolve(root, '..'), encoding: 'utf8', env: { ...process.env, RUNE_HOME: home },
    input: JSON.stringify(cases.map(c => ({ reason: 'completed', ...c.trace }))),
  });
  assert.equal(result.status, 0, result.stderr || String(result.error));
  payloads = JSON.parse(result.stdout);
  server = await createServer({
    root, server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ describeTrust, checkEvidence } = await server.ssrLoadModule('/src/utils/trust.ts'));
  ({ computeRunVerdict } = await server.ssrLoadModule('/src/utils/tooling.ts'));
  ({ TrustCard } = await server.ssrLoadModule('/src/components/TrustCard.tsx'));
  ({ ProgressPane } = await server.ssrLoadModule('/src/components/ProgressPane.tsx'));
  ({ WorkbenchPanel } = await server.ssrLoadModule('/src/components/WorkbenchPanel.tsx'));
  ({ abortedMessage, upsertRunMessage } = await server.ssrLoadModule('/src/utils/runEvents.ts'));
});
after(async () => {
  await server?.close();
  rmSync(home, { recursive: true, force: true });
});

test('past executions expose saved evidence without live workspace controls', () => {
  const props = {
    toolCalls: [{ id: 'edit-1', toolName: 'file_edit', args: { path: 'report.py' }, timestamp: 0, result: 'edited', success: true }],
    isRunning: false, activitySummary: null, onClose() {},
  };
  const past = renderToStaticMarkup(createElement(WorkbenchPanel, { ...props, historical: true }));
  assert.match(past, />Progress</);
  assert.match(past, />Activity</);
  assert.match(past, /report\.py/);
  assert.match(past, />Diff</);
  assert.doesNotMatch(past, />(File|Terminal|Follow)</);
  assert.doesNotMatch(past, /<button[^>]*title="report\.py"/);
  const live = renderToStaticMarkup(createElement(WorkbenchPanel, props));
  assert.match(live, />Terminal</);
  assert.match(live, /Requested changes/);
});

test('table evidence shows fixed requirements, data differences and unverified scope', () => {
  const html = renderToStaticMarkup(createElement(TrustCard, { trust: {
    reason: 'completed', verified: false, completionStatus: 'completed',
    verificationStatus: 'failed', verificationRequired: true,
    tableAcceptance: {
      required: true, status: 'fail', scope: 'tabular_data',
      contracts: [{ id: 'fixed', source_path: '/work/orders.csv', source_sha256: 'a',
        plan: { requirements: ['취소 주문 제외'], unverified: ['Report layout'] } }],
      results: [{ contract_id: 'fixed', output_path: '/work/summary.xlsx', status: 'fail',
        stats: { source_rows: 5, filtered_rows: 1, duplicates_removed: 1, output_rows: 2 },
        issues: [{ check: 'missing_or_incorrect_rows', count: 1, examples: [['B', '5.01']] }] }],
      unverified: ['Report layout'],
    },
  } }));
  for (const text of ['Table data checks', '취소 주문 제외', 'Filtered out 1', 'Duplicates removed 1',
    'Missing or incorrect rows: 1', 'B · 5.01', 'Not verified', 'Report layout']) {
    assert.ok(html.includes(text), text);
  }
});

test('code diffs stay available after research and in resumed history', async () => {
  const { shouldOpenWorkbench, preferredWorkbenchTab } = await server.ssrLoadModule('/src/utils/workbench.ts');
  const calls = ['web_search', 'web_fetch', 'web_search', 'file_edit'].map((toolName, index) => ({
    id: String(index), toolName, args: { path: 'app.py' }, timestamp: index,
  }));
  assert.equal(shouldOpenWorkbench(calls, []), true);
  assert.equal(preferredWorkbenchTab(calls, []), 'diff');
  const changes = [{ id: 'change-1', path: '/workspace/app.py', kind: 'modified', patch: '-return 1\n+return 2\n' }];
  assert.equal(shouldOpenWorkbench([], changes), true);
  const html = renderToStaticMarkup(createElement(WorkbenchPanel, {
    toolCalls: [], fileChanges: changes, historical: true, isRunning: false, activitySummary: null, onClose() {},
  }));
  assert.match(html, /Saved changes from this task/);
  assert.match(html, /-return 1/);
  assert.match(html, /\+return 2/);
  assert.doesNotMatch(html, /Workspace diff|>Terminal<|>File</);
});

cases.forEach((c, i) => test(c.name, () => {
  const trust = payloads[i];
  const view = describeTrust(trust);
  assert.equal(view.title, c.title);
  assert.equal(view.tone, c.tone);
  assert.equal(view.ok, c.ok);
  assert.equal(view.showCard, c.card);
  assert.equal(view.canEscalate, c.escalate ?? false);
  // Contradictory activity heuristics must not replace the server's result.
  assert.equal(computeRunVerdict(trust, { success: !c.ok }), c.ok);
  const card = renderToStaticMarkup(createElement(TrustCard, { trust }));
  if (c.card) assert.ok(card.includes(c.title), card);
  else assert.equal(card, '');
  if (trust.evidenceGate?.lastVerdict === 'fail') assert.ok(!card.includes('1 check passed'));
  if (trust.evidenceGate?.lastVerdict === 'pass' && c.card) assert.ok(!card.includes('no verdict'));
  assert.ok(!card.includes("I couldn&#x27;t confirm this result"));
  const progress = renderToStaticMarkup(createElement(ProgressPane, {
    trust, mode: 'research', toolCalls: [], isRunning: false,
    currentStep: null, activitySummary: null, orchestration: null,
  }));
  if (c.ok !== null) assert.ok(progress.includes(c.title), progress);
  assert.ok(!progress.includes('Not verified'));
  if (trust.completionCheck?.detail) {
    assert.ok(card.includes('show evidence'), card);
    assert.ok(progress.includes('show evidence'), progress);
    assert.ok(checkEvidence(trust).includes(trust.completionCheck.detail));
  }
}));

test('legacy output warning replaces the unsupported artifact claim', () => {
  const view = describeTrust({
    reason: 'completed_gate_warnings', verified: false, completionStatus: 'incomplete',
    honestNote: 'Delivered with caveats: the artifact exists but some quality checks did not fully pass — treat it as unverified.',
  });
  assert.equal(view.title, 'Review needed');
  assert.equal(view.ok, false);
  assert.ok(view.note.includes('specific check was not recorded'));
  assert.ok(!view.note.includes('artifact exists'));
});

test('legacy completed payload is not promoted to verified or treated as failure', () => {
  for (const verified of [true, false]) {
    const view = describeTrust({ reason: 'completed', verified });
    assert.equal(view.title, 'Completed');
    assert.equal(view.tone, 'neutral');
    assert.equal(view.showCard, false);
    assert.equal(view.canEscalate, false);
  }
});

test('legacy required checks still warn even if the old boolean claims success', () => {
  const view = describeTrust({ reason: 'completed', verified: true, testsPassedAfterEdit: false });
  assert.equal(view.title, 'Verification required');
  assert.equal(view.ok, false);
});

test('stop snapshots update one card with the final receipts', () => {
  const trust = payloads[cases.findIndex(c => c.name === 'cancelled run retains published document checks')];
  let messages = [{ id: 'answer', role: 'assistant', content: 'A file was saved.', timestamp: 0 }];
  messages = upsertRunMessage(messages, abortedMessage({ runId: 'run-1', trust: { ...trust, artifactReceipts: [] } }, 'unused', 1));
  const final = abortedMessage({ runId: 'run-1', trust }, 'unused', 2);
  messages = upsertRunMessage(upsertRunMessage(messages, final), final);
  assert.equal(messages.length, 2);
  assert.deepEqual(messages[1].trust.artifactReceipts, trust.artifactReceipts);
  const card = renderToStaticMarkup(createElement(TrustCard, { trust: messages[1].trust }));
  assert.ok(card.includes('published.xlsx') && card.includes('Stopped') && card.includes('Layout: not checked'), card);
});

test('legacy stop events retain the plain stop message', () => {
  const message = abortedMessage({}, 'legacy-stop', 1);
  assert.equal(message.content, 'Execution aborted.');
  assert.equal(message.id, 'legacy-stop');
  assert.equal(message.trust, undefined);
});

test('late snapshots only update an already owned stop notification', () => {
  const message = abortedMessage({ runId: 'previous-run' }, 'unused', 1);
  const otherChat = [{ id: 'other', role: 'user', content: 'new run', timestamp: 0 }];
  assert.equal(upsertRunMessage(otherChat, message, true), otherChat);
  const owned = [{ ...message, timestamp: 0 }, ...otherChat];
  const updated = upsertRunMessage(owned, message, true);
  assert.equal(updated.length, 2);
  assert.equal(updated[0].timestamp, 1);
  assert.equal(updated[1], otherChat[0]);
});
