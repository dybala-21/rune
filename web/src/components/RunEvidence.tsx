import type { ChangeReceipt, TrustInfo } from '../types';
import { checkEvidence } from '../utils/trust';
import { TableChecks } from './TableChecks';

const HTTP_TOOLS = new Set(['browser_act', 'web_fetch', 'connector_request']);

function receiptLabel(receipt: ChangeReceipt): string {
  switch (receipt.tier) {
    case 'verified': return HTTP_TOOLS.has(receipt.tool) ? 'Result read back' : 'Change checked';
    case 'acknowledged': return HTTP_TOOLS.has(receipt.tool) ? 'Response received' : 'Reported by tool';
    case 'observed': return 'File change observed';
    case 'rejected': return 'Request rejected';
    default: return 'Result unconfirmed';
  }
}

/** Keep unresolved actions visible when the details are collapsed. */
export function RunEvidence({ trust }: { trust: TrustInfo }) {
  const receipts = trust.changeReceipts ?? [];
  const artifacts = trust.artifactReceipts ?? [];
  const evidence = checkEvidence(trust);
  const unchecked = [...new Set([
    ...(trust.requirementAcceptance?.uncheckedCriteria ?? []),
    ...(trust.tableAcceptance?.unverified ?? []),
  ])];
  const excerptNumbers = trust.sourceReview?.excerptOnlyNumbers ?? [];
  const unknown = receipts.filter(r => r.tier === 'unknown' && r.tool !== 'ledger').length;
  const rejected = receipts.filter(r => r.tier === 'rejected').length;
  const reported = receipts.filter(r => r.tier === 'acknowledged' && !HTTP_TOOLS.has(r.tool)).length;
  const notices = [
    ...(unknown ? [`${unknown === 1 ? 'One action has' : `${unknown} actions have`} an unconfirmed result. Check the result before trying again.`] : []),
    ...(rejected ? [`${rejected === 1 ? 'One request was' : `${rejected} requests were`} rejected. The task history shows any later attempts.`] : []),
    ...(reported ? [`${reported === 1 ? 'One action was' : `${reported} actions were`} reported as successful, but the result was not read back.`] : []),
    ...artifacts.flatMap(receipt => receipt.scope !== 'download' && Object.values(receipt.checks).includes('fail')
      ? [`A document check failed for ${receipt.artifacts.map(a => a.path.split(/[\\/]/).pop()).join(', ')}. See the details before using it.`] : []),
    ...(trust.workspaceWarning?.trim() ? [trust.workspaceWarning.trim()] : []),
    ...(trust.unsourcedNumbers?.length
      ? [`Figures not found in retrieved sources: ${trust.unsourcedNumbers.slice(0, 5).join(', ')}. These may be calculated or rounded.`] : []),
  ];
  const hasDetails = Boolean(receipts.length || artifacts.length || evidence || excerptNumbers.length || trust.tableAcceptance?.required);
  if (!hasDetails && !notices.length && !unchecked.length) return null;

  return <div style={{ fontSize: 12, lineHeight: 1.6, minWidth: 0, overflowWrap: 'anywhere' }}>
    {notices.map(note => <div key={note} role="note" style={{ marginTop: 8, color: 'var(--text-primary)' }}>{note}</div>)}
    {unchecked.length > 0 && <div style={{ marginTop: 8 }}>
      <div style={{ fontWeight: 600 }}>Still to check</div>
      <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
        {unchecked.map((criterion, index) => <li key={index}>{criterion}</li>)}
      </ul>
    </div>}
    {artifacts.map(receipt => receipt.scope === 'download' ? <div key={receipt.id} style={{ marginTop: 8 }}>
      <a href={`/api/v1/files/download?sessionId=${encodeURIComponent(receipt.sessionId)}&path=${encodeURIComponent(receipt.path)}`} download>{receipt.name}</a>
      {!receipt.verified && <div>The download copy could not be confirmed.</div>}
    </div> : null)}
    {hasDetails && <details style={{ marginTop: 8 }}>
      <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>Task details</summary>
      {excerptNumbers.length > 0 && <div style={{ marginTop: 8 }}>Numbers found only in search excerpts: {excerptNumbers.join(', ')}.</div>}
      <TableChecks data={trust.tableAcceptance} />
      {artifacts.map(receipt => receipt.scope === 'download' ? <div key={receipt.id} style={{ marginTop: 8 }}>
        <div style={{ fontWeight: 600 }}>{receipt.name}</div>
        <div>Download copy: {receipt.verified ? 'content hash checked' : 'not confirmed'}</div>
        <div>Document quality and layout require separate checks.</div>
      </div> : <div key={receipt.revision} style={{ marginTop: 8 }}>
        <div style={{ fontWeight: 600 }}>Document checks</div>
        <div>{receipt.artifacts.map(a => a.path.split(/[\\/]/).pop()).join(', ')}</div>
        <div>Saved content: {receipt.checks.native_content === 'pass' ? 'passed' : 'not confirmed'}</div>
        <div>Source totals: {receipt.checks.source_metrics === 'pass' ? 'passed' : 'not confirmed'}</div>
        <div>Layout: {receipt.checks.visual_layout === 'pass' ? 'checked' : 'not checked'}</div>
        <div>Full task review: {receipt.checks.task_acceptance === 'pass' ? 'passed' : 'not performed'}</div>
      </div>)}
      {receipts.length > 0 && <div style={{ marginTop: 8 }}>
        <div style={{ fontWeight: 600 }}>Actions</div>
        {receipts.map((receipt, index) => <div key={index} style={{ marginTop: 6 }}>
          <div>{receipt.target}{' · '}<span style={{ color: 'var(--text-secondary)' }}>{receiptLabel(receipt)}</span></div>
          <div style={{ color: 'var(--text-muted)' }}>{receipt.evidence}</div>
        </div>)}
        {receipts.some(r => HTTP_TOOLS.has(r.tool) && r.tier === 'acknowledged') &&
          <div style={{ marginTop: 6, color: 'var(--text-muted)' }}>An HTTP response alone does not confirm a change in the app.</div>}
      </div>}
      {evidence && <pre style={{
        marginTop: 8, padding: '8px 10px', background: 'var(--bg-primary)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
        fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.5,
        whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', color: 'var(--text-secondary)',
        maxHeight: 160, overflow: 'auto',
      }}>{evidence}</pre>}
    </details>}
  </div>;
}
