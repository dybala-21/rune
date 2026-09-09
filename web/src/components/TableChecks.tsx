import type { TableAcceptanceInfo } from '../types';

const statusText = {
  pass: 'Passed', fail: 'Differences found', inconclusive: 'Could not confirm',
  unverified: 'Not checked', stale: 'Changed since check',
};
const checkText: Record<string, string> = {
  columns: 'Column mismatch', missing_or_incorrect_rows: 'Missing or incorrect rows',
  unexpected_or_duplicate_rows: 'Unexpected or duplicate rows',
  read_or_compute: 'Check unavailable', freshness: 'File changed',
  numeric_cells: 'Numeric cell types',
};
const filename = (path: string) => path.split(/[\\/]/).pop() || path;

export function TableChecks({ data }: { data?: TableAcceptanceInfo | null }) {
  if (!data?.required) return null;
  return <div style={{ marginTop: 10, fontSize: 12, lineHeight: 1.6, overflowWrap: 'anywhere' }}>
    <div style={{ fontWeight: 600 }}>Table data checks · {statusText[data.status]}</div>
    {data.contracts.map(contract => <details key={contract.id} style={{ marginTop: 4 }}>
      <summary>Interpreted requirements · {filename(contract.source_path)}</summary>
      <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
        {contract.plan.requirements.map((text, i) => <li key={i}>{text}</li>)}
      </ul>
    </details>)}
    {data.results.map(result => <div key={result.output_path} style={{ marginTop: 6 }}>
      <div>{filename(result.output_path)} · {statusText[result.status]}</div>
      {result.stats && <div style={{ color: 'var(--text-secondary)' }}>
        Source {result.stats.source_rows} rows · Filtered out {result.stats.filtered_rows} ·
        {' '}Duplicates removed {result.stats.duplicates_removed} · Result {result.stats.output_rows} rows
      </div>}
      {result.issues?.map((issue, i) => <div key={i}>
        {checkText[issue.check] || issue.check}{issue.count != null ? `: ${issue.count}` : ''}
        {issue.detail && ` — ${issue.detail}`}
        {issue.expected && <div>Expected: {issue.expected.join(', ')}</div>}
        {issue.actual && <div>Found: {issue.actual.join(', ')}</div>}
        {issue.examples?.map((row, j) => <div key={j} style={{ fontFamily: 'var(--font-mono)' }}>{row.join(' · ')}</div>)}
      </div>)}
    </div>)}
    <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>Checks cover table data; prose and layout are not checked.</div>
    {data.unverified.length > 0 && <div>
      <div style={{ fontWeight: 600 }}>Not verified</div>
      <ul style={{ margin: '4px 0', paddingLeft: 18 }}>{data.unverified.map((item, i) => <li key={i}>{item}</li>)}</ul>
    </div>}
    {!!data.out_of_scope?.length && <div>
      <div>Outside these data checks</div>
      <ul style={{ margin: '4px 0', paddingLeft: 18 }}>{data.out_of_scope.map((item, i) => <li key={i}>{item}</li>)}</ul>
    </div>}
  </div>;
}
