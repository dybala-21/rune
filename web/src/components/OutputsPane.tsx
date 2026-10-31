import type { TrustInfo } from '../types';
import { artifactHref } from '../utils/artifactLinks';
import { basename } from '../utils/tooling';
import type { WorkbenchOutput } from '../utils/workbench';
import { RunEvidence } from './RunEvidence';

export function OutputsPane({ outputs, trust, isRunning }: {
  outputs: WorkbenchOutput[]; trust?: TrustInfo | null; isRunning: boolean;
}) {
  return <div className="workbench-outputs">
    <h3>Files from this task</h3>
    {outputs.length === 0 && <p>No files returned yet.</p>}
    {outputs.map(output => {
      const href = output.state === 'created' ? artifactHref(output.path, output.sessionId) : null;
      const downloadable = href?.startsWith('/api/v1/files/download?');
      const status = output.state === 'created' ? 'Created' : output.state === 'failed' ? 'Creation failed'
        : output.state === 'pending' && isRunning ? 'Creating…' : 'Result unconfirmed';
      return <div className="workbench-output" key={output.path}>
        <div title={output.path}>
          {downloadable ? <a href={href!} download>{basename(output.path)}</a> : <strong>{basename(output.path)}</strong>}
          <small>{output.path}</small>
        </div>
        <span>{status}</span>
      </div>;
    })}
    {trust ? <RunEvidence trust={trust} /> : <p>Creation status comes from the tool. Content checks appear when available.</p>}
  </div>;
}
