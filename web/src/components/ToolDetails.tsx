import { useEffect, useState } from 'react';
import type { ToolCall } from '../types';
import './ToolActivity.css';

export function ToolOutput({ text, label = 'Output' }: { text: string; label?: string }) {
  const [full, setFull] = useState(false);
  const [copied, setCopied] = useState('Copy');
  useEffect(() => { setCopied('Copy'); }, [text]);
  const clipped = !full && text.length > 8000;
  return <div className="tool-output">
    <div className="tool-output-heading">
      <span>{label}</span>
      <button type="button" aria-label={`Copy ${label.toLowerCase()}`} onClick={async () => {
        try { await navigator.clipboard.writeText(text); setCopied('Copied'); }
        catch { setCopied('Could not copy'); }
      }}>{copied}</button>
    </div>
    <pre tabIndex={0} aria-label={label}>{text ? (clipped ? text.slice(0, 8000) : text) : 'No output returned.'}</pre>
    {clipped && <button className="tool-show-output" type="button" onClick={() => setFull(true)}>Show full output</button>}
  </div>;
}

export function ToolDetails({ tool, includeOutput = false }: { tool: ToolCall; includeOutput?: boolean }) {
  const [open, setOpen] = useState(false);
  return <details className="tool-technical" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Technical details{tool.durationMs != null ? ` · ${(tool.durationMs / 1000).toFixed(1)}s` : ''}</summary>
    {open && <div className="tool-technical-body">
      <span className="tool-identifier">{tool.toolName}</span>
      <ToolOutput text={JSON.stringify(tool.args, null, 2)} label="Input" />
      {includeOutput && tool.result !== undefined && <ToolOutput text={tool.result} label="Original output" />}
    </div>}
  </details>;
}
