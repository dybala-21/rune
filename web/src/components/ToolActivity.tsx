import { memo, useState } from 'react';
import type { ToolCall } from '../types';
import { activityStatus, presentTool, STATE_LABELS, toolLabel, toolState } from '../utils/toolActivity';
import { truncate } from '../utils/tooling';
import { ToolActivityIcon } from './ToolActivityIcon';
import { ToolDetails, ToolOutput } from './ToolDetails';
import './ToolActivity.css';

function Chevron() {
  return <svg className="tool-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.6" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>;
}

function ActivityStatus({ state, text, compact }: { state: string; text: string; compact: boolean }) {
  return <span className={`tool-status tool-state-${state}`} role="status">
    {compact ? <><span aria-hidden="true">✓</span><span className="activity-sr-only">{text}</span></> : text}
  </span>;
}

export function ToolBody({ tool, active = false }: { tool: ToolCall; active?: boolean }) {
  const view = presentTool(tool);
  const state = toolState(tool, active);
  return <div className="tool-body">
    {view.href && <a className="tool-page-link" href={view.href} target="_blank" rel="noopener noreferrer">Open page ↗</a>}
    {view.context && <div className="tool-context-full">{view.context}</div>}
    {view.kind === 'terminal' && view.target && <ToolOutput text={view.target} label="Command" />}
    {tool.result !== undefined ? <ToolOutput text={tool.result} label={tool.success === false ? 'Error' : 'Output'} />
      : <p className={`tool-note tool-state-${state}`}>{state === 'interrupted' ? 'No result was received before this run ended.'
        : state === 'waiting' ? 'Review the app connection request to continue.'
        : state === 'failed' ? 'The tool failed without returning output.'
        : state === 'unknown' || state === 'check-failed' || state === 'unconfirmed' ? 'This activity has no saved result.'
        : 'Waiting for the tool to return a result.'}</p>}
    {tool.checkStatus === 'pass' && <p className="tool-note">Requested checks passed.</p>}
    {tool.checkStatus === 'fail' && <p className="tool-note tool-state-check-failed">The returned result did not pass the requested checks.</p>}
    {tool.checkStatus === 'inconclusive' && <p className="tool-note">Requested checks could not be confirmed.</p>}
    {tool.outputTruncated && <p className="tool-note">Only part of the output was saved in this activity.</p>}
    <ToolDetails tool={tool} />
  </div>;
}

function ToolRow({ tool, active }: { tool: ToolCall; active: boolean }) {
  const [open, setOpen] = useState(false);
  const view = presentTool(tool);
  const state = toolState(tool, active);
  return <details className="tool-activity-row" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="tool-row-summary">
      <span className="tool-row-description"><span className="tool-row-label">{view.label}</span>
        {view.target && <span className={view.monospace ? 'tool-target tool-command' : 'tool-target'}>{truncate(view.target, 240)}</span>}</span>
      <ActivityStatus state={state} text={STATE_LABELS[state]} compact={state === 'succeeded'} /><Chevron />
    </summary>
    {open && <ToolBody tool={tool} active={active} />}
  </details>;
}

export const ToolActivity = memo(function ToolActivity({ tools, active = false }: { tools: ToolCall[]; active?: boolean }) {
  const [open, setOpen] = useState(false);
  if (!tools.length) return null;
  const status = activityStatus(tools, active);
  const current = tools.find(tool => ['pending', 'waiting'].includes(toolState(tool, active))) ?? tools[0];
  const view = presentTool(current);
  const grouped = tools.length > 1;
  const targets = grouped && !status.running ? [...new Set(tools.map(tool => presentTool(tool).target).filter(Boolean))] : [view.target];
  const target = truncate(targets.slice(0, 2).join(' · '), 240) + (targets.length > 2 ? ` +${targets.length - 2}` : '');
  const label = grouped ? toolLabel(current.toolName) : view.label;
  return <details className={`tool-activity tool-state-${status.state}`} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="tool-activity-summary" aria-label={`${label} · ${status.text}${target ? `: ${target}` : ''}`}>
      <span className="tool-activity-symbol"><ToolActivityIcon kind={view.kind} /></span>
      <span className="tool-summary-body">
        <span className="tool-heading">{label}</span>
        {target && <span className={view.monospace ? 'tool-target tool-command' : 'tool-target'} title={view.context || target}>{target}</span>}
      </span>
      <ActivityStatus state={status.state} text={status.text} compact={!grouped && status.state === 'succeeded'} /><Chevron />
    </summary>
    {open && <div className="tool-activity-content">
      {grouped ? tools.map(tool => <ToolRow key={tool.id} tool={tool} active={active} />) : <ToolBody tool={tools[0]} active={active} />}
    </div>}
  </details>;
});
