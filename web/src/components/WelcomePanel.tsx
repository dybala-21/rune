import { useState } from 'react';
import type { ConfigInfo } from '../api';
import { WorkspaceSetupRow } from './WorkspaceSetupRow';
import './WelcomePanel.css';

const SUGGESTIONS = [
  { label: 'Work with a document', detail: 'Find the decisions and next steps', draft: 'Review the document I attach and summarize the decisions, open questions, and next steps.', path: 'M7 3h8l4 4v14H7z M14 3v5h5 M10 12h6 M10 16h6' },
  { label: 'Research a question', detail: 'Compare options and check sources', draft: 'Help me compare the options for ', path: 'M21 21l-5-5 M18 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0' },
  { label: 'Work on a project', detail: 'Explore code, make changes, and test', draft: 'Review the project in my working folder and help me with ', path: 'M9 8l-5 4 5 4M15 8l5 4-5 4' },
];

interface WelcomePanelProps {
  onSuggest?: (text: string) => void;
  config?: ConfigInfo | null;
  connected?: boolean;
  onOpenSettings?: () => void;
}

export function WelcomePanel({ onSuggest, config, connected, onOpenSettings }: WelcomePanelProps) {
  const [helpOpen, setHelpOpen] = useState(false);
  const managed = config?.executionEnvironment?.managed === true;
  const model = config?.activeModel;

  return <section className="welcome-panel welcome-start" aria-labelledby="welcome-title">
    <h1 id="welcome-title">What’s on your list?</h1>
    <p className="welcome-start-description">Bring a question, a file, or a project.</p>
    <div className="welcome-start-suggestions" aria-label="Task ideas">
      {SUGGESTIONS.map(item => <button key={item.label} type="button" className="welcome-suggestion"
        onClick={onSuggest ? () => onSuggest(item.draft) : undefined} disabled={!onSuggest}>
        <span className="welcome-start-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d={item.path} />
        </svg></span>
        <span className="welcome-start-copy"><strong>{item.label}</strong><small>{item.detail}</small></span>
        <span className="welcome-start-arrow" aria-hidden="true">↗</span>
      </button>)}
    </div>
    <details className="welcome-start-help" onToggle={event => setHelpOpen(event.currentTarget.open)}>
      <summary>Getting started <svg width="14" height="14" viewBox="0 0 16 16" fill="none"
        stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m5 6 3 3 3-3" /></svg></summary>
      <div className="welcome-start-guide">
        <p>Choose an idea to edit, or write your own request below. Add files with the attachment button.</p>
        <dl className="welcome-start-context">
          <div><dt>Selected model</dt><dd>{model ? model.model : 'Not loaded yet'}</dd></div>
          <div><dt>Workspace</dt><dd>{config ? managed ? 'Hosted workspace' : 'Connected engine' : 'Not loaded yet'}</dd></div>
        </dl>
        <p>{connected === false
          ? 'Rune is reconnecting. You can keep writing your request.'
          : !config
            ? 'Workspace details will appear when Rune connects.'
            : managed
              ? 'Files and commands use your hosted workspace. Attach files to include them in a task.'
              : 'Files and commands use the connected Rune engine. Choose a working folder for project tasks.'}</p>
        {helpOpen && config && connected !== false && !managed && <div className="welcome-start-workspace"><WorkspaceSetupRow /></div>}
        <div className="welcome-start-settings">
          <span>{model ? 'Switch models below. Manage connections in Settings.' : 'Manage models and connections in Settings.'}</span>
          {onOpenSettings && <button type="button" onClick={onOpenSettings}>Open settings <span aria-hidden="true">→</span></button>}
        </div>
      </div>
    </details>
  </section>;
}
