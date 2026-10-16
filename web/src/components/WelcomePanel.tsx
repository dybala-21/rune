import { WorkspaceSetupRow } from './WorkspaceSetupRow';

const SUGGESTIONS = [
  { label: 'Review a document', description: 'Summarize the decisions and next steps in these meeting notes.', path: 'M7 3h8l4 4v14H7z M14 3v5h5 M10 12h6 M10 16h6' },
  { label: 'Explore a codebase', description: 'Review this project’s structure and suggest improvements.', path: 'M9 8l-5 4 5 4M15 8l5 4-5 4' },
  { label: 'Compare files', description: 'Compare the attached files and flag differences worth checking.', path: 'M4 5h6v14H4z M14 5h6v14h-6z M7 9h0 M17 9h0' },
];

export function WelcomePanel({ onSuggest }: { onSuggest?: (text: string) => void }) {
  return <section className="welcome-panel" aria-labelledby="welcome-title">
    <p className="welcome-eyebrow">New task</p>
    <h1 id="welcome-title">What’s next?</h1>
    <p className="welcome-description">Pick up where you left off, or bring something new.</p>
    <div className="welcome-suggestions">
      {SUGGESTIONS.map(item => <button key={item.label} type="button" className="welcome-suggestion"
        onClick={onSuggest ? () => onSuggest(item.description) : undefined} disabled={!onSuggest}>
        <span className="welcome-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d={item.path} />
        </svg></span>
        <span className="welcome-suggestion-copy"><strong>{item.label}</strong><small>{item.description}</small></span>
        <span className="welcome-arrow" aria-hidden="true">↗</span>
      </button>)}
    </div>
    <div className="welcome-workspace"><WorkspaceSetupRow /></div>
  </section>;
}
