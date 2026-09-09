import { RuneMark } from './RuneMark';
import { WorkspaceSetupRow } from './WorkspaceSetupRow';

const SUGGESTIONS = [
  { label: 'Build & fix', description: 'Find and fix a failing test', path: 'M9 8l-5 4 5 4M15 8l5 4-5 4' },
  { label: 'Make sense of it', description: 'Summarize this project', path: 'M7 3h8l4 4v14H7z M14 3v5h5 M10 12h6 M10 16h6' },
  { label: 'Explore & research', description: "What's new with AI agents?", path: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M3 12h18 M12 3c3 3 3 15 0 18 M12 3c-3 3-3 15 0 18' },
];

export function WelcomePanel({ onSuggest }: { onSuggest?: (text: string) => void }) {
  return <section className="welcome-panel" aria-labelledby="welcome-title">
    <div className="welcome-brand"><RuneMark state="idle" size={38} title="RUNE" /><span>Your personal workspace</span></div>
    <h1 id="welcome-title">What’s on your mind?</h1>
    <p className="welcome-description">A question, a rough idea, a task to finish.<br />Let’s work through it.</p>
    <div className="welcome-suggestions">
      {SUGGESTIONS.map(item => <button key={item.label} type="button" className="welcome-suggestion"
        onClick={onSuggest ? () => onSuggest(item.description) : undefined} disabled={!onSuggest}>
        <span className="welcome-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none"
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
