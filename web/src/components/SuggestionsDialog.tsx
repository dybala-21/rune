import { useRef } from 'react';
import { useFocusTrap } from '../hooks/useFocusTrap';
import type { ProactiveSuggestion } from '../types';
import { ProactiveCard } from './ProactiveCard';

export function SuggestionsDialog({ suggestions, onClose, returnFocus }: {
  suggestions: ProactiveSuggestion[];
  returnFocus?: HTMLElement | null;
  onClose: () => void;
}) {
  const trigger = useRef(returnFocus ?? (typeof document === 'undefined' ? null : document.activeElement as HTMLElement | null));
  const ref = useFocusTrap<HTMLDivElement>(true, trigger.current);
  return <div className="suggestions-overlay" onClick={event => {
    if (event.target === event.currentTarget) onClose();
  }}>
    <section ref={ref} className="suggestions-dialog" role="dialog" aria-modal="true" aria-labelledby="suggestions-title"
      onKeyDown={event => {
        if (event.key === 'Escape' && !event.defaultPrevented) {
          event.preventDefault();
          event.stopPropagation();
          onClose();
        }
      }}>
      <header>
        <h1 id="suggestions-title">Suggestions</h1>
        <button type="button" className="toolbar-icon" aria-label="Close suggestions" onClick={onClose}>×</button>
      </header>
      <div className="suggestions-content">
        {suggestions.length ? suggestions.map(suggestion => <ProactiveCard key={suggestion.id} suggestion={suggestion} />)
          : <p className="suggestions-empty">New suggestions will appear here. You decide which ones to run.</p>}
      </div>
    </section>
  </div>;
}
