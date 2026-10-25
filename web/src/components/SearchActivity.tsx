import { memo, useMemo, useState } from 'react';
import type { ToolCall } from '../types';
import { searchPreview, searchStatus, sourceHost } from '../utils/searchActivity';
import { ToolDetails } from './ToolDetails';
import './SearchActivity.css';

function SearchIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
    <circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4.5 4.5" />
  </svg>;
}

export function SearchResults({ tool, active = false, showQuery = true }: { tool: ToolCall; active?: boolean; showQuery?: boolean }) {
  const preview = searchPreview(tool);
  const status = searchStatus(tool, active);
  return <section className="search-query-results">
    {showQuery && <div className="search-query-heading">
      <span>{String(tool.args.query ?? 'Web search')}</span>
      {status === 'pending' && <span className="search-meta">Searching…</span>}
      {status === 'failed' && <span className="search-problem">Search failed</span>}
      {status === 'interrupted' && <span className="search-problem">Search interrupted</span>}
      {status === 'unknown' && <span className="search-meta">Status unavailable</span>}
    </div>}
    {preview.items.length > 0 && <ul className="search-result-list">
      {preview.items.map((item, index) => <li key={`${item.url}:${index}`}>
        <a className="search-result-link" href={item.url} target="_blank" rel="noopener noreferrer">
          <span className="search-source-mark" aria-hidden="true">{sourceHost(item.url).charAt(0).toUpperCase()}</span>
          <span className="search-result-label">
            <span className="search-domain">{sourceHost(item.url)}</span>
            <span className="search-result-title">{item.title}</span>
          </span>
          <svg className="search-external" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
            <path d="M8 5h11v11M19 5 5 19" />
          </svg>
        </a>
        {item.description && <p className="search-excerpt">{item.description}</p>}
        {item.date && <div className="search-date">Date reported by search: {item.date}</div>}
      </li>)}
    </ul>}
    {status === 'complete' && preview.count === 0 && <p className="search-empty">No results for this search.</p>}
    {preview.count !== undefined && preview.count > preview.items.length &&
      <p className="search-meta">Showing {preview.items.length} of {preview.count} results.</p>}
    {tool.outputTruncated && !tool.searchResults && <p className="search-meta">Only part of this search was saved in the activity log.</p>}
    <ToolDetails tool={tool} includeOutput />
  </section>;
}

export const SearchActivity = memo(function SearchActivity({ tools, active = false }: { tools: ToolCall[]; active?: boolean }) {
  const [open, setOpen] = useState(false);
  const previews = useMemo(() => tools.map(searchPreview), [tools]);
  const states = tools.map(tool => searchStatus(tool, active));
  const pending = states.filter(state => state === 'pending').length;
  const failed = states.filter(state => state === 'failed').length;
  const interrupted = states.filter(state => state === 'interrupted').length;
  const unknown = states.filter(state => state === 'unknown').length;
  const countKnown = previews.every((preview, i) => preview.count !== undefined && states[i] === 'complete');
  const count = countKnown ? previews.reduce((sum, preview) => sum + preview.count!, 0) : undefined;
  const query = tools.find((_, index) => states[index] === 'pending') ?? tools[0];
  const title = pending ? 'Searching the web' : failed === tools.length ? 'Search failed'
    : interrupted === tools.length ? 'Search interrupted' : count === 0 ? 'No search results' : 'Web search';
  const status = [
    tools.length > 1 ? `${tools.length} searches` : '',
    count ? `${count} result${count === 1 ? '' : 's'}` : '',
    failed && tools.length > 1 ? `${failed} failed` : '', interrupted && tools.length > 1 ? `${interrupted} interrupted` : '',
    unknown ? 'Status unavailable' : '',
  ].filter(Boolean).join(' · ');

  return <details className="search-activity" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="search-summary" aria-label={`${title}${status ? ` · ${status}` : ''}: ${String(query?.args.query ?? '')}`}>
      <span className={`search-symbol${pending ? ' search-symbol-active' : ''}`}><SearchIcon /></span>
      <span className="search-summary-body">
        <span className="search-heading">{title}</span>
        <span className="search-query" title={String(query?.args.query ?? '')}>{String(query?.args.query ?? '')}</span>
      </span>
      <span className={`search-status ${failed || interrupted ? 'search-problem' : 'search-meta'}`} role="status"><span className="activity-sr-only">{title} </span>{status}</span>
      <svg className="search-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>
    </summary>
    {open && <div className="search-content">
      {previews.some(preview => preview.items.length > 0) && <p className="search-context">Search excerpts · open a source for the full page.</p>}
      {tools.map(tool => <SearchResults key={tool.id} tool={tool} active={active} />)}
    </div>}
  </details>;
});
