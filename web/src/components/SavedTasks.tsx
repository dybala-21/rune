import { useEffect, useRef, useState } from 'react';
import { fetchSavedTasks, type SavedTask } from '../api';
import './SavedTasks.css';

const STATUS: Record<string, string> = {
  pending: 'To do', running: 'In progress', interrupted: 'Interrupted', cancelled: 'Stopped',
  failed: 'Needs attention', completed: 'Saved result', verified: 'Saved result',
};

export function SavedTasks({ sessionId, onSelect, disabled }: {
  sessionId: string; onSelect: (id: string) => void; disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [revision, setRevision] = useState(0);
  const [result, setResult] = useState<{ tasks: SavedTask[]; workspace: string; hasMore: boolean } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const generation = useRef(0);

  useEffect(() => {
    const reset = () => { generation.current++; setResult(null); setOffset(0); setRevision(value => value + 1); };
    reset();
    window.addEventListener('rune:workspace-changed', reset);
    return () => window.removeEventListener('rune:workspace-changed', reset);
  }, [sessionId]);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const version = ++generation.current;
    setLoading(true); setError(''); setResult(null);
    fetchSavedTasks(sessionId, search, offset, controller.signal)
      .then(value => { if (version === generation.current) setResult(value); })
      .catch(error => { if (!controller.signal.aborted && version === generation.current) setError(String(error.message)); })
      .finally(() => { if (version === generation.current) setLoading(false); });
    return () => { controller.abort(); generation.current++; };
  }, [open, sessionId, search, offset, revision]);

  return <section className="saved-tasks">
    <button type="button" className="saved-tasks-toggle" aria-expanded={open}
      onClick={() => setOpen(value => !value)}>Saved work <span aria-hidden="true">{open ? '−' : '+'}</span></button>
    {open && <div className="saved-tasks-content">
      <form onSubmit={event => { event.preventDefault(); setOffset(0); setSearch(query); setRevision(value => value + 1); }}>
        <input aria-label="Find saved work" placeholder="Find a task or file" value={query} maxLength={2000}
          onChange={event => setQuery(event.target.value)} />
        <button type="submit" aria-label="Search saved work">Find</button>
      </form>
      <p className="saved-tasks-hint">Choose a task to add it to your message.</p>
      {loading && <p role="status">Loading…</p>}
      {error && <p role="alert">{error} <button type="button" onClick={() => setRevision(value => value + 1)}>Retry</button></p>}
      {result && <>
        <p className="saved-tasks-workspace" title={result.workspace}>{result.workspace.split('/').filter(Boolean).slice(-1)[0]}</p>
        {!result.tasks.length && <p>No saved work found in this workspace.</p>}
        <div className="saved-tasks-list">
          {result.tasks.map(task => <article key={task.id}>
            <p className="saved-tasks-title">{task.goal}</p>
            <p className="saved-tasks-hint">{STATUS[task.status] || 'Review needed'} · {new Date(task.updatedAt * 1000).toLocaleDateString()}</p>
            {task.blocker && <p className="saved-tasks-blocker">{task.blocker}</p>}
            {task.files.length > 0 && <p className="saved-tasks-files">{task.files.map(file => file.split(/[\\/]/).slice(-1)[0]).join(', ')}</p>}
            <button type="button" disabled={disabled || loading} onClick={() => onSelect(task.id)}>Use in message</button>
          </article>)}
        </div>
        {(offset > 0 || result.hasMore) && <div className="saved-tasks-pages">
          <button type="button" disabled={offset === 0 || loading} onClick={() => setOffset(value => Math.max(0, value - 10))}>Previous</button>
          <button type="button" disabled={!result.hasMore || loading} onClick={() => setOffset(value => value + 10)}>Next</button>
        </div>}
      </>}
    </div>}
  </section>;
}
