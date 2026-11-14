import { SavedTasks } from './SavedTasks';
import { useEffect, useState } from 'react';
import { emptySessionList, SessionListLoader } from '../utils/sessionList';
import './SessionSidebar.css';
import { fetchSessions, getLiveSessionId } from '../api';

interface SessionSidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onNewChat?: () => void;
  liveTitle?: string;
  onOpenSchedule?: () => void;
  onOpenSettings?: () => void;
  onSelectTask?: (id: string) => void;
  taskSelectionDisabled?: boolean;
}

type DateGroup = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Earlier';
const GROUPS: DateGroup[] = ['Today', 'Yesterday', 'Previous 7 days', 'Earlier'];

function dateGroup(iso: string): DateGroup {
  const date = new Date(iso), now = new Date();
  if (date.toDateString() === now.toDateString()) return 'Today';
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return now.getTime() - date.getTime() < 7 * 86400000 ? 'Previous 7 days' : 'Earlier';
}

function dateLabel(iso: string): string {
  const date = new Date(iso);
  if (!Number.isFinite(date.getTime())) return '';
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric' }).format(date);
}

export function SessionSidebar({ currentSessionId, onSelectSession, onNewChat, liveTitle, onOpenSchedule, onOpenSettings, onSelectTask, taskSelectionDisabled }: SessionSidebarProps) {
  const [search, setSearch] = useState('');
  const [list, setList] = useState(emptySessionList);
  const [loader] = useState(() => new SessionListLoader(fetchSessions, setList));
  const { sessions, loading, loadingMore, hasMore, error } = list;

  useEffect(() => {
    loader.reset(search.trim());
    const timer = setTimeout(() => { void loader.load(); }, search.trim() ? 250 : 0);
    return () => { clearTimeout(timer); loader.cancel(); };
  }, [loader, search]);

  useEffect(() => {
    const timer = setInterval(() => { void loader.refresh(); }, 30000);
    return () => clearInterval(timer);
  }, [loader]);

  const liveSessionId = getLiveSessionId();
  const saved = sessions.filter(session => session.id !== liveSessionId);
  return <nav className="session-sidebar" aria-label="Chat list">
    <div className="session-create">
      <button type="button" className="new-task-button" onClick={() => onNewChat ? onNewChat() : onSelectSession(null)}>
        <span aria-hidden="true">＋</span> New chat
      </button>
    </div>
    <div className="session-current">
      <p className="session-group-label">Current</p>
      <button type="button" className="session-item" aria-current={currentSessionId === null ? 'page' : undefined}
        onClick={() => onSelectSession(null)} title={liveTitle || 'New chat'}>
        <span className="session-title">{liveTitle || 'New chat'}</span>
      </button>
    </div>
    {onSelectTask && <SavedTasks sessionId={currentSessionId || liveSessionId}
      onSelect={onSelectTask} disabled={!!taskSelectionDisabled} />}
    <div className="session-search">
      <input type="search" aria-label="Search chats by title" placeholder="Search chats"
        maxLength={200} value={search} onChange={event => setSearch(event.target.value)} />
      {search && <button type="button" aria-label="Clear chat search" onClick={event => { event.currentTarget.parentElement?.querySelector('input')?.focus(); setSearch(''); }}>×</button>}
    </div>
    <div className="session-list" aria-busy={loading || loadingMore}>
      {error && <div className="session-error" role="status">
        {error === 'more' ? 'Couldn’t load older chats.' : error === 'refresh' ? 'Couldn’t refresh chats.' : 'Couldn’t load chats.'}
        <button type="button" onClick={() => { void loader.retry(); }}>Try again</button>
      </div>}
      {loading ? <div className="session-loading" aria-label="Loading chats" aria-busy="true">
        {[0, 1, 2].map(index => <div key={index} className="skeleton" />)}
      </div> : !saved.length && !error ? <p className="session-empty" role="status">{search.trim() ? 'No chats match this title.' : 'Your conversations will appear here.'}</p> : GROUPS.map(group => {
        const items = saved.filter(session => dateGroup(session.updatedAt) === group);
        if (!items.length) return null;
        return <section key={group}>
          <h2 className="session-group-label">{group}</h2>
          {items.map(session => <button type="button" className="session-item" key={session.id}
            aria-current={currentSessionId === session.id ? 'page' : undefined}
            onClick={() => onSelectSession(session.id)} title={session.title || 'Untitled task'}>
            <span className="session-title">{session.title || 'Untitled task'}</span>
            <span className="session-meta"><span>{session.turnCount} {session.turnCount === 1 ? 'message' : 'messages'}</span><time dateTime={session.updatedAt}>{dateLabel(session.updatedAt)}</time></span>
          </button>)}
        </section>;
      })}
      {!loading && hasMore && <button type="button" className="session-load-more" disabled={loadingMore}
        onClick={() => { void loader.loadMore(); }}>{loadingMore ? 'Loading…' : 'Load more chats'}</button>}
    </div>
    <div className="session-footer">{onOpenSchedule && <button type="button" onClick={onOpenSchedule}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4m10-4v4M3 11h18"/></svg>
      Scheduled tasks
    </button>}
    {onOpenSettings && <button type="button" onClick={onOpenSettings}>
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden="true"><circle cx="8" cy="8" r="2.5"/><path d="M8 1v2m0 10v2M1 8h2m10 0h2M3 3l1.5 1.5m7 7L13 13M3 13l1.5-1.5m7-7L13 3"/></svg>
      Settings
    </button>}</div>
  </nav>;
}
