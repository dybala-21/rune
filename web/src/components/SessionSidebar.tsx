import { useEffect, useState } from 'react';
import { fetchSessions, getLiveSessionId, type SessionInfo } from '../api';

interface SessionSidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onNewChat?: () => void;
  liveTitle?: string;
  onOpenSchedule?: () => void;
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

export function SessionSidebar({ currentSessionId, onSelectSession, onNewChat, liveTitle, onOpenSchedule }: SessionSidebarProps) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const result = await fetchSessions({ limit: 50 });
        if (!cancelled) { setSessions(result.sessions); setError(false); }
      } catch {
        if (!cancelled) setError(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    const timer = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [retry]);

  const liveSessionId = getLiveSessionId();
  const saved = sessions.filter(session => session.id !== liveSessionId);
  return <nav className="session-sidebar" aria-label="Task list">
    <div className="session-create">
      <button type="button" className="new-task-button" onClick={() => onNewChat ? onNewChat() : onSelectSession(null)}>
        <span aria-hidden="true">＋</span> New task
      </button>
    </div>
    <div className="session-current">
      <p className="session-group-label">Current</p>
      <button type="button" className="session-item" aria-current={currentSessionId === null ? 'page' : undefined}
        onClick={() => onSelectSession(null)} title={liveTitle || 'New task'}>
        <span className="session-title">{liveTitle || 'New task'}</span>
        <span className="session-meta"><span className="session-current-dot" /> Current conversation</span>
      </button>
    </div>
    <div className="session-list">
      {error && <div className="session-error" role="status">Couldn’t load tasks.
        <button type="button" onClick={() => setRetry(value => value + 1)}>Try again</button>
      </div>}
      {loading ? <div className="session-loading" aria-label="Loading tasks" aria-busy="true">
        {[0, 1, 2].map(index => <div key={index} className="skeleton" />)}
      </div> : !saved.length && !error ? <p className="session-empty">Your past tasks will appear here.</p> : GROUPS.map(group => {
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
    </div>
    {onOpenSchedule && <div className="session-footer"><button type="button" onClick={onOpenSchedule}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4m10-4v4M3 11h18"/></svg>
      Scheduled tasks
    </button></div>}
  </nav>;
}
