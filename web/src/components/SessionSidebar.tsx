import { useEffect, useState } from 'react';
import { fetchSessions, type SessionInfo } from '../api';

interface SessionSidebarProps {
  currentSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onNewChat?: () => void;
}

type DateGroup = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Older';

function getDateGroup(iso: string): DateGroup {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = diff / 86400000;

  if (days < 1 && d.getDate() === now.getDate()) return 'Today';
  if (days < 2 && d.getDate() === new Date(now.getTime() - 86400000).getDate()) return 'Yesterday';
  if (days < 7) return 'Previous 7 days';
  return 'Older';
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return d.toLocaleDateString();
}

export function SessionSidebar({ currentSessionId, onSelectSession, onNewChat }: SessionSidebarProps) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const result = await fetchSessions({ limit: 50 });
        if (!cancelled) setSessions(result.sessions);
      } catch {
        // keep empty
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const timer = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  // Group sessions by date
  const groups = new Map<DateGroup, SessionInfo[]>();
  for (const s of sessions) {
    const group = getDateGroup(s.updatedAt);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)!.push(s);
  }

  const groupOrder: DateGroup[] = ['Today', 'Yesterday', 'Previous 7 days', 'Older'];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg-primary)',
    }}>
      {/* Header + New Chat */}
      <div style={{ padding: '14px 14px 10px' }}>
        <button
          onClick={() => {
            if (onNewChat) {
              onNewChat();
              return;
            }
            onSelectSession(null);
          }}
          style={{
            width: '100%',
            padding: '9px 14px',
            background: 'var(--accent-subtle)',
            color: 'var(--accent)',
            border: '1px solid transparent',
            borderRadius: 'var(--radius-md)',
            fontSize: 13,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            transition: 'all 0.15s',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <line x1="8" y1="3" x2="8" y2="13" />
            <line x1="3" y1="8" x2="13" y2="8" />
          </svg>
          New Chat
        </button>
      </div>

      {/* Live session */}
      <button
        onClick={() => onSelectSession(null)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          margin: '0 8px',
          padding: '9px 12px',
          background: currentSessionId === null ? 'var(--bg-tertiary)' : 'transparent',
          border: 'none',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-primary)',
          fontSize: 13,
          cursor: 'pointer',
          textAlign: 'left',
          width: 'calc(100% - 16px)',
          transition: 'background 0.15s',
          borderLeft: currentSessionId === null ? '2px solid var(--accent)' : '2px solid transparent',
        }}
      >
        <span className="status-dot status-dot--success" style={{ width: 6, height: 6 }} />
        <span style={{ fontWeight: 500 }}>Live</span>
      </button>

      {/* Session list */}
      <div style={{ flex: 1, overflowY: 'auto', paddingTop: 8 }}>
        {loading ? (
          <div style={{ padding: '24px 14px', color: 'var(--text-muted)', fontSize: 12, textAlign: 'center' }}>
            <span className="spinner" style={{ display: 'inline-block', marginBottom: 8 }} />
            <br />Loading sessions...
          </div>
        ) : sessions.length === 0 ? (
          <div style={{ padding: '24px 14px', color: 'var(--text-muted)', fontSize: 12, textAlign: 'center' }}>
            No past sessions
          </div>
        ) : (
          groupOrder.map(groupName => {
            const items = groups.get(groupName);
            if (!items || items.length === 0) return null;
            return (
              <div key={groupName}>
                <div style={{
                  padding: '10px 14px 4px',
                  fontSize: 11,
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  letterSpacing: '0.3px',
                }}>
                  {groupName}
                </div>
                {items.map(s => (
                  <button
                    key={s.id}
                    onClick={() => onSelectSession(s.id)}
                    style={{
                      display: 'block',
                      width: 'calc(100% - 16px)',
                      margin: '0 8px',
                      padding: '9px 12px',
                      background: currentSessionId === s.id ? 'var(--bg-tertiary)' : 'transparent',
                      border: 'none',
                      borderRadius: 'var(--radius-md)',
                      cursor: 'pointer',
                      textAlign: 'left',
                      borderLeft: currentSessionId === s.id ? '2px solid var(--accent)' : '2px solid transparent',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => {
                      if (currentSessionId !== s.id) e.currentTarget.style.background = 'var(--bg-hover)';
                    }}
                    onMouseLeave={e => {
                      if (currentSessionId !== s.id) e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <div style={{
                      fontSize: 13,
                      color: 'var(--text-primary)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      lineHeight: 1.4,
                    }}>
                      {s.title || 'Untitled'}
                    </div>
                    <div style={{
                      fontSize: 11,
                      color: 'var(--text-muted)',
                      marginTop: 3,
                      display: 'flex',
                      gap: 8,
                    }}>
                      <span>{s.turnCount} turns</span>
                      <span>{formatTime(s.updatedAt)}</span>
                    </div>
                  </button>
                ))}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
