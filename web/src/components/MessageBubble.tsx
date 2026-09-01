import { memo, useState } from 'react';
import type { ChatMessage, SentAttachment } from '../types';
import { Markdown } from './Markdown';
import { CopyButton } from './CopyButton';

interface MessageBubbleProps {
  message: ChatMessage;
  /** True while this assistant message is still streaming in. */
  streaming?: boolean;
  /** Re-run the last turn; passed only to the latest assistant message. */
  onRegenerate?: () => void;
  /** Resend an edited version of this user message as a new turn. */
  onEdit?: (text: string) => void;
}

// memo: only the streaming message's ref changes, so other bubbles skip re-parsing markdown.
export const MessageBubble = memo(function MessageBubble({ message, streaming = false, onRegenerate, onEdit }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  // An image sent with no caption is still a message.
  if (!message.content?.trim() && !message.attachments?.length) return null;

  if (isSystem) {
    const isError = message.level === 'error';
    return (
      <div className="fade-in" style={{
        padding: isError ? '8px 12px' : '6px 0',
        margin: isError ? '4px 0' : 0,
        fontSize: isError ? 13 : 12,
        color: isError ? 'var(--danger)' : 'var(--text-muted)',
        fontStyle: isError ? 'normal' : 'italic',
        borderLeft: isError ? '2px solid var(--danger)' : 'none',
        background: isError ? 'var(--danger-subtle)' : 'transparent',
        borderRadius: isError ? 'var(--radius-sm)' : 0,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {message.content}
      </div>
    );
  }

  if (isUser) {
    if (editing) {
      const submit = () => {
        const text = draft.trim();
        setEditing(false);
        if (text) onEdit?.(text);
      };
      return (
        <div className="slide-up" style={{ display: 'flex', justifyContent: 'flex-end', padding: '6px 0' }}>
          <div style={{ maxWidth: '75%', width: '100%', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <textarea
              autoFocus
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
                if (e.key === 'Escape') { setEditing(false); setDraft(message.content); }
              }}
              rows={Math.min(8, draft.split('\n').length + 1)}
              style={{
                width: '100%', resize: 'vertical', padding: '10px 14px',
                borderRadius: 'var(--radius-lg)', background: 'var(--bg-secondary)',
                border: '1px solid var(--accent)', color: 'var(--text-primary)',
                fontSize: 15, lineHeight: 1.6, fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button className="msg-action-btn" onClick={() => { setEditing(false); setDraft(message.content); }}>Cancel</button>
              <button className="msg-action-btn msg-action-btn--primary" onClick={submit}>Send</button>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div className="slide-up msg-hover" style={{
        display: 'flex',
        justifyContent: 'flex-end',
        alignItems: 'center',
        gap: 6,
        padding: '6px 0',
      }}>
        {onEdit && (
          <button
            className="msg-action-btn msg-edit-btn"
            onClick={() => { setDraft(message.content); setEditing(true); }}
            aria-label="Edit and resend"
            title="Edit & resend"
          >Edit</button>
        )}
        <div style={{
          maxWidth: '75%',
          padding: '10px 16px',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border)',
          fontSize: 15,
          lineHeight: 1.6,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          color: 'var(--text-primary)',
        }}>
          {message.attachments?.length ? <Attachments items={message.attachments} /> : null}
          {message.content ? <SimpleContent text={message.content} /> : null}
        </div>
      </div>
    );
  }

  // Assistant message - full-width card style
  return (
    <div className="slide-up msg-hover" style={{
      padding: '8px 0',
    }}>
      <div style={{ minWidth: 0 }}>
        <div
          className={streaming ? 'streaming-cursor' : undefined}
          style={{
            fontSize: 15,
            lineHeight: 1.7,
            color: 'var(--text-primary)',
            wordBreak: 'break-word',
          }}
        >
          <Markdown content={message.content} />
        </div>
        {!streaming && (
          <div className="msg-actions" style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 10 }}>
            <CopyButton text={message.content} />
            {onRegenerate && (
              <button className="msg-action-btn" onClick={onRegenerate} title="Re-run the last turn">
                ↻ Regenerate
              </button>
            )}
            {message.timestamp > 0 && (
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--text-muted)',
              }}>
                {formatClock(message.timestamp)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

function formatClock(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}


// ── Attachments ──

/** Images render inline; a restored message has no data, so it falls back to a chip. */
function Attachments({ items }: { items: SentAttachment[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
      {items.map((att, i) => {
        const isImage = att.mimeType.startsWith('image/') && att.dataUrl;
        return isImage ? (
          <img
            key={i}
            src={att.dataUrl}
            alt={att.name}
            title={att.name}
            style={{
              maxWidth: 240, maxHeight: 240, borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)', display: 'block',
            }}
          />
        ) : (
          <span
            key={i}
            title={att.name}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 8px', borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-secondary)', border: '1px solid var(--border)',
              fontSize: 12, color: 'var(--text-secondary)', maxWidth: 220,
            }}
          >
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {att.name}
            </span>
          </span>
        );
      })}
    </div>
  );
}


// ── Simple content (user messages) ──

function SimpleContent({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('`') && part.endsWith('`')) {
          return (
            <code key={i} style={{
              background: 'rgba(255,255,255,0.08)',
              padding: '2px 6px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.88em',
              color: 'inherit',
            }}>
              {part.slice(1, -1)}
            </code>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
