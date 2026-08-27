import { memo, useState } from 'react';
import type { ChatMessage } from '../types';
import { PixelWolf } from './PixelWolf';
import { HighlightedCode } from './Code';
import { toast } from '../utils/toast';

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

  if (!message.content?.trim()) return null;

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
          <SimpleContent text={message.content} />
        </div>
      </div>
    );
  }

  // Assistant message - full-width card style
  return (
    <div className="slide-up msg-hover" style={{
      padding: '8px 0',
    }}>
      <div style={{
        display: 'flex',
        gap: 12,
        alignItems: 'flex-start',
      }}>
        {/* Avatar */}
        <div style={{
          width: 28,
          height: 28,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: 2,
        }}>
          <PixelWolf state={streaming ? 'working' : 'idle'} px={1.6} title="RUNE" />
        </div>

        {/* Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            className={streaming ? 'streaming-cursor' : undefined}
            style={{
              fontSize: 15,
              lineHeight: 1.7,
              color: 'var(--text-primary)',
              wordBreak: 'break-word',
            }}
          >
            <RenderedContent content={message.content} />
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

// ── Copy button ──

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => toast.error('Copy failed'));
  };

  return (
    <button
      onClick={handleCopy}
      className="copy-btn"
      aria-label="Copy to clipboard"
      style={{
        position: 'static',
        opacity: 1,
        padding: '3px 10px',
        background: 'var(--bg-hover)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        color: 'var(--text-muted)',
        fontSize: 11,
        cursor: 'pointer',
        transition: 'color 0.15s, background 0.15s',
      }}
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

// ── Rendered content (assistant) ──

function RenderedContent({ content }: { content: string }) {
  const blocks = content.split(/(```[\s\S]*?```)/g);

  return (
    <>
      {blocks.map((block, i) => {
        if (block.startsWith('```') && block.endsWith('```')) {
          const inner = block.slice(3, -3);
          const newlineIdx = inner.indexOf('\n');
          const lang = newlineIdx !== -1 ? inner.slice(0, newlineIdx).trim() : '';
          const code = newlineIdx !== -1 ? inner.slice(newlineIdx + 1) : inner;
          return (
            <div key={i} style={{
              margin: '10px 0',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              overflow: 'hidden',
            }}>
              {/* Code header bar */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 12px',
                background: 'var(--bg-tertiary)',
                borderBottom: '1px solid var(--border-subtle)',
              }}>
                <span style={{
                  fontSize: 11,
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 500,
                }}>
                  {lang || 'code'}
                </span>
                <CopyButton text={code} />
              </div>
              <div style={{ background: 'var(--code-bg)' }}>
                <HighlightedCode code={code} lang={lang} lineNumbers={code.includes('\n')} />
              </div>
            </div>
          );
        }
        if (!block || !block.trim()) return null;
        return <RichContent key={i} text={block} />;
      })}
    </>
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

// ── Rich markdown ──

function RichContent({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let listItems: React.ReactNode[] = [];
  let listType: 'ul' | 'ol' | null = null;
  let elemKey = 0;

  const flushList = () => {
    if (listItems.length > 0) {
      const Tag = listType === 'ol' ? 'ol' : 'ul';
      elements.push(
        <Tag key={elemKey++} style={{ margin: '6px 0', paddingLeft: 22, lineHeight: 1.7 }}>
          {listItems}
        </Tag>,
      );
      listItems = [];
      listType = null;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const headerMatch = line.match(/^(#{1,3})\s+(.+)/);
    if (headerMatch) {
      flushList();
      const level = headerMatch[1].length;
      const sizes = [18, 16, 15];
      const weights = [700, 600, 600];
      elements.push(
        <div key={elemKey++} style={{
          fontWeight: weights[level - 1],
          fontSize: sizes[level - 1],
          margin: '16px 0 6px',
          color: 'var(--text-primary)',
          lineHeight: 1.4,
        }}>
          <InlineFormatted text={headerMatch[2]} />
        </div>,
      );
      continue;
    }

    if (line.match(/^\s*[-*]\s+/)) {
      if (listType !== 'ul') { flushList(); listType = 'ul'; }
      listItems.push(<li key={elemKey++}><InlineFormatted text={line.replace(/^\s*[-*]\s+/, '')} /></li>);
      continue;
    }

    if (line.match(/^\s*\d+\.\s+/)) {
      if (listType !== 'ol') { flushList(); listType = 'ol'; }
      listItems.push(<li key={elemKey++}><InlineFormatted text={line.replace(/^\s*\d+\.\s+/, '')} /></li>);
      continue;
    }

    flushList();

    if (line.trim() === '') {
      elements.push(<div key={elemKey++} style={{ height: 8 }} />);
    } else {
      elements.push(
        <span key={elemKey++}>
          <InlineFormatted text={line} />
          {i < lines.length - 1 && '\n'}
        </span>,
      );
    }
  }
  flushList();

  return <div style={{ whiteSpace: 'pre-wrap' }}>{elements}</div>;
}

// ── Inline formatting ──

function InlineFormatted({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g);

  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**'))
          return <strong key={i} style={{ fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**'))
          return <em key={i}>{part.slice(1, -1)}</em>;
        if (part.startsWith('`') && part.endsWith('`'))
          return (
            <code key={i} style={{
              background: 'var(--accent-subtle)',
              color: 'var(--accent-hover)',
              padding: '2px 7px',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.88em',
            }}>
              {part.slice(1, -1)}
            </code>
          );
        const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (linkMatch)
          return <a key={i} href={linkMatch[2]} target="_blank" rel="noopener noreferrer">{linkMatch[1]}</a>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
