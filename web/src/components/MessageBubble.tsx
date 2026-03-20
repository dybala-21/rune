import { useState } from 'react';
import type { ChatMessage } from '../types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (!message.content?.trim()) return null;

  if (isSystem) {
    return (
      <div className="fade-in" style={{
        padding: '6px 0',
        fontSize: 12,
        color: 'var(--text-muted)',
        fontStyle: 'italic',
      }}>
        {message.content}
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="slide-up" style={{
        display: 'flex',
        justifyContent: 'flex-end',
        padding: '6px 0',
      }}>
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
    <div className="slide-up" style={{
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
          borderRadius: 'var(--radius-md)',
          background: 'linear-gradient(135deg, var(--accent), #8b5cf6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: 2,
        }}>
          <span style={{
            fontSize: 12,
            fontWeight: 700,
            color: 'white',
            letterSpacing: '0.5px',
          }}>R</span>
        </div>

        {/* Content */}
        <div style={{
          flex: 1,
          minWidth: 0,
          fontSize: 15,
          lineHeight: 1.7,
          color: 'var(--text-primary)',
          wordBreak: 'break-word',
        }}>
          <RenderedContent content={message.content} />
        </div>
      </div>
    </div>
  );
}

// ── Copy button ──

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
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
              <pre style={{
                padding: '12px 14px',
                background: 'var(--code-bg)',
                fontSize: 13,
                overflow: 'auto',
                whiteSpace: 'pre',
                margin: 0,
                border: 'none',
                borderRadius: 0,
                color: 'var(--text-primary)',
                lineHeight: 1.5,
              }}>
                <code style={{ background: 'transparent', padding: 0, color: 'inherit' }}>{code}</code>
              </pre>
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
