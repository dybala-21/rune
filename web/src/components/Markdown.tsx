import type { ReactNode } from 'react';
import { HighlightedCode } from './Code';
import { CopyButton } from './CopyButton';

// Lightweight markdown renderer shared by chat messages and the file preview.
// Handles fenced code (syntax-highlighted), headers (h1–h3), ordered/unordered
// lists, and inline bold/italic/code/links. Deliberately small — enough to read
// assistant output and .md files at a glance, not a full CommonMark engine.

export function Markdown({ content }: { content: string }) {
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
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '6px 12px', background: 'var(--bg-tertiary)',
                borderBottom: '1px solid var(--border-subtle)',
              }}>
                <span style={{
                  fontSize: 11, color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)', fontWeight: 500,
                }}>{lang || 'code'}</span>
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

function RichContent({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: ReactNode[] = [];
  let listItems: ReactNode[] = [];
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
          fontWeight: weights[level - 1], fontSize: sizes[level - 1],
          margin: '16px 0 6px', color: 'var(--text-primary)', lineHeight: 1.4,
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
              background: 'var(--accent-subtle)', color: 'var(--accent-hover)',
              padding: '2px 7px', borderRadius: 'var(--radius-sm)', fontSize: '0.88em',
            }}>{part.slice(1, -1)}</code>
          );
        const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (linkMatch)
          return <a key={i} href={linkMatch[2]} target="_blank" rel="noopener noreferrer">{linkMatch[1]}</a>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
