import { useState } from 'react';
import type { ThinkingBlock } from '../types';

interface ThinkingBlockViewProps {
  block: ThinkingBlock;
}

export function ThinkingBlockView({ block }: ThinkingBlockViewProps) {
  const [expanded, setExpanded] = useState(false);

  if (!block.text?.trim()) return null;

  const preview = block.text.trim().slice(0, 60).replace(/\n/g, ' ');

  return (
    <div
      className="fade-in"
      style={{
        margin: '2px 0',
        borderRadius: 'var(--radius-md)',
        overflow: 'hidden',
      }}
    >
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setExpanded(!expanded); }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '5px 8px',
          color: 'var(--text-muted)',
          fontSize: 12,
          cursor: 'pointer',
          userSelect: 'none',
          borderRadius: 'var(--radius-sm)',
          transition: 'background 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'var(--bg-secondary)'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
      >
        <span style={{ fontSize: 11, fontWeight: 600, fontStyle: 'italic', flexShrink: 0 }}>
          thinking
        </span>
        {!expanded && (
          <span style={{
            flex: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            fontStyle: 'italic',
            opacity: 0.6,
          }}>
            {preview}{block.text.trim().length > 60 ? '...' : ''}
          </span>
        )}
        <span style={{
          fontSize: 10,
          transform: expanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
          flexShrink: 0,
        }}>
          {'\u25BC'}
        </span>
      </div>

      {expanded && (
        <div className="expand-content" style={{
          padding: '8px 12px',
          color: 'var(--text-secondary)',
          fontSize: 13,
          lineHeight: 1.6,
          whiteSpace: 'pre-wrap',
          maxHeight: 300,
          overflow: 'auto',
          fontStyle: 'italic',
          background: 'var(--bg-secondary)',
          borderRadius: 'var(--radius-sm)',
          margin: '2px 0',
        }}>
          {block.text}
        </div>
      )}
    </div>
  );
}
