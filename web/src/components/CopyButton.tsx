import { useState } from 'react';
import { toast } from '../utils/toast';

export function CopyButton({ text }: { text: string }) {
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
