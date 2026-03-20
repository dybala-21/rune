import { useEffect, useRef, useState } from 'react';
import type { ToolCall } from '../types';
import { normalizeToolName } from '../utils/tooling';

/** Tool category color mapping */
export function getToolColor(name: string): string {
  const normalized = normalizeToolName(name);
  if (normalized.startsWith('file.')) return 'var(--tool-file)';
  if (normalized === 'bash') return 'var(--tool-bash)';
  if (normalized.startsWith('web.')) return 'var(--tool-web)';
  if (normalized.startsWith('browser.')) return 'var(--tool-browser)';
  if (normalized.startsWith('memory.')) return 'var(--tool-memory)';
  if (normalized === 'think') return 'var(--tool-think)';
  if (normalized.startsWith('code.')) return 'var(--tool-code)';
  if (normalized.startsWith('delegate.')) return 'var(--tool-delegate)';
  if (normalized === 'project.map' || name === 'project_map') return 'var(--tool-project)';
  return 'var(--tool-default)';
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function getToolSummary(toolCall: ToolCall): string | null {
  const { toolName, args } = toolCall;
  const normalized = normalizeToolName(toolName);
  if (normalized === 'file.read' && args.path) return String(args.path).split('/').pop() ?? null;
  if (normalized === 'file.write' && args.path) return String(args.path).split('/').pop() ?? null;
  if (normalized === 'file.edit' && args.path) return String(args.path).split('/').pop() ?? null;
  if (normalized === 'file.search' && args.pattern) return `"${args.pattern}"`;
  if (normalized === 'file.list' && args.path) return String(args.path).split('/').pop() ?? null;
  if (normalized === 'bash' && args.command) {
    const cmd = String(args.command);
    return cmd.length > 50 ? cmd.slice(0, 47) + '...' : cmd;
  }
  if (normalized === 'web.search' && args.query) return `"${args.query}"`;
  if (normalized === 'memory.search' && args.query) return `"${args.query}"`;
  return null;
}

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
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

interface ToolCallCardProps {
  toolCall: ToolCall;
  isLatest?: boolean;
}

export function ToolCallCard({ toolCall, isLatest = false }: ToolCallCardProps) {
  if (!toolCall.toolName?.trim()) return null;

  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);

  const prevIsLatest = useRef(isLatest);
  useEffect(() => {
    if (prevIsLatest.current && !isLatest) {
      setManualExpanded(null);
    }
    prevIsLatest.current = isLatest;
  }, [isLatest]);

  const hasResult = toolCall.result !== undefined;
  const isPending = !hasResult;
  const expanded = manualExpanded ?? false;
  const toolColor = getToolColor(toolCall.toolName);

  const handleToggle = () => {
    setManualExpanded(!(manualExpanded ?? false));
  };

  const summary = getToolSummary(toolCall);

  return (
    <div className="fade-in" style={{
      margin: '2px 0',
      borderLeft: `2px solid ${toolColor}`,
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={handleToggle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') handleToggle(); }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 12px',
          cursor: 'pointer',
          userSelect: 'none',
          fontSize: 13,
        }}
      >
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--text-secondary)',
          fontWeight: 500,
        }}>
          {toolCall.toolName}
        </span>
        {summary && (
          <span style={{
            color: 'var(--text-muted)',
            fontSize: 12,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
            minWidth: 0,
          }}>
            {summary}
          </span>
        )}
        {!summary && <span style={{ flex: 1 }} />}

        {isPending ? (
          <span className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
        ) : (
          <>
            {toolCall.durationMs != null && (
              <span style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                flexShrink: 0,
              }}>
                {formatDuration(toolCall.durationMs)}
              </span>
            )}
            <span
              className="status-dot"
              style={{
                width: 6,
                height: 6,
                background: (toolCall.success ?? true) ? 'var(--success)' : 'var(--danger)',
              }}
            />
          </>
        )}

        <span style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          transform: expanded ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
          flexShrink: 0,
        }}>
          {'\u25BC'}
        </span>
      </div>

      {expanded && (
        <div className="expand-content" style={{
          padding: '4px 12px 10px',
          borderTop: '1px solid var(--border-subtle)',
        }}>
          {Object.keys(toolCall.args).length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                marginBottom: 4,
                fontWeight: 600,
                letterSpacing: '0.3px',
              }}>
                Arguments
              </div>
              <div className="code-block-wrapper">
                <CopyButton text={JSON.stringify(toolCall.args, null, 2)} />
                <pre style={{
                  background: 'var(--code-bg)',
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  overflow: 'auto',
                  maxHeight: 200,
                  border: '1px solid var(--border-subtle)',
                  whiteSpace: 'pre-wrap',
                  color: 'var(--text-primary)',
                  margin: 0,
                  lineHeight: 1.5,
                }}>
                  {JSON.stringify(toolCall.args, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {hasResult && (
            <div>
              <div style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                marginBottom: 4,
                fontWeight: 600,
                letterSpacing: '0.3px',
              }}>
                Result
              </div>
              <div className="code-block-wrapper">
                <CopyButton text={toolCall.result ?? ''} />
                <pre style={{
                  background: 'var(--code-bg)',
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  overflow: 'auto',
                  maxHeight: 300,
                  border: '1px solid var(--border-subtle)',
                  whiteSpace: 'pre-wrap',
                  color: 'var(--text-primary)',
                  margin: 0,
                  lineHeight: 1.5,
                }}>
                  {formatResult(toolCall.result ?? '')}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatResult(text: string): string {
  const trimmed = text.trim();
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2).slice(0, 2000);
    } catch {
      // not valid JSON
    }
  }
  return truncate(text, 2000);
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + '\n... (truncated)';
}
