import { createContext, useContext, type ReactNode } from 'react';
import { artifactHref } from '../utils/artifactLinks';
import { HighlightedCode } from './Code';
import { CopyButton } from './CopyButton';

// Lightweight markdown renderer shared by chat messages and the file preview.
// Handles fenced code (syntax-highlighted), headers (h1–h3), ordered/unordered
// lists, and inline bold/italic/code/links. Deliberately small — enough to read
// assistant output and .md files at a glance, not a full CommonMark engine.

const ArtifactSession = createContext<string | undefined>(undefined);

export function Markdown({ content, sessionId }: { content: string; sessionId?: string }) {
  const blocks = content.split(/(```[\s\S]*?```)/g);
  return (
    <ArtifactSession.Provider value={sessionId}>
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
    </ArtifactSession.Provider>
  );
}

const HEADER_SIZES = [20, 17, 15.5, 14, 13, 12.5];
const HEADER_WEIGHTS = [700, 700, 600, 600, 600, 600];
const LIST_ITEM_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

function RichContent({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: ReactNode[] = [];
  let elemKey = 0;
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const buf = paragraph;
    paragraph = [];
    elements.push(
      <p key={elemKey++} className="md-p">
        {buf.map((l, li) => (
          <span key={li}>
            <InlineFormatted text={l} />
            {li < buf.length - 1 && <br />}
          </span>
        ))}
      </p>,
    );
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // GitHub-style table: a row of cells, then a |---|---| separator.
    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushParagraph();
      const header = splitCells(line);
      const aligns = splitCells(lines[i + 1]).map(cellAlign);
      const rows: string[][] = [];
      let j = i + 2;
      while (j < lines.length && lines[j].includes('|') && lines[j].trim() !== '') {
        rows.push(splitCells(lines[j]));
        j++;
      }
      elements.push(<MdTable key={elemKey++} header={header} aligns={aligns} rows={rows} />);
      i = j - 1;
      continue;
    }

    // Horizontal rule: a line of only ---, ***, or ___.
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flushParagraph();
      elements.push(<hr key={elemKey++} className="md-hr" />);
      continue;
    }

    // Blockquote: one or more consecutive '>' lines.
    if (/^\s*>\s?/.test(line)) {
      flushParagraph();
      const quote: string[] = [];
      let j = i;
      while (j < lines.length && /^\s*>\s?/.test(lines[j])) {
        quote.push(lines[j].replace(/^\s*>\s?/, ''));
        j++;
      }
      elements.push(
        <blockquote key={elemKey++} className="md-quote">
          {quote.map((q, qi) => <div key={qi}><InlineFormatted text={q} /></div>)}
        </blockquote>,
      );
      i = j - 1;
      continue;
    }

    // Headers h1–h6.
    const headerMatch = line.match(/^(#{1,6})\s+(.+)/);
    if (headerMatch) {
      flushParagraph();
      const level = headerMatch[1].length;
      elements.push(
        <div key={elemKey++} className="md-h" style={{
          fontWeight: HEADER_WEIGHTS[level - 1], fontSize: HEADER_SIZES[level - 1],
        }}>
          <InlineFormatted text={headerMatch[2]} />
        </div>,
      );
      continue;
    }

    // List block (nested, ordered/unordered, task items).
    if (LIST_ITEM_RE.test(line)) {
      flushParagraph();
      const [list, next] = parseList(lines, i, line.match(LIST_ITEM_RE)![1].length);
      elements.push(renderList(list, elemKey++));
      i = next - 1;
      continue;
    }

    if (line.trim() === '') {
      flushParagraph();
    } else {
      paragraph.push(line);
    }
  }
  flushParagraph();

  return <>{elements}</>;
}

interface MdList { ordered: boolean; items: MdItem[]; }
interface MdItem { text: string; task: boolean | null; sub: MdList | null; }

// Consume a run of list lines at `baseIndent` into a nested structure and
// return the index where the list ended. A deeper-indented line becomes a
// sub-list of the item above it.
function parseList(lines: string[], start: number, baseIndent: number): [MdList, number] {
  const ordered = /^\s*\d+[.)]\s/.test(lines[start]);
  const list: MdList = { ordered, items: [] };
  let i = start;
  while (i < lines.length) {
    const m = lines[i].match(LIST_ITEM_RE);
    if (!m) break;
    const indent = m[1].length;
    if (indent < baseIndent) break;
    if (indent > baseIndent) {
      const [sub, next] = parseList(lines, i, indent);
      if (list.items.length) list.items[list.items.length - 1].sub = sub;
      i = next;
      continue;
    }
    let text = m[3];
    let task: boolean | null = null;
    const t = text.match(/^\[([ xX])\]\s+(.*)$/);
    if (t) { task = t[1].toLowerCase() === 'x'; text = t[2]; }
    list.items.push({ text, task, sub: null });
    i++;
  }
  return [list, i];
}

function renderList(list: MdList, key: number): ReactNode {
  const Tag = list.ordered ? 'ol' : 'ul';
  return (
    <Tag key={key} className="md-list">
      {list.items.map((it, i) => (
        <li key={i} className={it.task !== null ? 'md-task' : undefined}>
          {it.task !== null && (
            <input type="checkbox" checked={it.task} readOnly aria-hidden="true" />
          )}
          <InlineFormatted text={it.text} />
          {it.sub && renderList(it.sub, 1)}
        </li>
      ))}
    </Tag>
  );
}

function InlineFormatted({ text }: { text: string }) {
  const sessionId = useContext(ArtifactSession);
  const parts = text.split(/(\*\*[^*]+\*\*|~~[^~]+~~|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**'))
          return <strong key={i} style={{ fontWeight: 600 }}><InlineFormatted text={part.slice(2, -2)} /></strong>;
        if (part.startsWith('~~') && part.endsWith('~~'))
          return <del key={i} style={{ opacity: 0.7 }}><InlineFormatted text={part.slice(2, -2)} /></del>;
        if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**'))
          return <em key={i}><InlineFormatted text={part.slice(1, -1)} /></em>;
        if (part.startsWith('`') && part.endsWith('`'))
          return (
            <code key={i} style={{
              background: 'var(--accent-subtle)', color: 'var(--accent-hover)',
              padding: '2px 7px', borderRadius: 'var(--radius-sm)', fontSize: '0.88em',
            }}>{part.slice(1, -1)}</code>
          );
        const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (linkMatch) {
          const href = artifactHref(linkMatch[2], sessionId);
          return href
            ? <a key={i} href={href} target="_blank" rel="noopener noreferrer">{linkMatch[1]}</a>
            : <span key={i}>{linkMatch[1]}</span>;
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

type Align = 'left' | 'center' | 'right';

// A row of only dashes/colons/pipes marks the header separator of a GFM table.
function isTableSeparator(line: string): boolean {
  return line.includes('-')
    && /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/.test(line);
}

function splitCells(line: string): string[] {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  return s.split('|').map(c => c.trim());
}

function cellAlign(sep: string): Align {
  const l = sep.startsWith(':');
  const r = sep.endsWith(':');
  if (l && r) return 'center';
  if (r) return 'right';
  return 'left';
}

function MdTable({ header, aligns, rows }: { header: string[]; aligns: Align[]; rows: string[][] }) {
  return (
    <div className="md-table-scroll">
      <table className="md-table">
        <thead>
          <tr>
            {header.map((h, i) => (
              <th key={i} style={{ textAlign: aligns[i] ?? 'left' }}>
                <InlineFormatted text={h} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td key={ci} style={{ textAlign: aligns[ci] ?? 'left' }}>
                  <InlineFormatted text={c} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
