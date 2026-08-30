import { memo, useMemo } from 'react';
import { tokenize } from '../utils/highlight';

// Renders source with lightweight syntax colouring and an optional line-number
// gutter. Tokens are React children, so their text is escaped by React — no
// dangerouslySetInnerHTML, no injection surface. The gutter is a separate
// <pre> of numbers; sharing line-height with the code column keeps them
// aligned even when a token (block comment, template string) spans lines.

interface Props {
  code: string;
  lang?: string;
  lineNumbers?: boolean;
}

export const HighlightedCode = memo(function HighlightedCode(
  { code, lang = '', lineNumbers = false }: Props,
) {
  const tokens = useMemo(() => tokenize(code, lang), [code, lang]);
  const gutter = useMemo(() => {
    if (!lineNumbers) return '';
    const lines = code.split('\n').length;
    return Array.from({ length: lines }, (_, i) => i + 1).join('\n');
  }, [code, lineNumbers]);

  return (
    <div className="code-hl-scroll">
      <div className="code-hl-row">
        {lineNumbers && <pre className="code-hl-gutter" aria-hidden="true">{gutter}</pre>}
        <pre className="code-hl-body">
          <code>
            {tokens.map((t, i) =>
              t.kind === 'plain'
                ? t.text
                : <span key={i} className={`tok-${t.kind}`}>{t.text}</span>,
            )}
          </code>
        </pre>
      </div>
    </div>
  );
});
