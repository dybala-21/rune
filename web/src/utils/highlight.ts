// Dependency-free syntax highlighter. A single-pass scanner turns source into
// typed tokens; the caller wraps each in a themed span. Code is a structured
// format, so a regex/character scanner is the right tool here — this is not
// natural-language guessing. Fidelity is deliberately modest (strings,
// comments, numbers, keywords, function-ish idents); it is meant to make code
// readable at a glance, not to be a full grammar.

export type TokenKind =
  | 'plain' | 'comment' | 'string' | 'number' | 'keyword'
  | 'boolean' | 'function' | 'operator';

export interface Token { kind: TokenKind; text: string; }

// Keywords shared across the languages a general assistant most often emits.
// A superset is fine: colouring a word that happens to match in another
// language is harmless, and keeps this free of per-language tables.
const KEYWORDS = new Set([
  'abstract', 'and', 'as', 'async', 'await', 'break', 'case', 'catch',
  'class', 'const', 'continue', 'def', 'default', 'del', 'delete', 'do',
  'elif', 'else', 'enum', 'except', 'export', 'extends', 'finally', 'fn',
  'for', 'from', 'func', 'function', 'global', 'go', 'if', 'impl', 'import',
  'in', 'instanceof', 'interface', 'is', 'lambda', 'let', 'match', 'mod',
  'mut', 'namespace', 'new', 'nonlocal', 'not', 'or', 'package', 'pass',
  'private', 'protected', 'public', 'pub', 'raise', 'return', 'select',
  'self', 'static', 'struct', 'super', 'switch', 'template', 'this', 'throw',
  'trait', 'try', 'type', 'typeof', 'use', 'var', 'void', 'when', 'where',
  'while', 'with', 'yield',
]);
const BOOLEANS = new Set([
  'true', 'false', 'null', 'nil', 'none', 'undefined', 'True', 'False',
  'None', 'NULL',
]);

// Languages whose line comments start with '#'. Everything else uses '//'.
const HASH_COMMENT_LANGS = new Set([
  'python', 'py', 'ruby', 'rb', 'shell', 'sh', 'bash', 'zsh', 'yaml', 'yml',
  'toml', 'ini', 'r', 'perl', 'makefile', 'dockerfile',
]);

const isIdentStart = (c: string) => /[A-Za-z_$]/.test(c);
const isIdentPart = (c: string) => /[A-Za-z0-9_$]/.test(c);
const isDigit = (c: string) => c >= '0' && c <= '9';

export function tokenize(src: string, lang = ''): Token[] {
  const hashComments = HASH_COMMENT_LANGS.has(lang.toLowerCase());
  const tokens: Token[] = [];
  let i = 0;
  const n = src.length;
  let plainStart = 0;

  const flushPlain = (end: number) => {
    if (end > plainStart) tokens.push({ kind: 'plain', text: src.slice(plainStart, end) });
  };

  while (i < n) {
    const c = src[i];
    const next = i + 1 < n ? src[i + 1] : '';

    // Line comment
    if ((c === '/' && next === '/') || (hashComments && c === '#')) {
      flushPlain(i);
      let j = i;
      while (j < n && src[j] !== '\n') j++;
      tokens.push({ kind: 'comment', text: src.slice(i, j) });
      i = j; plainStart = i; continue;
    }
    // Block comment
    if (c === '/' && next === '*') {
      flushPlain(i);
      let j = i + 2;
      while (j < n && !(src[j] === '*' && src[j + 1] === '/')) j++;
      j = Math.min(n, j + 2);
      tokens.push({ kind: 'comment', text: src.slice(i, j) });
      i = j; plainStart = i; continue;
    }
    // String (single, double, backtick) with escape handling
    if (c === '"' || c === "'" || c === '`') {
      flushPlain(i);
      const quote = c;
      let j = i + 1;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === quote) { j++; break; }
        if (src[j] === '\n' && quote !== '`') { break; }
        j++;
      }
      tokens.push({ kind: 'string', text: src.slice(i, j) });
      i = j; plainStart = i; continue;
    }
    // Number
    if (isDigit(c) || (c === '.' && isDigit(next))) {
      flushPlain(i);
      let j = i;
      while (j < n && /[0-9a-fA-FxXbBoO._]/.test(src[j])) j++;
      tokens.push({ kind: 'number', text: src.slice(i, j) });
      i = j; plainStart = i; continue;
    }
    // Identifier / keyword
    if (isIdentStart(c)) {
      flushPlain(i);
      let j = i + 1;
      while (j < n && isIdentPart(src[j])) j++;
      const word = src.slice(i, j);
      // A trailing '(' marks a call/def — colour it as a function name.
      let k = j;
      while (k < n && (src[k] === ' ' || src[k] === '\t')) k++;
      let kind: TokenKind = 'plain';
      if (KEYWORDS.has(word)) kind = 'keyword';
      else if (BOOLEANS.has(word)) kind = 'boolean';
      else if (src[k] === '(') kind = 'function';
      tokens.push({ kind, text: word });
      i = j; plainStart = i; continue;
    }
    i++;
  }
  flushPlain(n);
  return tokens;
}

// Maps a file path's extension to a language hint for tokenize(). Only affects
// comment style (# vs //); unknown extensions fall back to c-style comments.
const EXT_LANG: Record<string, string> = {
  py: 'python', rb: 'ruby', sh: 'bash', bash: 'bash', zsh: 'bash',
  yml: 'yaml', yaml: 'yaml', toml: 'toml', ini: 'ini', r: 'r', pl: 'perl',
  ts: 'ts', tsx: 'ts', js: 'js', jsx: 'js', go: 'go', rs: 'rust',
  java: 'java', c: 'c', h: 'c', cpp: 'cpp', cc: 'cpp', cs: 'csharp',
  php: 'php', swift: 'swift', kt: 'kotlin', md: 'markdown', json: 'json',
};
export function langFromPath(path: string): string {
  const base = path.split('/').pop() || '';
  if (base.toLowerCase() === 'dockerfile') return 'dockerfile';
  if (base.toLowerCase() === 'makefile') return 'makefile';
  const ext = base.includes('.') ? base.slice(base.lastIndexOf('.') + 1).toLowerCase() : '';
  return EXT_LANG[ext] || '';
}
