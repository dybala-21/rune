import './RuneMark.css';

export type MarkState = 'idle' | 'thinking' | 'working' | 'passed' | 'failed' | 'warning';

interface RuneMarkProps {
  state?: MarkState;
  size?: number;
  title?: string;
}

export function RuneMark({ state = 'idle', size = 24, title = 'Rune' }: RuneMarkProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none"
      className="rune-mark" data-state={state} role="img" aria-label={title}>
      <path className="rune-wolf-body" d="M19 42C32 29 59 25 75 35c14 8 20 23 16 38-4 15-24 23-47 21C23 93 10 85 9 71 8 59 12 49 19 42Z" />
      <g className="rune-wolf-head">
        <path className="rune-wolf-body" d="M18 49c-1-8 0-24 5-27 5-1 13 5 17 8 4-2 7-3 12-3 4-9 9-16 13-15 5 1 10 15 12 24 8 6 11 14 9 22-3 12-19 20-37 21-19 0-33-11-34-22-1-3 1-6 3-8Z" />
        <path className="rune-wolf-cream" d="M23 40c-1-9 0-12 2-11l9 5-11 6Zm36-12 6-8c2 3 4 9 5 13l-11-5ZM26 55c5-6 15-6 22-3 6-2 15-8 23-6 11 1 10 11 4 17-7 8-28 14-40 7-2-8-6-12-9-15Z" />
        <g className="rune-wolf-eyes">
          <ellipse className="rune-wolf-ink" cx="41" cy="53" rx="2.7" ry="3.7" transform="rotate(-12 41 53)" />
          <ellipse className="rune-wolf-ink" cx="62" cy="49" rx="2.7" ry="3.7" transform="rotate(-12 62 49)" />
        </g>
        <path className="rune-wolf-ink" d="M51 58c2-1 5-2 7-1 1 1 0 4-2 5-1 1-5-2-5-4Z" />
      </g>
      <g className="rune-wolf-tail">
        <path className="rune-wolf-body" d="M14 63c7-6 19-4 29 2 6 4 10 10 12 15 7-10 17-18 31-15 2 14-15 29-38 29-26 1-38-9-38-21 0-4 1-7 4-10Z" />
        <path className="rune-wolf-cream" d="M67 63c9-5 19-2 20 4-1 10-8 18-18 22 5-6 6-12 5-16-6-2-11 0-14 1 2-5 5-9 7-11Z" />
      </g>
    </svg>
  );
}

export default RuneMark;
