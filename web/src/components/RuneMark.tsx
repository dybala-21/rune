import './RuneMark.css';

export type MarkState = 'idle' | 'thinking' | 'working' | 'passed' | 'failed' | 'warning';

interface RuneMarkProps {
  state?: MarkState;
  size?: number;
  title?: string;
}

export function RuneMark({ state = 'idle', size = 24, title = 'Rune' }: RuneMarkProps) {
  const portrait = size <= 32;

  return (
    <svg width={size} height={size} viewBox={portrait ? '138 70 520 520' : '-55 62 845 845'}
      className="rune-mark" data-state={state} role="img" aria-label={title}
      fill="none" stroke="currentColor" strokeWidth={portrait ? 20 : 14}
      strokeLinecap="round" strokeLinejoin="round">
      {!portrait && <g>
        <path d="M242 808c-59 14-114-19-134-64l28 10c-28-37-25-90 1-119 6 40 36 41 62 47 24 5 39 14 52 34" />
        <path d="M376 562c-9 36-63 84-108 139-37 43-45 74-39 107 7 36 42 57 88 59l157 1c43 0 47-33 9-43l-41-3c-24-16-38-39-44-74" />
        <path d="M381 737c-15 30-11 64 4 85M302 862c34 3 72-11 92-29M429 865c-1-12-5-19-13-25m41 25c1-11-3-18-9-23" />
        <path d="M471 557c1 13 4 20 10 27m38 106c-1 52-18 99-26 129 37 2 57 14 59 32 1 13-11 16-38 16h-31m-14-48c-15-35-23-71-27-113" />
        <path d="M402 602c1-21 23-34 48-29 29 6 49 26 59 58 12 37 8 68-10 77-19 9-34-8-40-31l-7-28c-28-3-52-16-50-47Z" />
        <path d="M410 613q3-12 14-15m-1 28q3-11 11-16" strokeWidth="8" />
      </g>}
      <g className="rune-wolf-head">
        <path fill="currentColor" stroke="none" d="M219 453c-59-47-66-196-27-322 15-50 44-42 78 4 36 48 68 105 94 156Z" />
        <path d="M219 452c31-66 71-140 143-175 25-64 51-93 107-64 40 22 66 38 106 58 40 20 55 51 47 78-12 37-53 31-82 14 5 20 17 32 26 40 5 49-37 94-90 120-21 11-37 22-50 43l-48-7c-28 1-69-3-91-19-24 4-58-8-71-23 0-11 6-19 17-26l-39-2c3-14 12-28 25-37Z" />
        <path d="M407 272c19-13 37-9 48 3-5-13-12-23-18-31 14 5 27 22 39 43 21 35 46 65 77 79" fill="currentColor" strokeWidth="7" />
        <path fill="currentColor" stroke="none" d="M379 306c-18 20-10 44 13 42 7 0 12-3 17-7-11 23-40 20-47 0-6-16 1-29 17-35Z" />
        <g fill="currentColor" stroke="none">
          <path className="rune-wolf-eye" d="M329 451c-3-25 8-50 30-55 18-3 33 18 31 39-2 20-16 25-27 13-10-10-13-28-8-40-13 6-22 24-26 43Z" />
          <path className="rune-wolf-eye" d="M451 393c-4-24 4-43 17-45 14-3 23 9 26 15-8-4-12-4-15-3 13 8 20 31 11 42-12 15-35-9-33-30-3 6-5 13-6 21Z" />
          <path d="M524 412c9-6 28-10 40-9 0 14-5 25-11 27-9 3-31-4-32-11 0-3 1-5 3-7Z" />
        </g>
      </g>
    </svg>
  );
}

export default RuneMark;
