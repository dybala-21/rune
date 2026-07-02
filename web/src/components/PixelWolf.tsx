import { useEffect, useState } from 'react';

/**
 * RUNE's 8-bit ice-wolf mascot. The brow rune colour encodes agent state;
 * the sprite blinks on its own unless `still` or reduced-motion.
 * Design reference: web/prototypes/pixel-wolf.html.
 */
export type WolfState =
  | 'idle'
  | 'thinking'
  | 'working'
  | 'passed'
  | 'failed'
  | 'warning';

// Character palette: fixed hex on purpose — the mascot must not retheme
// with the UI tokens. Only the brow rune ('c') follows theme state colors.
const BODY: Record<string, string> = {
  o: '#37687f', // outline
  b: '#dceffa', // body (ice)
  s: '#a9d6ee', // shade
  e: '#0d2733', // eye
  n: '#0d2733', // nose
  p: '#e79ab2', // blush
  w: '#ffffff', // eye glint
};

const RUNE_COLOR: Record<WolfState, string> = {
  idle: 'var(--rune-idle)',
  thinking: 'var(--rune-idle)',
  working: 'var(--rune-working)',
  passed: 'var(--rune-passed)',
  failed: 'var(--danger)',
  warning: 'var(--danger)',
};

// 16 cells wide; renderers assume this width.
const BASE = [
  '...o........o...',
  '..ooo......ooo..',
  '..obo......obo..',
  '.obboooooooobbo.',
  'obbbbbbccbbbbbo.',
  'obbbbbbcbbbbbbo.',
  'obbewbbbbbbwebbo',
  'obbeebbbbbbeebbo',
  'obpbbbbnnbbbbpbo',
  'obbbbbbnnbbbbbso',
  '.obbbbbbbbbbbso.',
  '.obbbbbbbbbbbo..',
  '.obboobbboobbo..',
  '.oooo.ooo.oooo..',
];

const EYES_OPEN: [number, string][] = [
  [6, 'obbewbbbbbbwebbo'],
  [7, 'obbeebbbbbbeebbo'],
];
const EYES_HAPPY: [number, string][] = [
  [6, 'obbbbbbbbbbbbbbo'],
  [7, 'obbeebbbbbbeebbo'],
];

function spriteFor(state: WolfState, blink: boolean): string[] {
  const rows = BASE.slice();
  const eyes = blink || state === 'passed' ? EYES_HAPPY : EYES_OPEN;
  for (const [i, row] of eyes) rows[i] = row;
  return rows;
}

interface PixelWolfProps {
  state?: WolfState;
  /** pixel size of one sprite cell (sprite is 16 cells wide) */
  px?: number;
  /** disable the idle blink */
  still?: boolean;
  title?: string;
  className?: string;
}

export function PixelWolf({
  state = 'idle',
  px = 2,
  still = false,
  title,
  className,
}: PixelWolfProps) {
  const [blink, setBlink] = useState(false);

  useEffect(() => {
    if (still) return;
    const reduce =
      typeof matchMedia !== 'undefined' &&
      matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const loop = () => {
      if (!alive) return;
      timer = setTimeout(() => {
        setBlink(true);
        timer = setTimeout(() => {
          setBlink(false);
          loop();
        }, 140);
      }, 4200 + Math.floor(Math.random() * 1600));
    };
    loop();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [still]);

  const rows = spriteFor(state, blink);
  const w = 16;
  const h = rows.length;
  const rune = RUNE_COLOR[state];

  const rects: React.ReactNode[] = [];
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '.' || ch === undefined) continue;
      const fill = ch === 'c' ? rune : BODY[ch];
      if (!fill) continue;
      rects.push(
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width={1.03}
          height={1.03}
          fill={fill}
        />,
      );
    }
  }

  return (
    <svg
      className={className}
      viewBox={`0 0 ${w} ${h}`}
      width={w * px}
      height={h * px}
      shapeRendering="crispEdges"
      style={{ imageRendering: 'pixelated', display: 'block' }}
      role="img"
      aria-label={title ?? `RUNE (${state})`}
    >
      {title ? <title>{title}</title> : null}
      {rects}
    </svg>
  );
}

export default PixelWolf;
