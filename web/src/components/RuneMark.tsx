/**
 * RUNE's mark: three concentric arcs with staggered gaps and one bright
 * leading segment. Same geometry as desktop/build/icon.png.
 */
export type MarkState =
  | 'idle'
  | 'thinking'
  | 'working'
  | 'passed'
  | 'failed'
  | 'warning';

// Only the leading segment takes the state colour — the arcs stay fixed so the
// silhouette doesn't change.
const LEAD_COLOR: Record<MarkState, string> = {
  idle: 'var(--rune-idle)',
  thinking: 'var(--rune-idle)',
  working: 'var(--rune-working)',
  passed: 'var(--rune-passed)',
  failed: 'var(--danger)',
  warning: 'var(--danger)',
};

const STRUCT = '#2C7E9B';

/** Dash pattern drawing `count` arcs of `arcDeg`, evenly spaced around radius `r`. */
function dashes(r: number, arcDeg: number, count: number): string {
  const circumference = 2 * Math.PI * r;
  const step = circumference / count;
  const on = circumference * (arcDeg / 360);
  return `${on} ${step - on}`;
}

interface RuneMarkProps {
  state?: MarkState;
  /** Rendered box in px (the mark is square). */
  size?: number;
  title?: string;
}

export function RuneMark({ state = 'idle', size = 24, title }: RuneMarkProps) {
  // viewBox is 32x32, centred on 16,16.
  const r1 = 13.2;
  const r2 = 9.9;
  const r3 = 6.6;
  const w = 1.7;
  const lead = LEAD_COLOR[state];
  const c1 = 2 * Math.PI * r1;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label={title ?? 'RUNE'}
      style={{ display: 'block', flexShrink: 0 }}
    >
      {title && <title>{title}</title>}

      {/* Gaps are offset per ring so they never line up */}
      <circle
        cx="16" cy="16" r={r1} stroke={STRUCT} strokeWidth={w} strokeLinecap="butt"
        strokeDasharray={dashes(r1, 72, 2)} transform="rotate(-70 16 16)"
      />
      <circle
        cx="16" cy="16" r={r2} stroke={STRUCT} strokeWidth={w} strokeLinecap="butt"
        strokeDasharray={dashes(r2, 61, 2)} transform="rotate(64 16 16)"
      />
      <circle
        cx="16" cy="16" r={r3} stroke={STRUCT} strokeWidth={w} strokeLinecap="butt"
        strokeDasharray={dashes(r3, 54, 2)} transform="rotate(-160 16 16)"
      />

      {/* Leading segment */}
      <circle
        cx="16" cy="16" r={r1} stroke={lead} strokeWidth={w} strokeLinecap="butt"
        strokeDasharray={`${c1 * (62 / 360)} ${c1}`} transform="rotate(-70 16 16)"
      />

      <circle cx="16" cy="16" r="2.6" fill={lead} />
    </svg>
  );
}

export default RuneMark;
