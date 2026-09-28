import type { ReactNode } from 'react';

const paths = {
  more: <><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></>,
  computer: <><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M8 21h8m-4-4v4" /></>,
  browser: <><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M3 8h18M7 5.5h.01M10 5.5h.01" /></>,
  globe: <><circle cx="12" cy="12" r="9" /><ellipse cx="12" cy="12" rx="4" ry="9" /><path d="M3 12h18" /></>,
  back: <path d="m14 6-6 6 6 6M8 12h12" />,
  forward: <path d="m10 6 6 6-6 6M4 12h12" />,
  reload: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5.5 8a7 7 0 0 1 11.6-3L20 8M4 16l2.9 3A7 7 0 0 0 18.5 16" /></>,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  pointer: <path d="m5 3 14 10-7 1-3 7-4-18Z" />,
  pause: <><path d="M8 5v14M16 5v14" strokeWidth="3" /></>,
  play: <path d="m8 4 12 8-12 8V4Z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  expand: <path d="M9 3H3v6m12 12h6v-6M3 3l6 6m12 12-6-6" />,
  shrink: <path d="M3 9h6V3m12 12h-6v6M9 9 3 3m12 12 6 6" />,
  panel: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></>,
} satisfies Record<string, ReactNode>;

export function ComputerIcon({ name, size = 16 }: { name: keyof typeof paths; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
