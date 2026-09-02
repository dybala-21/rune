/**
 * Line icons for product UI.
 *
 * Objects and status are drawn, not typed as emoji: emoji render differently
 * per platform, carry a colour the theme cannot control, and read as decoration
 * beside the geometric glyphs (✓ ◐ ○ ✗) the rest of the UI uses. These inherit
 * `currentColor`, so the caller sets the colour.
 */

import type { ReactNode } from 'react';

interface IconProps {
  size?: number;
  /** Nudge for very small or very large renderings. */
  strokeWidth?: number;
}

function Svg({ size = 14, strokeWidth = 1.6, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ flexShrink: 0 }}
    >
      {children}
    </svg>
  );
}

export function FolderIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4L11 8.5h8.5A1.5 1.5 0 0 1 21 10v7.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z" />
    </Svg>
  );
}

export function PlugIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M9 3v5M15 3v5" />
      <path d="M6 8h12v3a6 6 0 0 1-12 0z" />
      <path d="M12 17v4" />
    </Svg>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 4.5v3.5M12 16v3.5M4.5 12H8M16 12h3.5" />
      <circle cx="12" cy="12" r="3" />
    </Svg>
  );
}
