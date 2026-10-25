import type { ToolKind } from '../utils/toolActivity';

const paths: Record<ToolKind, string> = {
  file: 'M14 3H5v18h14V8z M14 3v5h5 M8 12h8 M8 16h6',
  terminal: 'M4 5h16v14H4z M7 9l3 3-3 3 M13 15h4',
  web: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M3 12h18 M12 3c-5 5-5 13 0 18 5-5 5-13 0-18',
  browser: 'M3 4h18v16H3z M3 9h18 M6 6.5h.01 M9 6.5h.01',
  desktop: 'M3 4h18v13H3z M8 21h8 M12 17v4',
  code: 'M8 6l-6 6 6 6 M16 6l6 6-6 6 M14 3l-4 18',
  memory: 'M6 3h12v18l-6-4-6 4z M9 7h6 M9 11h6',
  task: 'M9 6h12 M9 12h12 M9 18h12 M3 6l1 1 2-3 M3 12l1 1 2-3 M3 18l1 1 2-3',
  clock: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M12 7v5l3 2',
  connection: 'M9 3v5 M15 3v5 M6 8h12v3a6 6 0 0 1-12 0z M12 17v4',
  tool: 'M8 3H3v5 M16 3h5v5 M3 16v5h5 M21 16v5h-5 M9 12h6',
};

export function ToolActivityIcon({ kind }: { kind: ToolKind }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[kind]} /></svg>;
}
