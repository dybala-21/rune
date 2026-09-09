import { useState } from 'react';
import { getTheme, toggleTheme, type Theme } from '../utils/theme';

// Self-contained so it can drop into any bar without prop plumbing.
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getTheme());
  return (
    <button
      className="toolbar-icon theme-toggle"
      onClick={() => setTheme(toggleTheme())}
      aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
      title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {theme === 'dark' ? <path d="M20 14A8.5 8.5 0 0 1 10 4a8.5 8.5 0 1 0 10 10Z" /> : <>
          <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5" />
        </>}
      </svg>
    </button>
  );
}
