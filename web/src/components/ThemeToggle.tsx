import { useState } from 'react';
import { getTheme, toggleTheme, type Theme } from '../utils/theme';

// Self-contained so it can drop into any bar without prop plumbing.
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getTheme());
  return (
    <button
      className="theme-toggle"
      onClick={() => setTheme(toggleTheme())}
      aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
      title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
    >
      {theme === 'dark' ? '☾' : '☀'}
    </button>
  );
}
