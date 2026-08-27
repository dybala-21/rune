// Theme is dark by default (the base :root palette). Light is an override
// applied by stamping data-theme="light" on <html>; the choice persists in
// localStorage. Kept as a plain module so it can run before React mounts,
// which avoids a first-paint flash of the wrong palette.

export type Theme = 'dark' | 'light';
const KEY = 'rune:web:theme';

export function getTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === 'light' || v === 'dark') return v;
  } catch { /* storage unavailable */ }
  return 'dark';
}

export function applyTheme(t: Theme): void {
  const el = document.documentElement;
  if (t === 'light') el.setAttribute('data-theme', 'light');
  else el.removeAttribute('data-theme');
  try { localStorage.setItem(KEY, t); } catch { /* ignore */ }
}

export function applyStoredTheme(): void {
  applyTheme(getTheme());
}

export function toggleTheme(): Theme {
  const next: Theme = getTheme() === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  return next;
}
