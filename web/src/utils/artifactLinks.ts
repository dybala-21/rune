/** Local artifacts are resolved by the server against the displayed conversation. */
export function artifactHref(path: string, sessionId?: string): string | null {
  const value = path.trim().replace(/^<(.+)>$/, '$1');
  if (/^(https?:|mailto:)/i.test(value) || value.startsWith('#')) return value;
  if (!value || /^[a-z][a-z0-9+.-]*:/i.test(value) || value.startsWith('//')) return null;
  if (!sessionId) return null;
  const query = new URLSearchParams({ sessionId, path: value });
  return `/api/v1/files/download?${query}`;
}
