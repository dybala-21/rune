/** Native imports only fill the visible composer; sending stays a user action. */
export function appendImportedText(previous: string, value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim() || value.includes('\0')) return null;
  const next = previous ? `${previous}\n\n${value}` : value;
  // API limit is 100,000 characters; UTF-16 length over-counts, so this is conservative.
  return next.length <= 100_000 ? next : null;
}

declare global {
  interface Window { runeImportText?: (text: unknown) => boolean }
}

export function installMobileDraft(
  host: Pick<Window, 'runeImportText'>,
  read: () => string,
  write: (text: string) => void,
): () => void {
  const receive = (value: unknown) => {
    const next = appendImportedText(read(), value);
    if (next === null) return false;
    write(next);
    return true;
  };
  host.runeImportText = receive;
  return () => { if (host.runeImportText === receive) delete host.runeImportText; };
}
