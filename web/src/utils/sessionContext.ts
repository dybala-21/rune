let scope: string | null | undefined;

/** Return true if the account changed and storage must wait for a reload. */
export function bindStorageScope(value: unknown): boolean {
  const next = typeof value === 'string' && (value === 'local' || /^[a-f0-9]{64}$/.test(value)) ? value : null;
  // Old callbacks must keep the original account's storage scope.
  if (scope !== undefined && scope !== next) return true;
  scope = next;
  return false;
}

export function storageKey(base: string): string | null {
  if (!scope) return null;
  return scope === 'local' ? base : `${base}:${scope}`;
}

export function accountScope(): string | null {
  return scope && scope !== 'local' ? scope : null;
}

export function isLocalSession(): boolean {
  return scope === 'local';
}
