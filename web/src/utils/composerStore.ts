import { storageKey } from './sessionContext';

const KEY = 'rune.composer.v1';
const MAX_DRAFTS = 8;
const MAX_TEXT = 100_000;
const MAX_AGE = 24 * 60 * 60 * 1000;
interface Entry { conversation: string; text: string; hadAttachments: boolean; savedAt: number }

function entries(key: string): Entry[] {
  try {
    const parsed: unknown = JSON.parse(sessionStorage.getItem(key) ?? '[]');
    if (!Array.isArray(parsed)) return [];
    const now = Date.now();
    return parsed.filter((value): value is Entry => Boolean(value)
      && typeof value.conversation === 'string' && value.conversation.length <= 512
      && typeof value.text === 'string' && value.text.length <= MAX_TEXT
      && typeof value.hadAttachments === 'boolean'
      && typeof value.savedAt === 'number' && Number.isFinite(value.savedAt)
      && value.savedAt <= now && now - value.savedAt < MAX_AGE).slice(-MAX_DRAFTS);
  } catch { return []; }
}

export function readComposerDraft(conversation: string): { text: string; hadAttachments: boolean } | null {
  const key = storageKey(KEY);
  if (!key) return null;
  const entry = entries(key).find(value => value.conversation === conversation);
  return entry ? { text: entry.text, hadAttachments: entry.hadAttachments } : null;
}

/** Keep text for a same-tab sign-in; files must be attached again after a reload. */
export function saveComposerDraft(conversation: string, text: string, hadAttachments: boolean): void {
  const key = storageKey(KEY);
  if (!key || !conversation || conversation.length > 512 || text.length > MAX_TEXT) return;
  try {
    const saved = entries(key).filter(value => value.conversation !== conversation);
    if (text || hadAttachments) saved.push({ conversation, text, hadAttachments, savedAt: Date.now() });
    if (saved.length) sessionStorage.setItem(key, JSON.stringify(saved.slice(-MAX_DRAFTS)));
    else sessionStorage.removeItem(key);
  } catch { /* Storage errors must not block typing. */ }
}
