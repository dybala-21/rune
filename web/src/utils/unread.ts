// What RUNE wrote on its own since the owner last opened it. Only "From RUNE" is written to with no
// one watching: a scheduled message no chat app took is kept there (rune/daemon/owner_thread.py).
// The chat list marks it, and so do the computer's notification (hooks/useInboxNotice.ts) and the
// desktop app's dock icon. How many messages were there when it was last opened is kept in this browser.

import type { SessionInfo } from '../api';
import { PAGE_SIZE } from './sessionList';

export const INBOX_ID = 'rune:inbox';
const KEY = 'rune:web:seen-messages';

export type Seen = Record<string, number>;

export function loadSeen(storage: Pick<Storage, 'getItem'> | undefined = globalThis.localStorage): Seen {
  try {
    const value: unknown = JSON.parse(storage?.getItem(KEY) || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Seen : {};
  } catch { /* storage unavailable or unreadable: every message in "From RUNE" counts as new */ }
  return {};
}

export function markSeen(seen: Seen, id: string, messages: number,
  storage: Pick<Storage, 'setItem'> | undefined = globalThis.localStorage): Seen {
  if (seen[id] === messages) return seen;
  const next = { ...seen, [id]: messages };
  try { storage?.setItem(KEY, JSON.stringify(next)); } catch { /* storage unavailable: kept for this page */ }
  return next;
}

/** How many messages "From RUNE" holds as the chat list reads it: 0 when the list's first page
 * does not have it (a new message puts it on top), unknown while the list loads, is searched, or
 * holds older pages, which are not read again (SessionListLoader.refresh). */
export function inboxMessages(sessions: SessionInfo[], loading: boolean, searching: boolean): number | undefined {
  if (loading || searching || sessions.length > PAGE_SIZE) return undefined;
  return sessions.find(session => session.id === INBOX_ID)?.turnCount ?? 0;
}

export function unread(seen: Seen, id: string, messages: number): number {
  return id === INBOX_ID ? Math.max(0, messages - (seen[id] ?? 0)) : 0;
}
