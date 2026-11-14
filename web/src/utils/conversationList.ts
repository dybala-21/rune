import type { SessionInfo } from '../api';

export type ConversationListItem = Pick<SessionInfo, 'id' | 'title' | 'turnCount' | 'updatedAt'>;

/** Retain chats created locally until the history endpoint catches up. */
export function rememberLiveConversation(
  previous: ConversationListItem[], live: ConversationListItem, retain = true,
): ConversationListItem[] {
  // An unconfirmed request must not become a history entry when leaving chat.
  if (!retain) return previous.some(item => item.id === live.id)
    ? previous.filter(item => item.id !== live.id) : previous;
  // An untouched new chat has no server history to navigate back to.
  if (!live.turnCount) return previous;
  const existing = previous.find(item => item.id === live.id);
  if (existing?.title === live.title && existing.turnCount === live.turnCount && existing.updatedAt === live.updatedAt) return previous;
  return [live, ...previous.filter(item => item.id !== live.id)].slice(0, 50);
}

/** One row per conversation, with immediate local updates and title-only search. */
export function mergeConversationList(
  saved: ConversationListItem[], recent: ConversationListItem[],
  live: ConversationListItem, query: string,
): ConversationListItem[] {
  const merged = new Map(saved.map(item => [item.id, item]));
  for (const item of recent) {
    const existing = merged.get(item.id);
    if (!existing || Date.parse(item.updatedAt) > Date.parse(existing.updatedAt)) merged.set(item.id, item);
  }
  const persisted = merged.get(live.id);
  merged.set(live.id, { ...live, turnCount: Math.max(live.turnCount, persisted?.turnCount ?? 0) });
  const normalized = query.trim().toLocaleLowerCase();
  return [...merged.values()]
    .filter(item => !normalized || item.title.toLocaleLowerCase().includes(normalized))
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}
