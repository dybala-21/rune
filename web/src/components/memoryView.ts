import type { MemoryLine, MemoryOrigin } from '../api';

export type OriginTone = 'connector' | 'conversation' | 'learned' | 'unknown' | 'you';

export interface OriginChip {
  text: string;
  tone: OriginTone;
}

/** Where a remembered fact came from, as chips. A connector's name comes first: it is what
 * the owner can forget it by. */
export function originChips(origin: MemoryOrigin | null | undefined, zone: 'learned' | 'memory' | 'profile'): OriginChip[] {
  const chips: OriginChip[] = (origin?.connectors ?? []).map(name => ({ text: `from ${name}`, tone: 'connector' as const }));
  const conversations = origin?.conversations?.length ?? 0;
  if (conversations) {
    chips.push({ text: conversations === 1 ? 'from a conversation' : `from ${conversations} conversations`, tone: 'conversation' });
  }
  if (chips.length) return chips;
  if (origin?.kind === 'rule' || origin?.kind === 'autonomy') return [{ text: 'learned from how tasks ran', tone: 'learned' }];
  if (origin?.kind === 'agent') return [{ text: 'saved during a task', tone: 'conversation' }];
  if (zone !== 'learned') return [{ text: 'written by you', tone: 'you' }];
  return [{ text: 'origin not recorded', tone: 'unknown' }];
}

/** Lines grouped under their section, sections in the order they first appear. */
export function bySection(lines: MemoryLine[]): Array<{ section: string; lines: MemoryLine[] }> {
  const groups = new Map<string, MemoryLine[]>();
  for (const line of lines) groups.set(line.section, [...(groups.get(line.section) ?? []), line]);
  return [...groups].map(([section, grouped]) => ({ section, lines: grouped }));
}

export function hiddenReason(reason: string): string {
  if (reason.startsWith('connector_forgotten:')) return `forgotten with ${reason.slice('connector_forgotten:'.length)}`;
  if (reason === 'user_deleted') return 'you forgot it';
  if (reason === 'auto_pruned') return 'pruned as rarely useful';
  return reason || 'hidden';
}

export function forgottenSummary(done: { facts: number; episodes: number; dailyEntries: number }): string {
  const parts = [
    done.facts && `${done.facts} fact${done.facts === 1 ? '' : 's'}`,
    done.episodes && `${done.episodes} task record${done.episodes === 1 ? '' : 's'}`,
    done.dailyEntries && `${done.dailyEntries} daily entr${done.dailyEntries === 1 ? 'y' : 'ies'}`,
  ].filter(Boolean);
  return parts.length ? `Forgot ${parts.join(', ')}.` : 'Nothing was remembered from it.';
}
