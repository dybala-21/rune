import type { ChatMessage, ProactiveFeedItem, ProactiveStatus } from '../types';

export function withoutProactive(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(message => !message.suggestion);
}

export function withProactive(messages: ChatMessage[], items: ProactiveFeedItem[]): ChatMessage[] {
  const cards = [...new Map(items.map(item => [item.id, item])).values()].map((item): ChatMessage => {
    const timestamp = Date.parse(item.createdAt) || 0;
    return {
      id: `proactive:${item.id}`, role: 'system', content: item.title || item.description, timestamp,
      suggestion: {
        id: item.id, headline: item.title, body: item.description, timestamp,
        actions: [], confidence: item.confidence,
        intensity: item.confidence >= .8 ? 'intervene' : item.confidence >= .6 ? 'suggest' : 'nudge',
        state: { response: item.response, executionStatus: item.executionStatus, result: item.result },
      },
    };
  });
  cards.sort((a, b) => a.timestamp - b.timestamp);
  const merged: ChatMessage[] = [];
  let index = 0;
  for (const message of withoutProactive(messages)) {
    while (index < cards.length && cards[index].timestamp <= message.timestamp) merged.push(cards[index++]);
    merged.push(message);
  }
  return [...merged, ...cards.slice(index)];
}

export function proactiveStatus(state?: ProactiveStatus): string | undefined {
  if (!state) return 'Loading status…';
  if (state.response === 'dismissed') return 'Dismissed';
  if (state.response === 'expired') return 'Expired';
  if (state.executionStatus === 'success') return 'Verified';
  if (state.executionStatus) {
    const text = state.executionStatus.split('_').join(' ');
    return text[0].toUpperCase() + text.slice(1);
  }
  if (state.response === 'accepted') return 'Queued for the background worker';
  if (state.response !== 'pending' && state.response !== 'unconfirmed') return 'Status unavailable';
}
