import { useEffect, useState } from 'react';
import { fetchProactiveSuggestions } from '../api';
import type { ProactiveFeedItem } from '../types';
import { toast } from '../utils/toast';
import type { SseConnection } from './useSSE';

export function useProactiveSuggestions({ connected, clientId, addEventListener }: SseConnection) {
  const [items, setItems] = useState<ProactiveFeedItem[]>([]);

  useEffect(() => {
    if (!connected) return;
    const controller = new AbortController();
    let running = false;
    let again = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function refresh() {
      if (controller.signal.aborted) return;
      if (running) { again = true; return; }
      if (timer) clearTimeout(timer);
      running = true;
      let delay = 0;
      try {
        const latest = await fetchProactiveSuggestions(controller.signal);
        if (controller.signal.aborted) return;
        setItems(latest);
        failures = 0;
        if (latest.some(item => item.response === 'accepted' && !item.executionStatus)) delay = 5000;
      } catch {
        if (controller.signal.aborted) return;
        if (!failures) toast.error('Could not refresh suggestions. Retrying…');
        delay = Math.min(30_000, 2000 * 2 ** Math.min(failures++, 4));
      } finally {
        running = false;
        if (!controller.signal.aborted) {
          if (again) { again = false; void refresh(); }
          else if (delay) timer = setTimeout(() => void refresh(), delay);
        }
      }
    }

    const refreshVisible = () => { if (document.visibilityState === 'visible') void refresh(); };
    const unsubscribe = addEventListener('suggestion_created', () => void refresh());
    window.addEventListener('rune:proactive-changed', refreshVisible);
    window.addEventListener('focus', refreshVisible);
    document.addEventListener('visibilitychange', refreshVisible);
    void refresh();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
      unsubscribe();
      window.removeEventListener('rune:proactive-changed', refreshVisible);
      window.removeEventListener('focus', refreshVisible);
      document.removeEventListener('visibilitychange', refreshVisible);
    };
  }, [connected, clientId, addEventListener]);

  return items;
}
