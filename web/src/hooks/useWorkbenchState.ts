import { useCallback, useMemo, useState, type SetStateAction } from 'react';
import type { BenchTab } from '../components/WorkbenchPanel';

interface PanelState {
  open: boolean;
  dismissed: boolean;
  tab: BenchTab;
  expanded: boolean;
  computerRequest: number;
}

export function useWorkbenchState(sessionId: string | null) {
  const initial = useMemo((): PanelState & { sessionId: string | null } => ({
    sessionId, open: false, dismissed: false, tab: 'progress', expanded: false, computerRequest: 0,
  }), [sessionId]);
  const [stored, setStored] = useState(initial);
  const current = stored.sessionId === sessionId ? stored : initial;
  const update = useCallback(<K extends keyof PanelState>(key: K, value: SetStateAction<PanelState[K]>) => {
    setStored(previous => {
      const state = previous.sessionId === sessionId ? previous : initial;
      const next = typeof value === 'function' ? (value as (p: PanelState[K]) => PanelState[K])(state[key]) : value;
      return state[key] === next ? state : { ...state, [key]: next };
    });
  }, [sessionId, initial]);
  const setOpen = useCallback((v: SetStateAction<boolean>) => update('open', v), [update]);
  const setDismissed = useCallback((v: SetStateAction<boolean>) => update('dismissed', v), [update]);
  const setTab = useCallback((v: SetStateAction<BenchTab>) => update('tab', v), [update]);
  const setExpanded = useCallback((v: SetStateAction<boolean>) => update('expanded', v), [update]);
  const setComputerRequest = useCallback((v: SetStateAction<number>) => update('computerRequest', v), [update]);
  return { ...current, setOpen, setDismissed, setTab, setExpanded, setComputerRequest };
}

export type WorkbenchState = ReturnType<typeof useWorkbenchState>;
