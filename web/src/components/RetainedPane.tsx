import { useEffect, useState, type ReactNode } from 'react';

/** Keep visited panes alive while hiding their controls from focus and accessibility. */
export function RetainedPane({ active, children }: { active: boolean; children: ReactNode }) {
  const [visited, setVisited] = useState(active);
  useEffect(() => { if (active) setVisited(true); }, [active]);
  return <div className="retained-pane" hidden={!active}>{(active || visited) && children}</div>;
}
