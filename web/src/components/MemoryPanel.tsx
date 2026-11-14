import { useCallback, useEffect, useState } from 'react';
import {
  fetchMemory,
  forgetConnectorMemory,
  forgetMemory,
  promoteMemory,
  saveMemory,
  unsuppressMemory,
  type MemoryLine,
  type MemorySnapshot,
  type MemoryZone,
} from '../api';
import { useFocusTrap } from '../hooks/useFocusTrap';
import { bySection, forgottenSummary, hiddenReason, originChips, type OriginChip } from './memoryView';

type Tab = 'learned' | 'memory' | 'profile' | 'hidden' | 'sources';

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'learned', label: 'Learned' },
  { id: 'memory', label: 'Your notes' },
  { id: 'profile', label: 'Profile' },
  { id: 'hidden', label: 'Hidden' },
  { id: 'sources', label: 'Sources' },
];

const CHIP_COLOR: Record<OriginChip['tone'], string> = {
  connector: 'var(--warning)',
  conversation: 'var(--accent)',
  learned: 'var(--text-muted)',
  unknown: 'var(--text-muted)',
  you: 'var(--success)',
};

const button = (primary = false): React.CSSProperties => ({
  padding: '4px 10px', fontSize: 12, borderRadius: 'var(--radius-md)', cursor: 'pointer',
  border: primary ? 'none' : '1px solid var(--border)',
  background: primary ? 'var(--accent)' : 'transparent',
  color: primary ? 'white' : 'var(--text-secondary, var(--text-primary))',
});

const input: React.CSSProperties = {
  flex: 1, minWidth: 0, padding: '5px 8px', fontSize: 13,
  background: 'var(--bg-primary)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
};

function Chips({ chips }: { chips: OriginChip[] }) {
  return (
    <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
      {chips.map(chip => (
        <span key={chip.text} style={{
          fontSize: 11, padding: '1px 7px', borderRadius: 999,
          border: `1px solid ${CHIP_COLOR[chip.tone]}`, color: CHIP_COLOR[chip.tone],
        }}>{chip.text}</span>
      ))}
    </span>
  );
}

interface RowProps {
  text: string;
  detail?: string;
  chips: OriginChip[];
  onSave: (value: string) => Promise<void>;
  onForget: () => Promise<void>;
  extra?: { label: string; run: () => Promise<void> };
}

function Row({ text, detail, chips, onSave, onForget, extra }: RowProps) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      {draft === null ? (
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, overflowWrap: 'anywhere' }}>{text}</div>
            <div style={{ marginTop: 4, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              {detail && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{detail}</span>}
              <Chips chips={chips} />
            </div>
          </div>
          <button style={button()} onClick={() => setDraft(text)}>Edit</button>
          {extra && <button style={button()} onClick={() => void extra.run().catch(() => {})}>{extra.label}</button>}
          <button style={button()} onClick={() => void onForget().catch(() => {})}>Forget</button>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 8 }}>
          <input style={input} value={draft} autoFocus aria-label="Edit memory"
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Escape') { e.stopPropagation(); setDraft(null); } }} />
          <button style={button(true)} disabled={!draft.trim()}
            onClick={() => void onSave(draft).then(() => setDraft(null), () => {})}>Save</button>
          <button style={button()} onClick={() => setDraft(null)}>Cancel</button>
        </div>
      )}
    </div>
  );
}

function AddLine({ onAdd }: { onAdd: (value: string) => Promise<void> }) {
  const [value, setValue] = useState('');
  return (
    <div style={{ display: 'flex', gap: 8, padding: '8px 0' }}>
      <input style={input} value={value} placeholder="Add a line" aria-label="Add a line"
        onChange={e => setValue(e.target.value)} />
      <button style={button()} disabled={!value.trim()}
        onClick={() => void onAdd(value).then(() => setValue(''), () => {})}>Add</button>
    </div>
  );
}

interface MemoryPanelProps {
  onClose: () => void;
  onOpenFiles?: () => void;
}

export function MemoryPanel({ onClose, onOpenFiles }: MemoryPanelProps) {
  const [tab, setTab] = useState<Tab>('learned');
  const [snapshot, setSnapshot] = useState<MemorySnapshot | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [confirming, setConfirming] = useState<string | null>(null);
  const trapRef = useFocusTrap<HTMLDivElement>();

  const run = useCallback(async (action: () => Promise<MemorySnapshot>) => {
    setError('');
    try {
      const next = await action();
      setSnapshot(next);
      if (next.forgotten) setNotice(forgottenSummary(next.forgotten));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Couldn't change memory");
      throw exc;
    }
  }, []);

  useEffect(() => { void run(fetchMemory).catch(() => {}); }, [run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Errors are shown in the panel and rethrown, so a rejected edit keeps its draft open.
  const act = run;

  const lines = (zone: Exclude<MemoryZone, 'learned'>, rows: MemoryLine[]) => {
    const groups = bySection(rows);
    if (!groups.length) return <Empty text={zone === 'memory' ? 'No notes yet.' : 'No profile entries yet.'} />;
    return groups.map(group => (
      <section key={group.section} style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.4, color: 'var(--text-muted)', margin: '8px 0 0' }}>
          {group.section}
        </h3>
        {group.lines.map(row => (
          <Row key={row.line} text={row.line} chips={originChips(row.origin, zone)}
            onSave={value => act(() => saveMemory({ zone, section: row.section, old: row.line, value }))}
            onForget={() => act(() => forgetMemory({ zone, section: row.section, line: row.line }))} />
        ))}
        <AddLine onAdd={value => act(() => saveMemory({ zone, section: group.section, value }))} />
      </section>
    ));
  };

  const body = () => {
    if (!snapshot) return <Empty text={error ? '' : 'Loading…'} />;
    if (tab === 'learned') {
      if (!snapshot.learned.length) return <Empty text="Nothing learned yet." />;
      return snapshot.learned.map(fact => (
        <Row key={`${fact.category}:${fact.key}`} text={fact.value}
          detail={`${fact.category} · ${fact.key}${fact.hits ? ` · used ${fact.hits}×` : ''}`}
          chips={originChips(fact.origin, 'learned')}
          onSave={value => act(() => saveMemory({ zone: 'learned', key: fact.key, value }))}
          onForget={() => act(() => forgetMemory({ zone: 'learned', key: fact.key }))}
          extra={{ label: 'Keep', run: () => act(() => promoteMemory(fact.key)) }} />
      ));
    }
    if (tab === 'memory') return lines('memory', snapshot.memory);
    if (tab === 'profile') return lines('profile', snapshot.profile);
    if (tab === 'hidden') {
      if (!snapshot.suppressed.length) return <Empty text="Nothing is hidden." />;
      return (
        <>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>RUNE won't learn these again unless you allow it.</p>
          {snapshot.suppressed.map(item => (
            <div key={item.key} style={{ display: 'flex', gap: 8, padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, overflowWrap: 'anywhere' }}>{item.value || item.key}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{hiddenReason(item.reason)}</div>
              </div>
              <button style={button()} onClick={() => void act(() => unsuppressMemory(item.key)).catch(() => {})}>Allow again</button>
            </div>
          ))}
        </>
      );
    }
    return (
      <>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          What RUNE remembers from each connector. Forgetting one removes the facts and task records drawn from
          it, and keeps them from being learned again. Conversations themselves stay; delete them from the sidebar.
        </p>
        {!snapshot.connectors.length && <Empty text="Nothing is remembered from a connector." />}
        {snapshot.connectors.map(c => (
          <div key={c.name} style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{c.name}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{c.facts} facts · {c.episodes} task records</div>
            </div>
            {confirming === c.name ? (
              <>
                <button style={{ ...button(true), background: 'var(--danger)' }}
                  onClick={() => { setConfirming(null); void act(() => forgetConnectorMemory(c.name)).catch(() => {}); }}>
                  Forget all from {c.name}
                </button>
                <button style={button()} onClick={() => setConfirming(null)}>Cancel</button>
              </>
            ) : (
              <button style={button()} onClick={() => setConfirming(c.name)}>Forget…</button>
            )}
          </div>
        ))}
      </>
    );
  };

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.5)' }} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={trapRef} role="dialog" aria-modal="true" aria-label="Memory" className="glass" style={{
        width: 'min(900px, calc(100vw - 32px))', height: '80vh', borderRadius: 'var(--radius-xl)',
        border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--border)', gap: 12 }}>
          <span style={{ fontSize: 16, fontWeight: 600, flex: 1 }}>Memory</span>
          {onOpenFiles && <button style={button()} onClick={onOpenFiles}>Edit files</button>}
          <button onClick={onClose} aria-label="Close" style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: 18, cursor: 'pointer' }}>
            {'✕'}
          </button>
        </div>
        <div role="tablist" style={{ display: 'flex', gap: 4, padding: '8px 16px', borderBottom: '1px solid var(--border)', overflowX: 'auto' }}>
          {TABS.map(t => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => { setTab(t.id); setNotice(''); }}
              style={{ ...button(tab === t.id), whiteSpace: 'nowrap' }}>
              {t.label}{t.id === 'hidden' && snapshot?.suppressed.length ? ` (${snapshot.suppressed.length})` : ''}
            </button>
          ))}
        </div>
        {(error || notice) && (
          <div role="status" style={{ padding: '8px 20px', fontSize: 12, color: error ? 'var(--danger)' : 'var(--success)' }}>
            {error || notice}
          </div>
        )}
        <div style={{ flex: 1, overflow: 'auto', padding: '4px 20px 16px' }}>{body()}</div>
      </div>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <div style={{ padding: '32px 0', textAlign: 'center', fontSize: 13, color: 'var(--text-muted)' }}>{text}</div>;
}
