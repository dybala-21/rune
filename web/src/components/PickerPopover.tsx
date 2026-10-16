import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface PickerPopoverProps {
  label: string;
  trigger: ReactNode;
  children: (close: () => void) => ReactNode;
  onOpen?: () => void;
  menuKey?: string;
  className?: string;
  busy?: boolean;
}

export function pickerPlacement(anchor: Pick<DOMRect, 'top' | 'bottom' | 'left'>, viewport: { width: number; height: number }) {
  const width = Math.min(320, Math.max(0, viewport.width - 24));
  const below = viewport.height - anchor.bottom - 20;
  const above = anchor.top - 20;
  const opensUp = below < Math.min(320, above);
  return {
    top: opensUp ? anchor.top - 8 : anchor.bottom + 8,
    left: Math.max(12, Math.min(anchor.left, viewport.width - width - 12)),
    width,
    maxHeight: Math.max(0, Math.min(480, opensUp ? above : below)),
    transform: opensUp ? 'translateY(-100%)' : undefined,
  };
}

export function PickerPopover({ label, trigger, children, onOpen, menuKey, className = '', busy }: PickerPopoverProps) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<ReturnType<typeof pickerPlacement>>({ top: 0, left: 0, width: 320, maxHeight: 480, transform: undefined });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const enterFromEnd = useRef(false);
  const id = useId();

  const close = () => {
    setOpen(false);
    buttonRef.current?.focus();
  };
  const show = () => {
    onOpen?.();
    setOpen(true);
  };

  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const anchor = buttonRef.current?.getBoundingClientRect();
      if (!anchor) return;
      setPosition(pickerPlacement(anchor, { width: window.innerWidth, height: window.innerHeight }));
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const panel = panelRef.current;
    const items = panel?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)');
    const selected = panel?.querySelector<HTMLButtonElement>('[aria-checked="true"]:not(:disabled)');
    const target = enterFromEnd.current ? items?.[items.length - 1] : selected ?? items?.[0];
    (target ?? panel)?.focus();
    enterFromEnd.current = false;
  }, [open, menuKey]);

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!buttonRef.current?.contains(target) && !panelRef.current?.contains(target)) setOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [open]);

  const navigate = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape' || event.key === 'Tab') {
      if (event.key === 'Escape') event.preventDefault();
      event.stopPropagation();
      close();
      return;
    }
    const items = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)'));
    if (!items.length) return;
    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    let next: number;
    if (event.key === 'ArrowDown') next = (current + 1) % items.length;
    else if (event.key === 'ArrowUp') next = (current - 1 + items.length) % items.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = items.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    items[next]?.focus();
  };

  return <>
    <button
      ref={buttonRef} type="button" className={`picker-trigger ${className}`}
      aria-label={label} aria-haspopup="menu" aria-expanded={open} aria-controls={open ? id : undefined}
      aria-busy={busy || undefined}
      onClick={() => open ? close() : show()}
      onKeyDown={event => {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        event.preventDefault();
        enterFromEnd.current = event.key === 'ArrowUp';
        show();
      }}
    >
      {trigger}
      {busy ? <span className="spinner" /> : <ChevronDown />}
    </button>
    {open && createPortal(
      <div ref={panelRef} id={id} role="menu" aria-label={label} tabIndex={-1}
        className="picker-popover" style={position} onKeyDown={navigate}>
        {children(close)}
      </div>, document.body,
    )}
  </>;
}

export function ChevronDown() {
  return <svg className="picker-chevron" width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

export function SelectedCheck() {
  return <svg className="picker-check" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <path d="m3.5 8 3 3 6-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}
