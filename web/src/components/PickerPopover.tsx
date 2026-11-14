import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface PickerPopoverProps {
  label: string;
  trigger: ReactNode;
  children: (close: () => void) => ReactNode;
  onOpen?: () => void;
  openRequest?: number;
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

export function PickerPopover({ label, trigger, children, onOpen, openRequest, menuKey, className = '', busy }: PickerPopoverProps) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<ReturnType<typeof pickerPlacement>>({ top: 0, left: 0, width: 320, maxHeight: 480, transform: undefined });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const enterFromEnd = useRef(false);
  const id = useId();
  const handledRequest = useRef(0);

  const close = () => {
    setOpen(false);
    buttonRef.current?.focus();
  };
  const show = () => {
    onOpen?.();
    setOpen(true);
  };

  useEffect(() => {
    if (!openRequest || openRequest === handledRequest.current) return;
    handledRequest.current = openRequest;
    show();
  }, [openRequest]);

  useLayoutEffect(() => {
    if (open && buttonRef.current?.closest('[inert]')) setOpen(false);
  });

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
    const search = panel?.querySelector<HTMLInputElement>('[data-picker-search]');
    if (search && document.activeElement === search) return;
    const selected = panel?.querySelector<HTMLButtonElement>('[aria-checked="true"]:not(:disabled)');
    const target = enterFromEnd.current ? items?.[items.length - 1] : search ?? selected ?? items?.[0];
    (target ?? panel)?.focus();
    enterFromEnd.current = false;
  }, [open, menuKey]);

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent | FocusEvent) => {
      const target = event.target as Node;
      if (!buttonRef.current?.contains(target) && !panelRef.current?.contains(target)) setOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('focusin', dismiss);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('focusin', dismiss);
    };
  }, [open]);

  const navigate = (event: KeyboardEvent<HTMLDivElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      close();
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.stopPropagation();
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      close();
      return;
    }
    const panel = event.currentTarget;
    const search = panel.querySelector<HTMLInputElement>('[data-picker-search]:not(:disabled)');
    if (event.key === 'Tab') {
      event.stopPropagation();
      if (search) {
        const controls = Array.from(panel.querySelectorAll<HTMLElement>(
          '[data-picker-search]:not(:disabled), [data-picker-control]:not(:disabled), [role^="menuitem"]:not(:disabled)',
        ));
        const current = controls.indexOf(document.activeElement as HTMLElement);
        const next = current + (event.shiftKey ? -1 : 1);
        if (current >= 0 && next >= 0 && next < controls.length) {
          event.preventDefault();
          controls[next].focus();
          return;
        }
      }
      close();
      return;
    }
    const typing = ['INPUT', 'TEXTAREA'].includes((event.target as HTMLElement)?.tagName);
    if (typing && ['Home', 'End', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
      event.stopPropagation();
      return;
    }
    const items = Array.from(panel.querySelectorAll<HTMLElement>('[role^="menuitem"]:not(:disabled)'));
    if (!items.length) return;
    const arrows = search ? [search, ...items] : items;
    const current = arrows.indexOf(document.activeElement as HTMLElement);
    let next: number;
    if (event.key === 'ArrowDown') next = (current + 1) % arrows.length;
    else if (event.key === 'ArrowUp') next = current < 0 ? arrows.length - 1 : (current - 1 + arrows.length) % arrows.length;
    else if (event.key === 'Home') next = search ? 1 : 0;
    else if (event.key === 'End') next = arrows.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    arrows[next]?.focus();
  };

  return <>
    <button
      ref={buttonRef} type="button" className={`picker-trigger ${className}`}
      aria-label={label} aria-haspopup="menu" aria-expanded={open} aria-controls={open ? id : undefined}
      aria-busy={busy || undefined}
      onClick={() => open ? close() : show()}
      onKeyDown={event => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
          if (open) close();
          return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
          event.stopPropagation();
          return;
        }
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        event.preventDefault();
        event.stopPropagation();
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
