import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';

export interface TooltipProps {
  /** Plain text or nodes — never raw HTML. */
  content: ReactNode;
  children: ReactNode;
  /** Underline the trigger to advertise that an explanation exists. */
  underline?: boolean;
  className?: string;
}

interface Position {
  left: number;
  top: number;
}

/**
 * Hover/focus explanation. The trigger is focusable and describes itself with
 * `aria-describedby`, so the text is reachable by keyboard and by AT.
 */
export function Tooltip({ content, children, underline = true, className }: TooltipProps) {
  const id = useId();
  const triggerRef = useRef<HTMLSpanElement>(null);
  const bubbleRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<Position>({ left: 0, top: 0 });

  useLayoutEffect(() => {
    if (!open) return;
    const trigger = triggerRef.current;
    const bubble = bubbleRef.current;
    if (!trigger || !bubble) return;

    const anchor = trigger.getBoundingClientRect();
    const box = bubble.getBoundingClientRect();
    const margin = 8;

    let left = anchor.left + anchor.width / 2 - box.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - box.width - margin));

    let top = anchor.top - box.height - 6;
    if (top < margin) top = anchor.bottom + 6;

    setPosition({ left, top });
  }, [open, content]);

  const close = useCallback(() => setOpen(false), []);

  return (
    <span
      className={['tt-wrap', className].filter(Boolean).join(' ')}
      ref={triggerRef}
      tabIndex={0}
      aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={close}
      onFocus={() => setOpen(true)}
      onBlur={close}
      onKeyDown={(event) => {
        if (event.key === 'Escape') close();
      }}
    >
      <span className={underline ? 'tt-underline' : undefined}>{children}</span>
      {open && (
        <span
          className="tt-bubble"
          role="tooltip"
          id={id}
          ref={bubbleRef}
          style={{ left: position.left, top: position.top }}
        >
          {content}
        </span>
      )}
    </span>
  );
}
