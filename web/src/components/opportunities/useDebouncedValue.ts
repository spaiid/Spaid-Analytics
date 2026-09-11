import { useEffect, useState } from 'react';

/**
 * The value, settled. The input stays fully controlled by the caller's state so
 * typing is instant; only the *applied* filter lags by `delay`, which keeps the
 * table and both charts from re-deriving on every keystroke.
 */
export function useDebouncedValue<T>(value: T, delay = 250): T {
  const [settled, setSettled] = useState(value);

  useEffect(() => {
    if (Object.is(value, settled)) return;
    const id = window.setTimeout(() => setSettled(value), delay);
    return () => window.clearTimeout(id);
  }, [value, settled, delay]);

  return settled;
}
