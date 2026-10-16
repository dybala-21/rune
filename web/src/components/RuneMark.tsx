export type MarkState = 'idle' | 'thinking' | 'working' | 'passed' | 'failed' | 'warning';

interface RuneMarkProps {
  state?: MarkState;
  size?: number;
  title?: string;
}

export function RuneMark({ state = 'idle', size = 24, title = 'Rune' }: RuneMarkProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="currentColor"
      className="rune-mark" data-state={state} role="img" aria-label={title}>
      <path d="M4.5 9.2 20.8 1.6a4 4 0 0 1 3.2-.1l4.1 1.7a2 2 0 0 1 1.2 1.8v4.3a2 2 0 0 1-1.2 1.8L17.2 16 4.5 9.2Z" />
      <path d="M3.7 10.3c-.8 1.2-1 2.5-1 4V29a1.1 1.1 0 0 0 1.6 1l8.5-3.8a2 2 0 0 0 1.2-1.8v-6.7c0-1-.4-1.8-1.1-2.5l-9.2-4.9Z" />
    </svg>
  );
}

export default RuneMark;
