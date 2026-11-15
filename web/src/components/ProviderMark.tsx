import { useState } from 'react';

// Vendor logos from LiteLLM are served from /logos; other providers use a monogram.

const HAS_LOGO = new Set([
  'openai', 'anthropic', 'gemini', 'xai', 'azure', 'cohere', 'mistral', 'deepseek',
  'ollama',
]);

// Keeps providers without a logo visually distinct from each other.
const FALLBACK_COLORS: Record<string, string> = {
  groq: '#F55036',
  together: '#6E56CF',
  openrouter: '#7C5CFF',
  lmstudio: '#7FCF9A',
  perplexity: '#20A5A5',
};

function monogram(provider: string): string {
  const cleaned = provider.replace(/[^a-z0-9]/gi, '');
  return (cleaned.slice(0, 2) || '?').toUpperCase();
}

export function ProviderMark({ provider, size = 16 }: { provider: string; size?: number }) {
  const key = provider.toLowerCase();
  // Fall back to a monogram if the logo fails to load.
  const [failed, setFailed] = useState(false);

  if (HAS_LOGO.has(key) && !failed) {
    return (
      <img
        src={`/logos/${key}.svg`}
        alt=""
        aria-hidden="true"
        draggable={false}
        width={size}
        height={size}
        onError={() => setFailed(true)}
        style={{ flexShrink: 0, display: 'block', borderRadius: size * 0.2 }}
      />
    );
  }

  const color = FALLBACK_COLORS[key] ?? 'var(--text-muted)';
  return (
    <span
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        flexShrink: 0,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: size * 0.3,
        background: `color-mix(in srgb, ${color} 20%, transparent)`,
        color,
        fontSize: size * 0.5,
        fontWeight: 700,
        fontFamily: 'var(--font-mono)',
        lineHeight: 1,
      }}
    >
      {monogram(provider)}
    </span>
  );
}

export default ProviderMark;
