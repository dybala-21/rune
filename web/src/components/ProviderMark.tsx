import { useState } from 'react';

/**
 * The provider's logo, for the model picker.
 *
 * Logos are served from /logos rather than inlined: the CSP already allows
 * same-origin images, so this keeps them out of the JS bundle and lets the
 * browser cache them. The set covers every provider `models.list` returns; a
 * provider added later — or a self-hosted endpoint — falls back to a monogram
 * instead of a missing-image icon.
 *
 * Files come from litellm's bundled vendor assets, so each mark is the one its
 * vendor publishes rather than an approximation.
 */

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
  // A logo file can go missing on a partial deploy; drop to the monogram
  // rather than leaving a broken-image glyph in the menu.
  const [failed, setFailed] = useState(false);

  if (HAS_LOGO.has(key) && !failed) {
    return (
      <img
        src={`/logos/${key}.svg`}
        alt=""
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
