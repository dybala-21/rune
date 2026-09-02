import type { TokenUsage as TokenUsageType } from '../types';

// Rough mid-tier cloud rates, used only to give a sense of scale. They are not
// billing figures and are meaningless for a locally hosted model, so the cost
// row is hidden for those providers instead of showing a made-up number.
const INPUT_PRICE_PER_M = 2.50;
const OUTPUT_PRICE_PER_M = 10.00;
const CACHE_READ_PRICE_PER_M = 0.50;

const LOCAL_PROVIDERS = ['ollama', 'lmstudio', 'llamacpp', 'local', 'vllm'];

interface TokenUsageProps {
  usage: TokenUsageType;
  /** Provider of the active model; cost is hidden when it runs locally. */
  provider?: string;
}

export function TokenUsage({ usage, provider }: TokenUsageProps) {
  const isLocal = !!provider && LOCAL_PROVIDERS.includes(provider.toLowerCase());
  const inputActual = usage.input - (usage.cacheRead ?? 0);
  const estimatedCost =
    (inputActual / 1_000_000) * INPUT_PRICE_PER_M +
    ((usage.cacheRead ?? 0) / 1_000_000) * CACHE_READ_PRICE_PER_M +
    (usage.output / 1_000_000) * OUTPUT_PRICE_PER_M;

  const CONTEXT_BUDGET = 500_000;
  const budgetPct = (usage.total / CONTEXT_BUDGET) * 100;

  return (
    <div style={{ fontSize: 12 }}>
      <div style={{
        fontWeight: 600,
        color: 'var(--text-muted)',
        marginBottom: 10,
        fontSize: 10,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        Token Usage
      </div>

      <UsageRow label="Total" value={formatTokens(usage.total)} highlight />
      <UsageRow label="Input" value={formatTokens(usage.input)} />
      {usage.cacheRead != null && usage.cacheRead > 0 && (
        <>
          <UsageRow label="  Cached" value={formatTokens(usage.cacheRead)} color="var(--success)" />
          <UsageRow label="  Actual" value={formatTokens(inputActual)} />
        </>
      )}
      <UsageRow label="Output" value={formatTokens(usage.output)} />

      <div style={{
        marginTop: 10,
        paddingTop: 8,
        borderTop: '1px solid var(--border-subtle)',
      }}>
        {isLocal ? (
          <UsageRow label="Cost" value="Runs locally" color="var(--text-muted)" />
        ) : (
          <UsageRow
            label="Est. cost"
            value={`~$${estimatedCost.toFixed(4)}`}
            color="var(--warning)"
          />
        )}
      </div>

      {/* Budget bar */}
      <div style={{ marginTop: 10 }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: 'var(--text-muted)',
          marginBottom: 4,
        }}>
          <span title={`RUNE compacts the conversation as it approaches ${formatTokens(CONTEXT_BUDGET)} tokens`}>
            Used of {formatTokens(CONTEXT_BUDGET)}
          </span>
          <span>{budgetPct.toFixed(1)}%</span>
        </div>
        <div style={{
          height: 4,
          background: 'var(--bg-tertiary)',
          borderRadius: 2,
          overflow: 'hidden',
        }}>
          <div style={{
            height: '100%',
            width: `${Math.min(100, budgetPct)}%`,
            background: budgetPct > 85
              ? 'var(--danger)'
              : budgetPct > 60
                ? 'var(--warning)'
                : 'var(--accent)',
            borderRadius: 2,
            transition: 'width 0.3s ease',
          }} />
        </div>
      </div>
    </div>
  );
}

function UsageRow({ label, value, color, highlight }: {
  label: string;
  value: string;
  color?: string;
  highlight?: boolean;
}) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      padding: '2px 0',
    }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)',
        color: color || (highlight ? 'var(--text-primary)' : 'var(--text-secondary)'),
        fontWeight: highlight ? 600 : 400,
        fontSize: 11,
      }}>
        {value}
      </span>
    </div>
  );
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}
