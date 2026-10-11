import type { TokenUsage as TokenUsageType } from '../types';

interface TokenUsageProps {
  usage: TokenUsageType;
  provider?: string;
}

export function TokenUsage({ usage }: TokenUsageProps) {
  const uncached = Math.max(0, usage.input - (usage.cacheRead ?? 0) - (usage.cacheCreation ?? 0));
  const cost = usage.cost?.usd;
  const partial = !!(usage.unreportedCalls || usage.cost?.incomplete);
  return (
    <div style={{ fontSize: 12 }}>
      <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: 10, fontSize: 10,
        textTransform: 'uppercase', letterSpacing: '0.5px' }}>Token Usage</div>
      <UsageRow label={partial ? 'Reported tokens' : 'Total'} value={formatTokens(usage.total)} highlight />
      {usage.modelCalls !== undefined && <UsageRow label="Model requests" value={String(usage.modelCalls)} />}
      <UsageRow label="Input" value={formatTokens(usage.input)} />
      {!!usage.cacheRead && <UsageRow label="Cached" value={formatTokens(usage.cacheRead)} color="var(--success)" />}
      {!!usage.cacheCreation && <UsageRow label="Cache writes" value={formatTokens(usage.cacheCreation)} />}
      {!!(usage.cacheRead || usage.cacheCreation) && <UsageRow label="Uncached" value={formatTokens(uncached)} />}
      <UsageRow label="Output" value={formatTokens(usage.output)} />
      <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid var(--border-subtle)' }}>
        <UsageRow label="Est. token cost" value={usage.cost?.pending ? 'Updating…' : cost == null ? 'Unavailable' : `~$${cost.toFixed(4)}`}
          color="var(--warning)" />
        {cost == null && (usage.cost?.knownUsd ?? 0) > 0 && (
          <UsageRow label="Known portion" value={`~$${usage.cost!.knownUsd.toFixed(4)}`} />
        )}
        <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 10 }}>
          {usage.cost?.unpricedCalls ? `${usage.cost.unpricedCalls} request(s) could not be priced. ` : ''}
          {usage.cost?.pending ? 'Background work is still being accounted for. ' : ''}
          {usage.cost?.incomplete ? 'Some usage was not fully reported. ' : ''}
          {usage.unreportedCalls ? `${usage.unreportedCalls} request(s) have no reported token usage. ` : ''}
          Estimate, not an invoice. Excludes embeddings, tool fees, storage and account discounts.
        </div>
      </div>
    </div>
  );
}

function UsageRow({ label, value, color, highlight }: {
  label: string; value: string; color?: string; highlight?: boolean;
}) {
  return <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 4 }}>
    <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    <span style={{ color: color ?? 'var(--text)', fontWeight: highlight ? 600 : 400 }}>{value}</span>
  </div>;
}

function formatTokens(tokens: number): string {
  return tokens >= 1_000_000 ? `${(tokens / 1_000_000).toFixed(2)}M`
    : tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens);
}
