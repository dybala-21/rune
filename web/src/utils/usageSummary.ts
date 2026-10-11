import type { TokenUsage } from '../types';

export function usageSummary(usage: TokenUsage): string {
  const cost = usage.cost;
  const partial = !!(usage.unreportedCalls || cost?.incomplete);
  const lines = [`${partial ? 'Reported tokens' : 'Token usage'}: ${usage.input} input / ${usage.output} output (total ${usage.total}).`];
  if (usage.modelCalls !== undefined) lines.push(`Model requests: ${usage.modelCalls}.`);
  lines.push(`Estimated token cost: ${cost?.pending ? 'updating' : cost?.usd == null ? 'unavailable' : `~$${cost.usd.toFixed(4)}`}.`);
  if (cost?.usd == null && cost && cost.knownUsd > 0) {
    lines.push(`Known portion: ~$${cost.knownUsd.toFixed(4)}; the total is not yet known.`);
  }
  if (partial) lines.push('Some usage was not fully reported.');
  if (cost?.unpricedCalls) lines.push(`${cost.unpricedCalls} request(s) could not be priced.`);
  lines.push('Estimate, not an invoice. Excludes embeddings, tool fees, storage and account discounts.');
  return lines.join('\n');
}
