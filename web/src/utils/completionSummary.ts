import type { ActivitySummary } from '../types';

export function buildCompletionNarrative(summary: ActivitySummary): string {
  if (!summary.success) {
    if (summary.totalToolCalls > 0 && summary.filesWritten > 0) {
      return `I hit a stopping point after ${summary.totalToolCalls} tool call${pluralize(summary.totalToolCalls)} and ${summary.filesWritten} file update${pluralize(summary.filesWritten)}.`;
    }
    if (summary.totalToolCalls > 0) {
      return `I hit a stopping point after ${summary.totalToolCalls} tool call${pluralize(summary.totalToolCalls)}.`;
    }
    return 'I hit a stopping point before the run could finish cleanly.';
  }

  if (summary.totalToolCalls > 0 && summary.filesWritten > 0) {
    return `I wrapped up after ${summary.totalToolCalls} tool call${pluralize(summary.totalToolCalls)} and ${summary.filesWritten} file update${pluralize(summary.filesWritten)}.`;
  }
  if (summary.totalToolCalls > 0) {
    return `I wrapped up after ${summary.totalToolCalls} tool call${pluralize(summary.totalToolCalls)}.`;
  }
  return 'I wrapped up the run.';
}

function pluralize(count: number): string {
  return count === 1 ? '' : 's';
}
