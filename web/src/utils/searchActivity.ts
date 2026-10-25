import type { SearchResultItem, ToolCall } from '../types';

export function sourceUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export function sourceHost(value: string): string {
  const url = sourceUrl(value);
  return url ? new URL(url).hostname.replace(/^www\./, '') : '';
}

export function excerptText(value: string): string {
  return value.replace(/<[^>]*>/g, '').replace(/&(?:amp|quot|apos|lt|gt|nbsp);/g, entity => ({
    '&amp;': '&', '&quot;': '"', '&apos;': "'", '&lt;': '<', '&gt;': '>', '&nbsp;': ' ',
  })[entity] ?? entity).trim();
}

export function searchStatus(tool: ToolCall, active: boolean) {
  if (tool.success === false || tool.checkStatus === 'fail') return 'failed';
  if (tool.result === undefined) return active ? 'pending' : 'interrupted';
  return tool.success === true ? 'complete' : 'unknown';
}

export interface SearchPreview {
  items: SearchResultItem[];
  count?: number;
}

export function searchPreview(tool: ToolCall): SearchPreview {
  if (tool.searchResults && Array.isArray(tool.searchResults.items)) {
    const items = tool.searchResults.items.slice(0, 20).flatMap(item => {
      if (!item || typeof item.url !== 'string' || !sourceUrl(item.url)) return [];
      return [{ title: excerptText(String(item.title || sourceHost(item.url))).slice(0, 300),
        url: sourceUrl(item.url)!, description: excerptText(String(item.description || '')).slice(0, 600),
        date: excerptText(String(item.date || '')).slice(0, 100) }];
    });
    const count = tool.searchResults.count;
    return { items, count: Number.isSafeInteger(count) && count >= items.length ? count : undefined };
  }

  // Older runs only have numbered text results.
  const text = tool.result ?? '';
  if (text.trim() === 'No search results found.') return { items: [], count: 0 };
  const lines = text.split('\n');
  const items: SearchResultItem[] = [];
  let sequential = true;
  for (let index = 0; index < lines.length && items.length < 20; index++) {
    const heading = /^(\d+)\. (.+)$/.exec(lines[index]);
    const url = heading ? sourceUrl((lines[index + 1] ?? '').trim()) : null;
    if (!heading || !url) continue;
    sequential &&= Number(heading[1]) === items.length + 1;
    const body: string[] = [];
    let date: string | undefined;
    index += 2;
    for (; index < lines.length && lines[index].trim(); index++) {
      const line = lines[index].trim();
      if (line.startsWith('... (')) break;
      const reported = /^Page date reported by search: (.+) \(not necessarily the date of the facts\)$/.exec(line);
      if (reported) date = reported[1];
      else body.push(line);
    }
    items.push({ title: excerptText(heading[2]), url, description: excerptText(body.join(' ')), date });
  }
  const complete = sequential && !tool.outputTruncated && text.length < 2000 && !text.includes('... (');
  return { items, count: complete && items.length ? items.length : undefined };
}
