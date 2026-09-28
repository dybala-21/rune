const DOCUMENT_MIMES: Record<string, string> = {
  txt: 'text/plain', md: 'text/markdown', csv: 'text/csv', tsv: 'text/tab-separated-values',
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
};
const IMAGE_MIMES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);
export const ATTACHMENT_ACCEPT = [...IMAGE_MIMES, ...Object.keys(DOCUMENT_MIMES).map(extension => `.${extension}`)].join(',');
export const ATTACHMENT_HELP = 'Images, PDF, TXT, Markdown, CSV, TSV, DOCX, XLSX or PPTX · 20MB per file · 10 files per message / 40MB across drafts';
export const MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024;
export const MAX_TOTAL_BYTES = 40 * 1024 * 1024;
export const MAX_ATTACHMENTS = 10;

export function attachmentMime(file: { name: string; type: string }): string | undefined {
  if (IMAGE_MIMES.has(file.type)) return file.type;
  const extension = file.name.split('.').pop()?.toLowerCase() ?? '';
  return DOCUMENT_MIMES[extension];
}
