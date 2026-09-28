import type { ChatMessage, SentAttachment } from '../types';

export function retryRequestId(message: ChatMessage | undefined, text: string, attachments: SentAttachment[] = []): string | undefined {
  if (!message?.requestId || message.delivery !== 'unknown' || message.content !== text) return;
  const previous = message.attachments ?? [];
  if (previous.length !== attachments.length) return;
  return previous.every((file, i) => {
    const next = attachments[i];
    return file.name === next.name && file.mimeType === next.mimeType
      && (file.ref ? file.ref === next.ref : file.dataUrl === next.dataUrl);
  }) ? message.requestId : undefined;
}

export function updateDelivery(messages: ChatMessage[], requestId: string, delivery: ChatMessage['delivery'], attachments?: SentAttachment[]): ChatMessage[] {
  return messages.filter(message => delivery !== 'accepted' || message.id !== `delivery-${requestId}`)
    .map(message => message.requestId === requestId ? {
    ...message, delivery,
    attachments: attachments?.length ? attachments.map((file, i) => ({ ...message.attachments?.[i], ...file })) : message.attachments,
  } : message);
}
