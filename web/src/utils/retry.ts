import type { ChatMessage } from '../types';
import { describeTrust } from './trust';

export function latestTurn(messages: ChatMessage[]) {
  let user: ChatMessage | undefined;
  let assistant: ChatMessage | undefined;
  let failure: ChatMessage | undefined;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role === 'user') { user = message; break; }
    if (!assistant && message.role === 'assistant' && !message.trust) assistant = message;
    if (!failure && (message.level === 'error' || message.trust && describeTrust(message.trust).ok === false)) failure = message;
  }
  return { user, assistant, retryAnchor: user ? failure ?? assistant : undefined };
}

export function retryAttachments(message: ChatMessage) {
  return (message.attachments ?? []).map(attachment => {
    if (attachment.ref) return { name: attachment.name, mimeType: attachment.mimeType, ref: attachment.ref };
    const match = /^data:[^,]*;base64,([\s\S]+)$/.exec(attachment.dataUrl ?? '');
    if (!match) throw new Error(`Reattach ${attachment.name} to retry this request. The saved conversation does not contain the file.`);
    return { name: attachment.name, mimeType: attachment.mimeType, data: match[1] };
  });
}
