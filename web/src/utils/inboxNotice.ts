// What the computer's notification says when RUNE writes in "From RUNE": the message's first line
// as its title, the rest on one line under it. The message is what code wrote (a scheduled job's
// title and result, rune/daemon/gateway.py), so the notification repeats it and adds nothing.

export const INBOX_TITLE = 'From RUNE';

// macOS keeps 256 bytes of a notification (Electron, docs/tutorial/notifications); a Korean
// character is three of them in UTF-8. Cut here, between characters, rather than mid-character there.
const TITLE_BYTES = 64;
const NOTICE_BYTES = 250;
const ELLIPSIS = '…';

const encoder = new TextEncoder();
const bytes = (text: string) => encoder.encode(text).length;

function cut(text: string, limit: number): string {
  if (bytes(text) <= limit) return text;
  let kept = '';
  let used = bytes(ELLIPSIS);
  for (const char of text) {
    used += bytes(char);
    if (used > limit) break;
    kept += char;
  }
  return kept.trimEnd() + ELLIPSIS;
}

export function noticeText(message: string): { title: string; body: string } {
  const [first = '', ...rest] = message.trim().split('\n');
  const title = cut(first.trim(), TITLE_BYTES) || INBOX_TITLE;
  const body = cut(rest.join(' ').replace(/\s+/g, ' ').trim(), NOTICE_BYTES - bytes(title));
  return { title, body };
}

export type NoticePermission = NotificationPermission | 'unsupported';

export function noticePermission(): NoticePermission {
  return typeof Notification === 'undefined' ? 'unsupported' : Notification.permission;
}
