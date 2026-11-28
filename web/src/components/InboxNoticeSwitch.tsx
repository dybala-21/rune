import { useState, type CSSProperties } from 'react';
import { INBOX_TITLE, noticePermission } from '../utils/inboxNotice';

/** Asks once, from the owner's click (MDN: Notification.requestPermission "should be made in
 * response to user interaction"), whether RUNE may tell this computer when it writes in "From RUNE".
 * Shown there and with the scheduled tasks, whose results go there: "From RUNE" is listed only once
 * it holds a message, and asking only there would miss the first one. */
export function InboxNoticeSwitch({ buttonStyle }: { buttonStyle?: CSSProperties }) {
  const [permission, setPermission] = useState(noticePermission);
  if (permission === 'unsupported') return null;
  if (permission === 'granted') return <span className="conversation-heading-note">Notifications on</span>;
  if (permission === 'denied') {
    return <span className="conversation-heading-note">Notifications are blocked for this page in the browser’s settings</span>;
  }
  return <button type="button" style={buttonStyle} title={`Shows a notification when RUNE writes in "${INBOX_TITLE}"`} onClick={() => {
    Notification.requestPermission().then(setPermission, error => {
      console.warn('From RUNE: could not ask to show notifications', error);
      setPermission(noticePermission());
    });
  }}>Notify me on this computer</button>;
}
