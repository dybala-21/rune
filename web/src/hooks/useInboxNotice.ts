import { useEffect, useRef } from 'react';
import { fetchSessionTurns } from '../api';
import { noticePermission, noticeText } from '../utils/inboxNotice';
import { INBOX_ID } from '../utils/unread';

/** Tells the computer when RUNE writes in "From RUNE" and the owner is not reading it there: the
 * chat list counts its messages (every 30 s), and a count higher than the last one is a new
 * message. What was there when the app opened is not told again; the chat list marks it. */
export function useInboxNotice(messages: number | undefined, reading: boolean, open: () => void) {
  const known = useRef<number | undefined>(undefined);
  const latest = useRef({ reading, open });
  latest.current = { reading, open };

  useEffect(() => {
    if (messages === undefined) return;
    const before = known.current;
    known.current = messages;
    if (before === undefined || messages <= before || noticePermission() !== 'granted') return;
    if (latest.current.reading && document.visibilityState === 'visible' && document.hasFocus()) return;
    void tell(() => latest.current.open());
  }, [messages]);
}

async function tell(open: () => void) {
  try {
    const { turns } = await fetchSessionTurns(INBOX_ID);
    const last = [...turns].reverse().find(turn => turn.role === 'assistant');
    if (!last?.content.trim()) return;
    const { title, body } = noticeText(last.content);
    // One per conversation: a newer message replaces the one still on screen, and two open tabs
    // show it once.
    const notice = new Notification(title, { body, tag: INBOX_ID });
    notice.onclick = () => { window.focus(); open(); notice.close(); };
  } catch (error) {
    console.warn('From RUNE: no notification was shown for the new message', error);
  }
}
