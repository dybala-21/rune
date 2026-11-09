/**
 * Behind the hosted gateway, a Rune machine goes to sleep when nobody uses it, and only what a
 * person does should wake it or keep it awake. The gateway marks its answers with
 * `X-Rune-Machine`; once we have seen that, requests sent while nobody is at the keyboard
 * (polls, refreshes, reconnects) go out with `X-Rune-Background: 1`, and input sends a
 * presence ping now and then. Against a local daemon none of this happens.
 */

// a request this soon after input is taken as caused by it
export const RECENT_INPUT_MS = 10_000;
// presence pings at most this often while someone is using the page
export const PRESENCE_EVERY_MS = 60_000;
export const PRESENCE_PATH = '/cloud/presence';
export const MACHINE_EVENT = 'rune:machine';

const INPUT_EVENTS = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart', 'focus'] as const;

interface PageLike {
  hidden: boolean;
}

let lastInput = Number.NEGATIVE_INFINITY;
let lastPresence = Number.NEGATIVE_INFINITY;
let hosted = false;
let machine: string | null = null;
let page: PageLike = typeof document === 'undefined' ? { hidden: false } : document;
let send: typeof fetch = (...args) => fetch(...args);
let announce: (status: string) => void = status => {
  if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(MACHINE_EVENT, { detail: status }));
};

/** The machine's last known status, or null when not behind the hosted gateway. */
export function machineStatus(): string | null {
  return machine;
}

export function noteResponse(response: Response): void {
  const status = response.headers.get('x-rune-machine');
  if (!status) return;
  hosted = true;
  if (status !== machine) {
    machine = status;
    announce(status);
  }
}

export function presenceHeaders(now = Date.now()): Record<string, string> {
  const active = !page.hidden && now - lastInput < RECENT_INPUT_MS;
  return hosted && !active ? { 'X-Rune-Background': '1' } : {};
}

export function noteInput(now = Date.now()): void {
  lastInput = now;
  if (hosted && !page.hidden && now - lastPresence >= PRESENCE_EVERY_MS) {
    lastPresence = now;
    void ping();
  }
}

async function ping(): Promise<void> {
  try {
    noteResponse(await send(PRESENCE_PATH, { method: 'POST', credentials: 'include', cache: 'no-store' }));
  } catch (error) {
    // the next input tries again; requests made meanwhile still wake the machine
    lastPresence = Number.NEGATIVE_INFINITY;
    console.debug('presence ping failed', error);
  }
}

/** Listen for input and for the page coming back into view. Returns the cleanup. */
export function watchPresence(target: Window = window, doc: Document = document): () => void {
  const onInput = () => noteInput();
  const onVisible = () => { if (!doc.hidden) noteInput(); };
  for (const name of INPUT_EVENTS) target.addEventListener(name, onInput, { capture: true, passive: true });
  doc.addEventListener('visibilitychange', onVisible);
  return () => {
    for (const name of INPUT_EVENTS) target.removeEventListener(name, onInput, { capture: true });
    doc.removeEventListener('visibilitychange', onVisible);
  };
}

/** Tests only: swap the page, the network and the event sink, and forget what was seen. */
export function resetPresenceForTests(options: { page?: PageLike; send?: typeof fetch; announce?: (status: string) => void } = {}): void {
  lastInput = Number.NEGATIVE_INFINITY;
  lastPresence = Number.NEGATIVE_INFINITY;
  hosted = false;
  machine = null;
  if (options.page) page = options.page;
  if (options.send) send = options.send;
  if (options.announce) announce = options.announce;
}
