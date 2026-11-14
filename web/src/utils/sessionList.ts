import type { SessionInfo } from '../api';

const PAGE_SIZE = 50;
type Page = { sessions: SessionInfo[]; total: number };
type ReadPage = (params: { limit: number; offset: number; query: string }) => Promise<Page>;
export interface SessionListState {
  sessions: SessionInfo[];
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  error: 'initial' | 'more' | 'refresh' | null;
}
export const emptySessionList = (): SessionListState => ({
  sessions: [], loading: true, loadingMore: false, hasMore: false, error: null,
});

/** Keeps stale searches and overlapping refreshes from replacing a newer history page. */
export class SessionListLoader {
  private generation = 0;
  private pending = false;
  private offset = 0;
  private query = '';
  private state = emptySessionList();

  constructor(private read: ReadPage, private update: (state: SessionListState) => void) {}

  cancel() { this.generation++; this.pending = false; }

  reset(query: string) {
    this.cancel();
    this.query = query;
    this.offset = 0;
    this.publish(emptySessionList());
  }

  load() { return this.request(false); }
  loadMore() { return this.state.hasMore ? this.request(true) : Promise.resolve(); }
  retry() { return this.state.error === 'more' ? this.loadMore() : this.load(); }

  refresh() {
    // Browsing older pages/search results is user-driven. Never grow a periodic
    // request or discard those pages just to keep the recent list up to date.
    if (this.query || this.offset > PAGE_SIZE || this.state.loading) return Promise.resolve();
    return this.request(false, true);
  }

  private publish(state: SessionListState) { this.state = state; this.update(state); }

  private async request(append: boolean, background = false) {
    if (this.pending) return;
    this.pending = true;
    const generation = ++this.generation;
    const offset = append ? this.offset : 0;
    this.publish({ ...this.state, loading: !append && !background,
      loadingMore: append, error: null });
    try {
      const page = await this.read({ limit: PAGE_SIZE, offset, query: this.query });
      if (generation !== this.generation) return;
      this.offset = offset + page.sessions.length;
      const sessions = append
        ? [...new Map([...this.state.sessions, ...page.sessions].map(session => [session.id, session])).values()]
        : page.sessions;
      this.publish({ sessions, loading: false, loadingMore: false,
        hasMore: page.sessions.length > 0 && this.offset < page.total, error: null });
    } catch {
      if (generation !== this.generation) return;
      this.publish({ ...this.state, loading: false, loadingMore: false,
        error: append ? 'more' : background ? 'refresh' : 'initial' });
    } finally {
      if (generation === this.generation) this.pending = false;
    }
  }
}
