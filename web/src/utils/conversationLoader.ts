import type { SessionContents } from '../api';

interface LoadOptions {
  read: (sessionId: string, signal: AbortSignal) => Promise<SessionContents>;
  currentSession: () => string;
  isBusy: () => boolean;
  activate: (sessionId: string, contents: SessionContents) => void;
  forMessage?: boolean;
}

export class ConversationLoader {
  private request: AbortController | null = null;

  get pending(): boolean { return this.request !== null; }

  cancel(): void {
    this.request?.abort();
    this.request = null;
  }

  async load(sessionId: string, options: LoadOptions): Promise<boolean> {
    if (options.isBusy()) throw new Error('Wait for the current task to finish before continuing another chat.');
    this.cancel();
    const request = new AbortController();
    this.request = request;
    const source = options.currentSession();
    const cancelled = () => request.signal.aborted || this.request !== request || source !== options.currentSession();
    try {
      const contents = await options.read(sessionId, request.signal);
      if (cancelled()) return false;
      if (options.isBusy()) throw new Error('The current chat is busy. Your draft has been kept.');
      const run = contents.run;
      if (options.forMessage && run && !['completed', 'failed', 'cancelled', 'interrupted'].includes(run.status)) {
        throw new Error('This chat already has a running task. Use Continue this chat to view it.');
      }
      options.activate(sessionId, contents);
      return true;
    } catch (error) {
      if (cancelled()) return false;
      throw error;
    } finally {
      if (this.request === request) this.request = null;
    }
  }
}
