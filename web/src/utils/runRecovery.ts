import type { SseEventType } from '../types';

type RunEvent = { runId?: string; seq?: number };
type BufferedEvent = { type: SseEventType; data: RunEvent };

/** Buffer live events while a snapshot is fetched, then apply only newer events. */
export class RunRecovery {
  private pending = false;
  private overflow = false;
  private buffer: BufferedEvent[] = [];
  private sequences = new Map<string, number>();

  constructor(private emit: (type: SseEventType, data: unknown) => void) {}

  begin() {
    this.pending = true;
    this.overflow = false;
    this.buffer = [];
  }

  receive(type: SseEventType, data: RunEvent) {
    if (this.pending) {
      if (this.buffer.length >= 512) {
        this.overflow = true;
        this.buffer = [];
      }
      this.buffer.push({ type, data });
    } else {
      this.deliver(type, data);
    }
  }

  finish(snapshot: RunEvent | null | undefined): boolean {
    if (this.overflow) return false;
    if (snapshot?.runId) {
      this.sequences.set(snapshot.runId, snapshot.seq ?? 0);
    }
    if (snapshot !== undefined) this.emit('run_snapshot', snapshot);
    this.pending = false;
    for (const { type, data } of this.buffer) this.deliver(type, data);
    this.buffer = [];
    return true;
  }

  private deliver(type: SseEventType, data: RunEvent) {
    if (data.runId && data.seq !== undefined) {
      if (data.seq <= (this.sequences.get(data.runId) ?? 0)) return;
      this.sequences.set(data.runId, data.seq);
      if (this.sequences.size > 100) this.sequences.delete(this.sequences.keys().next().value!);
    }
    this.emit(type, data);
  }
}
