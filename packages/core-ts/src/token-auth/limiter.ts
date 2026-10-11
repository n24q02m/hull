/**
 * Per-principal request limiter, mirroring core-py
 * `hull_core.limits.limiter.SlidingWindowLimiter`.
 *
 * Deliberately in-memory: limits guard a single server process. Hosts that
 * shard one namespace across processes terminate the limiter at their load
 * balancer instead.
 */

const WINDOW_SECONDS = 60

function monotonicSeconds(): number {
  return performance.now() / 1000
}

export class SlidingWindowLimiter {
  private readonly events = new Map<string, number[]>()

  /**
   * Record a request for `key` and report whether it is allowed.
   *
   * `rpm = null` means unlimited. The request is recorded either way so
   * bursts cannot pre-load the window after a limit change. `now` is in
   * seconds on a monotonic clock (tests inject it).
   */
  check(key: string, rpm: number | null, now?: number): boolean {
    const current = now ?? monotonicSeconds()
    let events = this.events.get(key)
    if (events === undefined) {
      events = []
      this.events.set(key, events)
    }
    const cutoff = current - WINDOW_SECONDS
    let drop = 0
    while (drop < events.length && events[drop] <= cutoff) drop++
    if (drop > 0) events.splice(0, drop)
    events.push(current)
    return rpm === null || events.length <= rpm
  }
}
