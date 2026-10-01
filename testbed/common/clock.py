"""Clocks. The real clock drives the testbed; the simulated clock lets the same scheduler and
traffic code run instantly in tests and in the synthetic stand-in."""
from __future__ import annotations

import threading
import time


class RealClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def sleep(self, seconds: float, stop: threading.Event | None = None) -> bool:
        """Sleep, waking early if `stop` is set. Returns False if stopped."""
        if stop is None:
            time.sleep(max(0.0, seconds))
            return True
        return not stop.wait(max(0.0, seconds))


class SimClock:
    """Deterministic clock: sleep() advances time instantly."""

    def __init__(self, start_ms: int):
        self._t = int(start_ms)
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        with self._lock:
            return self._t

    def sleep(self, seconds: float, stop: threading.Event | None = None) -> bool:
        if stop is not None and stop.is_set():
            return False
        with self._lock:
            self._t += int(round(seconds * 1000))
        return True

    def advance_to(self, t_ms: int) -> None:
        with self._lock:
            self._t = max(self._t, int(t_ms))
