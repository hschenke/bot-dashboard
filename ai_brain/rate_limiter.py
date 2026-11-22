"""Thread-sicherer Sliding-Window Rate Limiter gegen Free-Tier 429-Limits."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_RPM = 12
DEFAULT_MIN_CALL_INTERVAL_SECONDS = 4.0
DEFAULT_429_COOLDOWN_SECONDS = 15.0


def _is_test_environment() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


class LlmRateLimiter:
    """Sichert prozessweit konfigurierte RPM-Grenzen und Mindestabstände zwischen LLM-Calls."""

    def __init__(
        self,
        max_rpm: int = DEFAULT_RATE_LIMIT_RPM,
        min_interval_seconds: float = DEFAULT_MIN_CALL_INTERVAL_SECONDS,
    ) -> None:
        self._max_rpm = max(1, max_rpm)
        self._min_interval = 0.0 if _is_test_environment() else max(0.0, min_interval_seconds)
        self._timestamps: deque[float] = deque()
        self._last_call_monotonic: float = 0.0
        self._cooldown_until: float = 0.0
        self._lock = threading.Lock()

    def update_config(self, max_rpm: int, min_interval_seconds: float) -> None:
        with self._lock:
            self._max_rpm = max(1, max_rpm)
            self._min_interval = 0.0 if _is_test_environment() else max(0.0, min_interval_seconds)

    def notify_rate_limit(self, cooldown_seconds: float = DEFAULT_429_COOLDOWN_SECONDS) -> None:
        if _is_test_environment():
            return
        with self._lock:
            target = time.monotonic() + max(1.0, cooldown_seconds)
            if target > self._cooldown_until:
                self._cooldown_until = target
                logger.warning("Rate-Limit (429) gemeldet – Cooldown %.1fs.", cooldown_seconds)

    def acquire(self, cancel_check: Callable[[], None] | None = None) -> None:
        if _is_test_environment():
            if cancel_check is not None:
                cancel_check()
            return
        while True:
            if cancel_check is not None:
                cancel_check()
            wait_needed = 0.0
            with self._lock:
                now = time.monotonic()
                if now < self._cooldown_until:
                    wait_needed = self._cooldown_until - now
                else:
                    window_start = now - 60.0
                    while self._timestamps and self._timestamps[0] <= window_start:
                        self._timestamps.popleft()
                    if len(self._timestamps) >= self._max_rpm:
                        wait_needed = max(wait_needed, self._timestamps[0] + 60.0 - now)
                    interval_remaining = (self._last_call_monotonic + self._min_interval) - now
                    if interval_remaining > 0:
                        wait_needed = max(wait_needed, interval_remaining)
                    if wait_needed <= 0.001:
                        self._timestamps.append(now)
                        self._last_call_monotonic = now
                        return
            time.sleep(min(0.25, max(0.05, wait_needed)))

    def get_status(self) -> dict[str, float | int | bool]:
        with self._lock:
            now = time.monotonic()
            window_start = now - 60.0
            recent = sum(1 for ts in self._timestamps if ts > window_start)
            active = now < self._cooldown_until
            rem = max(0.0, self._cooldown_until - now) if active else 0.0
            return {
                "max_rpm": self._max_rpm,
                "current_window_rpm": recent,
                "min_interval_seconds": self._min_interval,
                "cooldown_active": active,
                "cooldown_remaining_seconds": round(rem, 1),
            }

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._last_call_monotonic = 0.0
            self._cooldown_until = 0.0


_GLOBAL_RATE_LIMITER = LlmRateLimiter()


def get_global_rate_limiter() -> LlmRateLimiter:
    return _GLOBAL_RATE_LIMITER
