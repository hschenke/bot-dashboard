"""Thread-sicherer Brute-Force-Schutz für Dashboard-Anmeldungen."""

from __future__ import annotations

import threading
import time

from src.config.constants import LOGIN_LOCKOUT_SECONDS, MAX_FAILED_LOGIN_ATTEMPTS


class LoginRateLimiter:
    """Thread-sicherer Brute-Force-Schutz für Passwort-Anmeldungen."""

    def __init__(
        self,
        max_attempts: int = MAX_FAILED_LOGIN_ATTEMPTS,
        lockout_seconds: int = LOGIN_LOCKOUT_SECONDS,
    ) -> None:
        self._max_attempts = max_attempts
        self._lockout_seconds = lockout_seconds
        self._lock = threading.Lock()
        self._failed_timestamps: dict[str, list[float]] = {}

    def is_locked_out(self, client_key: str = "global") -> tuple[bool, int]:
        """Prüft, ob eine Anmeldung gesperrt ist und liefert verbleibende Sperrsekunden."""
        now = time.monotonic()
        with self._lock:
            history = self._prune(client_key, now)
            if len(history) < self._max_attempts:
                return False, 0
            remaining = int(max(1.0, (history[0] + self._lockout_seconds) - now))
            return True, remaining

    def record_failure(self, client_key: str = "global") -> None:
        now = time.monotonic()
        with self._lock:
            history = self._prune(client_key, now)
            history.append(now)
            self._failed_timestamps[client_key] = history

    def reset(self, client_key: str = "global") -> None:
        with self._lock:
            self._failed_timestamps.pop(client_key, None)

    def _prune(self, client_key: str, now: float) -> list[float]:
        cutoff = now - self._lockout_seconds
        valid = [ts for ts in self._failed_timestamps.get(client_key, []) if ts > cutoff]
        self._failed_timestamps[client_key] = valid
        return valid
