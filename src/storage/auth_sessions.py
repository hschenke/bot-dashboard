"""Server-seitige Session-Registrierung und Widerruf für Dashboard-Admin-Tokens."""

from __future__ import annotations

import logging
import threading
import time

from src.config.constants import AUTH_SESSION_CACHE_TTL_SECONDS
from src.config.settings import PROJECT_ROOT, AppSettings
from src.security.sanitization import resolve_safe_sqlite_path
from src.storage.executor import SqlExecutor
from src.storage.schema import AUTH_DDL_STATEMENTS
from src.storage.sqlite_executor import SqliteExecutor
from src.storage.turso_executor import TursoExecutor

logger = logging.getLogger(__name__)
_STORE_LOCK = threading.Lock()
_STORES_BY_TARGET: dict[str, AuthSessionStore] = {}

_SQL_UPSERT_SESSION = (
    "INSERT INTO dashboard_auth_sessions "
    "(session_id, fp_tag, device_key_tag, issued_at, expires_at, revoked, revoked_at, last_seen_at) "
    "VALUES (?, ?, ?, ?, ?, 0, 0, ?) "
    "ON CONFLICT(session_id) DO UPDATE SET fp_tag=excluded.fp_tag, "
    "device_key_tag=excluded.device_key_tag, expires_at=excluded.expires_at, "
    "revoked=0, revoked_at=0, last_seen_at=excluded.last_seen_at"
)
_SQL_REVOKE_SESSION = (
    "INSERT INTO dashboard_auth_sessions "
    "(session_id, fp_tag, device_key_tag, issued_at, expires_at, revoked, revoked_at, last_seen_at) "
    "VALUES (?, '', '', ?, ?, 1, ?, ?) "
    "ON CONFLICT(session_id) DO UPDATE SET revoked=1, revoked_at=excluded.revoked_at"
)


class AuthSessionStore:
    """Verwaltet aktive und widerrufene Admin-Sessions auf SQLite oder Turso Cloud."""

    def __init__(self, executor: SqlExecutor, *, is_remote_persistent: bool) -> None:
        self._executor = executor
        self._is_remote_persistent = is_remote_persistent
        self._lock = threading.Lock()
        self._initialized = False
        self._cache: dict[str, tuple[float, bool]] = {}
        self._revoked_ids: set[str] = set()
        self._revoked_before_ts: int = 0

    @property
    def is_remote_persistent(self) -> bool:
        return self._is_remote_persistent

    def _ensure_schema(self) -> None:
        if not self._initialized:
            self._executor.execute_batch([(ddl, ()) for ddl in AUTH_DDL_STATEMENTS])
            self._initialized = True

    def register_session(
        self, *, session_id: str, fp_tag: str, device_key_tag: str, issued_at: int, expires_at: int
    ) -> None:
        now_ts = int(time.time())
        with self._lock:
            self._revoked_ids.discard(session_id)
            self._cache[session_id] = (time.monotonic() + AUTH_SESSION_CACHE_TTL_SECONDS, True)
            try:
                self._ensure_schema()
                self._executor.execute_batch(
                    [
                        (_SQL_UPSERT_SESSION, (session_id, fp_tag, device_key_tag, issued_at, expires_at, now_ts)),
                        ("DELETE FROM dashboard_auth_sessions WHERE expires_at < ?", (now_ts,)),
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session-Registrierung fehlgeschlagen (%s).", exc)

    def revoke_session(self, session_id: str) -> None:
        if not session_id:
            return
        now_ts = int(time.time())
        with self._lock:
            self._revoked_ids.add(session_id)
            self._cache[session_id] = (time.monotonic() + AUTH_SESSION_CACHE_TTL_SECONDS, False)
            try:
                self._ensure_schema()
                self._executor.execute(_SQL_REVOKE_SESSION, (session_id, now_ts, now_ts + 86400, now_ts, now_ts))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session-Widerruf fehlgeschlagen (%s).", exc)

    def revoke_all_sessions(self) -> None:
        now_ts = int(time.time()) + 1
        with self._lock:
            self._revoked_before_ts = max(self._revoked_before_ts, now_ts)
            self._cache.clear()
            try:
                self._ensure_schema()
                self._executor.execute(
                    "INSERT INTO dashboard_auth_policy (policy_key, policy_value) "
                    "VALUES ('revoked_before_ts', ?) "
                    "ON CONFLICT(policy_key) DO UPDATE SET policy_value=excluded.policy_value",
                    (str(self._revoked_before_ts),),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Globaler Session-Widerruf fehlgeschlagen (%s).", exc)

    def is_session_active(
        self, session_id: str, issued_at: int, *, require_registered: bool = False
    ) -> bool:
        if not session_id:
            return False
        now_mono = time.monotonic()
        with self._lock:
            if session_id in self._revoked_ids or issued_at < self._revoked_before_ts:
                return False
            cached = self._cache.get(session_id)
            if cached and cached[0] > now_mono:
                return cached[1]
            active = self._query_session_active(session_id, issued_at, require_registered=require_registered)
            self._cache[session_id] = (now_mono + AUTH_SESSION_CACHE_TTL_SECONDS, active)
            if not active:
                self._revoked_ids.add(session_id)
            return active

    def _query_session_active(self, session_id: str, issued_at: int, *, require_registered: bool) -> bool:
        try:
            self._ensure_schema()
            sess_res, pol_res = self._executor.execute_batch(
                [
                    ("SELECT revoked FROM dashboard_auth_sessions WHERE session_id = ? LIMIT 1", (session_id,)),
                    ("SELECT policy_value FROM dashboard_auth_policy WHERE policy_key = 'revoked_before_ts' LIMIT 1", ()),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Session-Prüfung fehlgeschlagen (%s).", exc)
            return not require_registered
        if pol_res.rows:
            raw_wm = str(pol_res.rows[0].get("policy_value", "0") or "0").strip()
            if raw_wm.isdigit():
                self._revoked_before_ts = max(self._revoked_before_ts, int(raw_wm))
        if issued_at < self._revoked_before_ts:
            return False
        if not sess_res.rows:
            return not require_registered
        return int(sess_res.rows[0].get("revoked", 0) or 0) == 0

    def touch_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            try:
                self._ensure_schema()
                self._executor.execute(
                    "UPDATE dashboard_auth_sessions SET last_seen_at = ? WHERE session_id = ?",
                    (int(time.time()), session_id),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Touch-Session übersprungen (%s).", exc)

    def close(self) -> None:
        with self._lock:
            self._executor.close()


def _resolve_store_target(settings: AppSettings) -> tuple[str, str, str, bool]:
    if settings.has_turso_configured:
        return (f"turso:{settings.turso_url}", settings.turso_url, settings.turso_auth_token, True)
    if settings.brain_turso_url.strip() and settings.brain_turso_auth_token.strip():
        return (f"brain_turso:{settings.brain_turso_url}", settings.brain_turso_url, settings.brain_turso_auth_token, True)
    if settings.sqlite_path == ":memory:":
        return ("sqlite::memory:", "", "", False)
    safe_path = resolve_safe_sqlite_path(PROJECT_ROOT, settings.sqlite_path)
    return (f"sqlite:{safe_path}", str(safe_path), "", False)


def get_auth_session_store(settings: AppSettings) -> AuthSessionStore:
    """Liefert den gecachten `AuthSessionStore` für die aktuelle Konfiguration."""
    target_key, url_or_path, auth_token, is_remote = _resolve_store_target(settings)
    with _STORE_LOCK:
        existing = _STORES_BY_TARGET.get(target_key)
        if existing is not None:
            return existing
        executor: SqlExecutor = (
            TursoExecutor(database_url=url_or_path, auth_token=auth_token)
            if is_remote
            else SqliteExecutor(":memory:" if target_key == "sqlite::memory:" else url_or_path)
        )
        store = AuthSessionStore(executor, is_remote_persistent=is_remote)
        _STORES_BY_TARGET[target_key] = store
        return store


def reset_auth_session_stores() -> None:
    """Schließt und leert alle gecachten Session-Stores (für Tests)."""
    with _STORE_LOCK:
        for store in _STORES_BY_TARGET.values():
            store.close()
        _STORES_BY_TARGET.clear()
