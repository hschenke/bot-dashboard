"""Einheitliches Dashboard-Repository für Check-Protokolle auf SQLite & Turso."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from src.config.constants import MAX_CHECK_LOG_HISTORY
from src.config.settings import PROJECT_ROOT, AppSettings
from src.domain.models import AppFleetSnapshot, FleetCheckLog
from src.security.sanitization import resolve_safe_sqlite_path
from src.storage.executor import SqlExecutor
from src.storage.schema import DDL_STATEMENTS
from src.storage.sqlite_executor import SqliteExecutor
from src.storage.turso_executor import TursoExecutor

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Aktueller UTC-Zeitstempel im ISO-8601-Format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class SqlDashboardRepository:
    """Speichert und lädt historische Flotten-Checks mit 100 % parametrisierten Queries."""

    def __init__(self, executor: SqlExecutor) -> None:
        self._executor = executor
        self.init_schema()

    @property
    def backend_name(self) -> str:
        return self._executor.backend_name

    def init_schema(self) -> None:
        self._executor.execute_batch([(ddl, ()) for ddl in DDL_STATEMENTS])

    def record_snapshot(self, snapshot: AppFleetSnapshot) -> FleetCheckLog:
        """Speichert einen neuen Prüflauf für eine überwachte App."""
        entry = FleetCheckLog(
            id=uuid.uuid4().hex[:12],
            app_id=snapshot.spec.app_id.value,
            overall_status=snapshot.overall_status.value,
            url_status=snapshot.url_check.status.value,
            url_latency_ms=snapshot.url_check.latency_ms,
            workflow_state=snapshot.workflow_check.state.value,
            db_status=snapshot.db_check.status.value,
            db_total_rows=snapshot.db_check.total_rows,
            summary_text=(
                f"URL: {snapshot.url_check.message} | "
                f"Worker: {snapshot.workflow_check.message} | "
                f"DB: {snapshot.db_check.message}"
            )[:400],
            created_at=snapshot.checked_at or utc_now_iso(),
        )
        self._executor.execute(
            """
            INSERT INTO fleet_check_logs (
                id, app_id, overall_status, url_status, url_latency_ms,
                workflow_state, db_status, db_total_rows, summary_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.app_id,
                entry.overall_status,
                entry.url_status,
                entry.url_latency_ms,
                entry.workflow_state,
                entry.db_status,
                entry.db_total_rows,
                entry.summary_text,
                entry.created_at,
            ),
        )
        return entry

    def list_recent_logs(
        self,
        app_id: str | None = None,
        limit: int = MAX_CHECK_LOG_HISTORY,
    ) -> list[FleetCheckLog]:
        """Lädt die jüngsten Prüfprotokolle optional gefiltert nach `app_id`."""
        safe_limit = max(1, min(200, int(limit)))
        if app_id:
            res = self._executor.execute(
                """
                SELECT * FROM fleet_check_logs
                WHERE app_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (app_id.strip(), safe_limit),
            )
        else:
            res = self._executor.execute(
                """
                SELECT * FROM fleet_check_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
        return [
            FleetCheckLog(
                id=str(row.get("id", "")),
                app_id=str(row.get("app_id", "")),
                overall_status=str(row.get("overall_status", "")),
                url_status=str(row.get("url_status", "")),
                url_latency_ms=int(row.get("url_latency_ms", 0) or 0),
                workflow_state=str(row.get("workflow_state", "")),
                db_status=str(row.get("db_status", "")),
                db_total_rows=int(row.get("db_total_rows", 0) or 0),
                summary_text=str(row.get("summary_text", "")),
                created_at=str(row.get("created_at", "")),
            )
            for row in res.rows
        ]

    def close(self) -> None:
        self._executor.close()


def create_dashboard_repository(settings: AppSettings) -> SqlDashboardRepository:
    """Erstellt ein `SqlDashboardRepository` (Turso Cloud mit lokalem SQLite-Fallback)."""
    if settings.has_turso_configured:
        try:
            executor: SqlExecutor = TursoExecutor(
                database_url=settings.turso_url,
                auth_token=settings.turso_auth_token,
            )
            return SqlDashboardRepository(executor)
        except Exception as exc:
            logger.warning("Turso für Dashboard nicht erreichbar (%s) – nutze SQLite.", exc)

    if settings.sqlite_path == ":memory:":
        return SqlDashboardRepository(SqliteExecutor(":memory:"))

    safe_path = resolve_safe_sqlite_path(PROJECT_ROOT, settings.sqlite_path)
    return SqlDashboardRepository(SqliteExecutor(safe_path))
