"""Datenbank-Inspektor für die 3 Bot-Datenbanken (Turso Cloud LibSQL & lokales SQLite)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from src.checks.db_bot_inspectors import (
    BotDbInspectionDetails,
    inspect_alert_bot_details,
    inspect_news_bot_details,
    inspect_research_bot_details,
)
from src.config.settings import PROJECT_ROOT
from src.domain.enums import AppId, HealthStatus
from src.domain.exceptions import StorageError, ValidationError
from src.domain.models import (
    DatabaseHealthSummary,
    DatabaseIssue,
    DatabaseTableMetric,
    ManagedAppSpec,
)
from src.security.sanitization import (
    resolve_safe_sqlite_path,
    validate_whitelisted_table_name,
)
from src.storage.executor import SqlExecutor
from src.storage.sqlite_executor import SqliteExecutor
from src.storage.turso_executor import TursoExecutor


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _open_bot_executor(spec: ManagedAppSpec, base_dir: Path) -> SqlExecutor:
    """Öffnet bevorzugt die Turso-Cloud-Datenbank eines Bots, sonst lokales Read-Only-SQLite."""
    if spec.has_turso_credentials:
        return TursoExecutor(database_url=spec.turso_url, auth_token=spec.turso_auth_token)

    if spec.local_sqlite_path:
        safe_sqlite = resolve_safe_sqlite_path(
            base_dir,
            spec.local_sqlite_path,
            allow_workspace_parent=True,
        )
        return SqliteExecutor(safe_sqlite, read_only=True)

    raise StorageError("Weder Turso-Zugangsdaten noch lokaler SQLite-Pfad konfiguriert.")


def _collect_table_metrics(
    executor: SqlExecutor,
    expected_tables: tuple[str, ...],
) -> list[DatabaseTableMetric]:
    """Zählt Zeilen aller erwarteten Whitelist-Tabellen fehlertolerant."""
    metrics: list[DatabaseTableMetric] = []
    for raw_table in expected_tables:
        table_name = validate_whitelisted_table_name(raw_table)
        try:
            res = executor.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")  # noqa: S608
            count = int(res.rows[0].get("cnt", 0) or 0) if res.rows else 0
            metrics.append(DatabaseTableMetric(table_name=table_name, row_count=count))
        except StorageError:
            metrics.append(DatabaseTableMetric(table_name=table_name, row_count=0))
    return metrics


def _inspect_bot_details(app_id: AppId, executor: SqlExecutor) -> BotDbInspectionDetails:
    if app_id == AppId.NEWS_BOT:
        return inspect_news_bot_details(executor)
    if app_id == AppId.RESEARCH_BOT:
        return inspect_research_bot_details(executor)
    return inspect_alert_bot_details(executor)


def _format_db_status_message(
    backend_name: str,
    total_rows: int,
    latency_ms: int,
    details: BotDbInspectionDetails,
) -> str:
    if details.error_count <= 0:
        return f"Verbindung OK ({backend_name}, {total_rows} Zeilen, {latency_ms} ms)"
    preview = details.issues[0].title if details.issues else "App-Daten"
    return (
        f"DB erreichbar ({backend_name}, {total_rows} Zeilen, {latency_ms} ms) — "
        f"{details.error_count} Fehler in App-Daten ({preview})"
    )


class BotDatabaseInspector:
    """Inspiziert den Gesundheitszustand und die Inhalte der 3 Bot-Datenbanken."""

    def __init__(self, base_dir: Path = PROJECT_ROOT) -> None:
        self._base_dir = base_dir

    def inspect_bot_database(
        self,
        spec: ManagedAppSpec,
        executor_override: SqlExecutor | None = None,
    ) -> DatabaseHealthSummary:
        """Prüft Tabellen, Zeilenanzahlen, offene Fehlerbefunde und jüngste Einträge eines Bots."""
        checked_at = _utc_now_iso()
        start_ts = time.perf_counter()
        owns_executor = executor_override is None
        executor: SqlExecutor | None = executor_override

        try:
            if executor is None:
                executor = _open_bot_executor(spec, self._base_dir)
            metrics = _collect_table_metrics(executor, spec.expected_tables)
            details = _inspect_bot_details(spec.app_id, executor)

            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            total_rows = sum(m.row_count for m in metrics)
            status = HealthStatus.DEGRADED if details.error_count > 0 else HealthStatus.ONLINE
            msg = _format_db_status_message(executor.backend_name, total_rows, latency_ms, details)

            return DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label=executor.backend_name,
                status=status,
                latency_ms=latency_ms,
                table_metrics=tuple(metrics),
                recent_items=details.items,
                active_count=details.active_count,
                error_count=details.error_count,
                last_activity_at=details.last_activity,
                message=msg,
                checked_at=checked_at,
                issues=details.issues,
            )
        except (StorageError, ValidationError) as exc:
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            err_issue = DatabaseIssue(
                source="DB-Verbindung",
                title=spec.title,
                status="Offline",
                detail=str(exc)[:220],
                timestamp=checked_at,
            )
            return DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label="Nicht verbunden",
                status=HealthStatus.OFFLINE,
                latency_ms=latency_ms,
                message=str(exc),
                checked_at=checked_at,
                issues=(err_issue,),
            )
        finally:
            if owns_executor and executor is not None:
                executor.close()
