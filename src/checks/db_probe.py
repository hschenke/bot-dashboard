"""Datenbank-Inspektor für die 3 Bot-Datenbanken (Turso Cloud LibSQL & lokales SQLite)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from src.config.constants import MAX_RECENT_ITEMS_PER_APP
from src.config.settings import PROJECT_ROOT
from src.domain.enums import AppId, HealthStatus
from src.domain.exceptions import StorageError, ValidationError
from src.domain.models import (
    DatabaseHealthSummary,
    DatabaseRecentItem,
    DatabaseTableMetric,
    ManagedAppSpec,
)
from src.security.sanitization import (
    resolve_safe_sqlite_path,
    sanitize_display_url,
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


def _inspect_news_bot_details(
    executor: SqlExecutor,
) -> tuple[list[DatabaseRecentItem], int, int, str]:
    """Liest die jüngsten Artikel und KI-Briefings aus der News-Bot-Datenbank."""
    items: list[DatabaseRecentItem] = []
    last_activity = ""

    try:
        briefings = executor.execute(
            "SELECT briefing_date, model_used, created_at FROM briefings ORDER BY id DESC LIMIT ?",
            (3,),
        ).rows
        for row in briefings:
            ts = str(row.get("created_at", "") or row.get("briefing_date", ""))
            if ts and not last_activity:
                last_activity = ts
            items.append(
                DatabaseRecentItem(
                    category="Briefing",
                    title=f"KI-Briefing vom {row.get('briefing_date', '-')}",
                    subtitle=f"Modell: {row.get('model_used', 'unbekannt')}",
                    status="completed",
                    timestamp=ts,
                )
            )
    except StorageError:
        pass

    try:
        articles = executor.execute(
            """
            SELECT title, source, category, created_at, url
            FROM articles
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (MAX_RECENT_ITEMS_PER_APP,),
        ).rows
        for row in articles:
            ts = str(row.get("created_at", "") or "")
            if ts and not last_activity:
                last_activity = ts
            items.append(
                DatabaseRecentItem(
                    category=str(row.get("category", "") or "Artikel"),
                    title=str(row.get("title", "") or "Ohne Titel")[:120],
                    subtitle=str(row.get("source", "") or "RSS"),
                    status="saved",
                    timestamp=ts,
                    url=sanitize_display_url(str(row.get("url", "") or "")),
                )
            )
    except StorageError:
        pass

    return items[:MAX_RECENT_ITEMS_PER_APP], len(items), 0, last_activity


def _inspect_research_bot_details(
    executor: SqlExecutor,
) -> tuple[list[DatabaseRecentItem], int, int, str]:
    """Liest laufende/abgeschlossene Jobs und Berichte aus der Research-Bot-Datenbank."""
    items: list[DatabaseRecentItem] = []
    active_count = 0
    error_count = 0
    last_activity = ""

    try:
        counts = executor.execute(
            """
            SELECT
                SUM(CASE WHEN status IN ('running', 'queued') THEN 1 ELSE 0 END) AS active_cnt,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS err_cnt
            FROM research_jobs
            """
        ).rows
        if counts:
            active_count = int(counts[0].get("active_cnt", 0) or 0)
            error_count = int(counts[0].get("err_cnt", 0) or 0)

        jobs = executor.execute(
            """
            SELECT topic, status, phase, progress_percent, created_at, completed_at
            FROM research_jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (MAX_RECENT_ITEMS_PER_APP,),
        ).rows
        for row in jobs:
            ts = str(row.get("completed_at", "") or row.get("created_at", "") or "")
            if ts and not last_activity:
                last_activity = ts
            status = str(row.get("status", "unknown"))
            progress = int(row.get("progress_percent", 0) or 0)
            items.append(
                DatabaseRecentItem(
                    category="Research-Job",
                    title=str(row.get("topic", "Unbekanntes Thema"))[:120],
                    subtitle=f"Phase: {row.get('phase', '-')} ({progress}%)",
                    status=status,
                    timestamp=ts,
                )
            )
    except StorageError:
        pass

    return items, active_count, error_count, last_activity


def _inspect_alert_bot_details(
    executor: SqlExecutor,
) -> tuple[list[DatabaseRecentItem], int, int, str]:
    """Liest aktive Wächter-Regeln und ausgelöste Alarme aus der Alert-Bot-Datenbank."""
    items: list[DatabaseRecentItem] = []
    active_count = 0
    error_count = 0
    last_activity = ""

    try:
        counts = executor.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_cnt,
                SUM(CASE WHEN consecutive_failures > 0 OR status = 'error' THEN 1 ELSE 0 END) AS err_cnt
            FROM alert_rules
            """
        ).rows
        if counts:
            active_count = int(counts[0].get("active_cnt", 0) or 0)
            error_count = int(counts[0].get("err_cnt", 0) or 0)

        rules = executor.execute(
            """
            SELECT title, detector_type, status, last_value_text, last_checked_at, target_url
            FROM alert_rules
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (5,),
        ).rows
        for row in rules:
            ts = str(row.get("last_checked_at", "") or "")
            if ts and not last_activity:
                last_activity = ts
            val_txt = str(row.get("last_value_text", "") or "Noch kein Messwert")
            items.append(
                DatabaseRecentItem(
                    category=f"Wächter ({row.get('detector_type', '-')})",
                    title=str(row.get("title", "Regel"))[:120],
                    subtitle=val_txt[:100],
                    status=str(row.get("status", "active")),
                    timestamp=ts,
                    url=sanitize_display_url(str(row.get("target_url", "") or "")),
                )
            )

        events = executor.execute(
            """
            SELECT rule_title, headline, measured_display, created_at, source_url
            FROM alert_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (4,),
        ).rows
        for row in events:
            ts = str(row.get("created_at", "") or "")
            if ts and not last_activity:
                last_activity = ts
            items.append(
                DatabaseRecentItem(
                    category="Alarm-Treffer",
                    title=str(row.get("headline", row.get("rule_title", "Alarm")))[:120],
                    subtitle=str(row.get("measured_display", ""))[:100],
                    status="triggered",
                    timestamp=ts,
                    url=sanitize_display_url(str(row.get("source_url", "") or "")),
                )
            )
    except StorageError:
        pass

    return items[:MAX_RECENT_ITEMS_PER_APP], active_count, error_count, last_activity


class BotDatabaseInspector:
    """Inspiziert den Gesundheitszustand und die Inhalte der 3 Bot-Datenbanken."""

    def __init__(self, base_dir: Path = PROJECT_ROOT) -> None:
        self._base_dir = base_dir

    def inspect_bot_database(
        self,
        spec: ManagedAppSpec,
        executor_override: SqlExecutor | None = None,
    ) -> DatabaseHealthSummary:
        """Prüft Tabellen, Zeilenanzahlen und jüngste Einträge eines Bots."""
        checked_at = _utc_now_iso()
        start_ts = time.perf_counter()
        owns_executor = executor_override is None
        executor: SqlExecutor | None = executor_override

        try:
            if executor is None:
                executor = _open_bot_executor(spec, self._base_dir)
            metrics = _collect_table_metrics(executor, spec.expected_tables)

            if spec.app_id == AppId.NEWS_BOT:
                items, active_cnt, err_cnt, last_ts = _inspect_news_bot_details(executor)
            elif spec.app_id == AppId.RESEARCH_BOT:
                items, active_cnt, err_cnt, last_ts = _inspect_research_bot_details(executor)
            else:
                items, active_cnt, err_cnt, last_ts = _inspect_alert_bot_details(executor)

            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            total_rows = sum(m.row_count for m in metrics)
            status = HealthStatus.DEGRADED if err_cnt > 0 else HealthStatus.ONLINE
            msg = f"{executor.backend_name} ({total_rows} Zeilen, {latency_ms} ms)"

            return DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label=executor.backend_name,
                status=status,
                latency_ms=latency_ms,
                table_metrics=tuple(metrics),
                recent_items=tuple(items),
                active_count=active_cnt,
                error_count=err_cnt,
                last_activity_at=last_ts,
                message=msg,
                checked_at=checked_at,
            )
        except (StorageError, ValidationError) as exc:
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            return DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label="Nicht verbunden",
                status=HealthStatus.OFFLINE,
                latency_ms=latency_ms,
                message=str(exc),
                checked_at=checked_at,
            )
        finally:
            if owns_executor and executor is not None:
                executor.close()
