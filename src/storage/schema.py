"""Idempotentes Datenbankschema für die eigene Historie des Bot-Dashboards."""

from __future__ import annotations

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS fleet_check_logs (
        id TEXT PRIMARY KEY,
        app_id TEXT NOT NULL,
        overall_status TEXT NOT NULL,
        url_status TEXT NOT NULL,
        url_latency_ms INTEGER NOT NULL DEFAULT 0,
        workflow_state TEXT NOT NULL,
        db_status TEXT NOT NULL,
        db_total_rows INTEGER NOT NULL DEFAULT 0,
        summary_text TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fleet_check_logs_created
    ON fleet_check_logs (created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fleet_check_logs_app
    ON fleet_check_logs (app_id, created_at DESC)
    """,
)
