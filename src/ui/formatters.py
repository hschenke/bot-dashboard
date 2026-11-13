"""Darstellungs-Hilfsfunktionen für Badges, Zeitstempel und Status-Texte."""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any
from urllib.parse import quote

import streamlit as st

from src.domain.enums import HealthStatus, WorkflowRunState
from src.domain.exceptions import ValidationError
from src.security.sanitization import sanitize_display_url, validate_workflow_filename

_TABLE_SUPPORTS_BORDER = "border" in inspect.signature(st.table).parameters
_TABLE_SUPPORTS_HIDE_INDEX = "hide_index" in inspect.signature(st.table).parameters


def format_health_badge(status: HealthStatus) -> str:
    """Liefert ein kurzes Status-Badge im englischen Originalbegriff."""
    mapping = {
        HealthStatus.ONLINE: "🟢 Online",
        HealthStatus.ASLEEP: "🌙 Sleeping",
        HealthStatus.DEGRADED: "🟡 Degraded",
        HealthStatus.OFFLINE: "🔴 Offline",
        HealthStatus.UNKNOWN: "⚪ Unknown",
    }
    return mapping.get(status, "⚪ Unknown")


def format_workflow_badge(state: WorkflowRunState) -> str:
    """Liefert ein kurzes Status-Badge für GitHub Actions Worker (`Queued`, `Running`, etc.)."""
    mapping = {
        WorkflowRunState.RUNNING: "🔵 Running",
        WorkflowRunState.QUEUED: "⏳ Queued",
        WorkflowRunState.SUCCESS: "🟢 Success",
        WorkflowRunState.FAILURE: "🔴 Failed",
        WorkflowRunState.CANCELLED: "⚪ Cancelled",
        WorkflowRunState.UNKNOWN: "⚪ Unknown",
    }
    return mapping.get(state, "⚪ Unknown")


def format_worker_status(state: WorkflowRunState, status_label: str) -> str:
    """Kombiniert Worker-Badge und Detail-Label ohne redundante Dopplungen."""
    badge = format_workflow_badge(state)
    clean_label = (status_label or "").strip()
    badge_word = badge.split(" ", 1)[-1].lower()
    if not clean_label or clean_label.lower() == badge_word:
        return badge
    return f"{badge} ({clean_label})"


def format_iso_timestamp(raw_iso: str | datetime | None) -> str:
    """Formatiert einen Zeitstempel einheitlich im Format tt.mm.yyyy - hh:mm."""
    if not raw_iso:
        return "–"
    if isinstance(raw_iso, datetime):
        return raw_iso.strftime("%d.%m.%Y - %H:%M")
    cleaned = str(raw_iso).strip()
    if not cleaned:
        return "–"
    try:
        normalized = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%d.%m.%Y - %H:%M")
    except Exception:
        return cleaned


def build_workflow_file_url(
    github_repo_url: str,
    workflow_file: str,
    branch: str = "main",
) -> str:
    """Erstellt einen sicheren GitHub-Link direkt zur Workflow-Datei im Repository."""
    base = sanitize_display_url(github_repo_url).rstrip("/")
    if base == "#":
        return "#"
    try:
        safe_wf = validate_workflow_filename(workflow_file)
    except ValidationError:
        return "#"
    clean_branch = quote((branch or "main").strip() or "main", safe="/")
    return sanitize_display_url(f"{base}/blob/{clean_branch}/.github/workflows/{safe_wf}")


def render_markdown_table(rows: list[dict[str, Any]]) -> None:
    """Rendert eine kompakte Markdown-Tabelle ohne Index-Spalte (inkl. Versions-Fallback)."""
    kwargs: dict[str, Any] = {}
    if _TABLE_SUPPORTS_BORDER:
        kwargs["border"] = "horizontal"
    if _TABLE_SUPPORTS_HIDE_INDEX:
        kwargs["hide_index"] = True
    st.table(rows, **kwargs)
