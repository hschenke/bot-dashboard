"""Darstellungs-Hilfsfunktionen für Badges, Zeitstempel und Status-Texte."""

from __future__ import annotations

from datetime import datetime

from src.domain.enums import HealthStatus, WorkflowRunState


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
