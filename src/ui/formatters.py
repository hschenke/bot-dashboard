"""Darstellungs-Hilfsfunktionen für Badges, Zeitstempel und Status-Texte."""

from __future__ import annotations

from datetime import datetime

from src.domain.enums import HealthStatus, WorkflowRunState


def format_health_badge(status: HealthStatus) -> str:
    """Liefert ein übersichtliches Status-Label mit passendem Icon."""
    mapping = {
        HealthStatus.ONLINE: "🟢 Online",
        HealthStatus.ASLEEP: "🌙 Ruhezustand (Wakeup)",
        HealthStatus.DEGRADED: "🟡 Eingeschränkt",
        HealthStatus.OFFLINE: "🔴 Offline / Nicht verbunden",
        HealthStatus.UNKNOWN: "⚪ Unbekannt",
    }
    return mapping.get(status, "⚪ Unbekannt")


def format_workflow_badge(state: WorkflowRunState) -> str:
    """Liefert ein Status-Badge für einen GitHub Actions Workflow-Lauf."""
    mapping = {
        WorkflowRunState.RUNNING: "🔵 Läuft gerade",
        WorkflowRunState.QUEUED: "⏳ Warteschlange",
        WorkflowRunState.SUCCESS: "🟢 Erfolgreich",
        WorkflowRunState.FAILURE: "🔴 Fehlgeschlagen",
        WorkflowRunState.CANCELLED: "⚪ Abgebrochen",
        WorkflowRunState.UNKNOWN: "⚪ Unbekannt",
    }
    return mapping.get(state, "⚪ Unbekannt")


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
