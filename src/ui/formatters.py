"""Darstellungs-Hilfsfunktionen für Badges, Zeitstempel und Status-Texte."""

from __future__ import annotations

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


def format_iso_timestamp(raw_iso: str) -> str:
    """Formatiert einen ISO-Zeitstempel kompakt für die UI-Anzeige."""
    cleaned = (raw_iso or "").strip()
    if not cleaned:
        return "–"
    return cleaned.replace("T", " ").replace("+00:00", " UTC").replace("Z", " UTC")[:23]
