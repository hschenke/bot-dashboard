"""Typisierte Enums für App-Identifikatoren, Gesundheitsstatus, Rollen und Workflows."""

from __future__ import annotations

from enum import StrEnum


class AppId(StrEnum):
    """Eindeutige Kennung der drei überwachten Streamlit-Applikationen."""

    NEWS_BOT = "news_bot"
    RESEARCH_BOT = "research_bot"
    ALERT_BOT = "alert_bot"


class HealthStatus(StrEnum):
    """Normierter Gesundheits- und Betriebszustand einer Komponente."""

    ONLINE = "online"
    ASLEEP = "asleep"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class WorkflowRunState(StrEnum):
    """Status eines GitHub Actions Hintergrund-Workflows."""

    RUNNING = "running"
    QUEUED = "queued"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AuthRole(StrEnum):
    """Zugriffsrolle innerhalb der Dashboard-Web-App."""

    GUEST = "guest"
    READONLY = "readonly"
    ADMIN = "admin"
