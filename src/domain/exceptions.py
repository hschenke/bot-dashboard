"""Anwendungsspezifische Exception-Hierarchie für saubere Fehlerbehandlung."""

from __future__ import annotations


class DashboardError(Exception):
    """Basisklasse für alle fachlichen und technischen Fehler im Bot-Dashboard."""


class SSRFBlockedError(DashboardError):
    """Wird ausgelöst, wenn eine URL gegen SSRF- oder DNS-Rebinding-Regeln verstößt."""


class ValidationError(DashboardError):
    """Wird ausgelöst, wenn Eingabeparameter nicht der Whitelist entsprechen."""


class StorageError(DashboardError):
    """Allgemeiner Datenbank- oder Persistenzfehler."""


class StorageConnectionError(StorageError):
    """Verbindungsfehler zu Turso Cloud LibSQL oder lokalem SQLite."""


class ProbeError(DashboardError):
    """Fehler beim Prüfen einer URL, GitHub Action oder Fremd-Datenbank."""


class LLMProviderError(DashboardError):
    """Fehler bei der Kommunikation mit der Google Gemini Interactions API."""
