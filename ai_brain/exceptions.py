"""Typisierte Exception-Hierarchie für das AI & Websearch Brain Framework."""

from __future__ import annotations


class BrainError(Exception):
    """Basisklasse für alle Fehler im AI & Websearch Brain."""


class BrainSSRFError(BrainError):
    """Wird ausgelöst, wenn eine URL gegen SSRF- oder DNS-Rebinding-Regeln verstößt."""


class BrainValidationError(BrainError):
    """Wird ausgelöst, wenn Eingaben, Modell-IDs oder Tool-Parameter ungültig sind."""


class BrainStorageError(BrainError):
    """Datenbank- oder Persistenzfehler des Brain-Katalogs."""


class BrainProviderError(BrainError):
    """Fehler bei der Kommunikation mit einem AI-Provider oder Such-Tool."""


class BrainAuthError(BrainError):
    """Wird ausgelöst, wenn eine App nicht für das AI-Brain autorisiert ist."""

