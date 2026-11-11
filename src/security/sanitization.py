"""Whitelist-Validierung, Pfadschutz und Sanitization für Ausgaben und Parameter."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from src.config.constants import (
    ALLOWED_INSPECTION_TABLES,
    ALLOWED_URL_SCHEMES,
    MAX_QUESTION_LENGTH,
)
from src.domain.exceptions import ValidationError

_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}/[A-Za-z0-9_.-]{1,64}$")
_WORKFLOW_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.ya?ml$")


def validate_whitelisted_table_name(table_name: str) -> str:
    """Stellt sicher, dass nur explizit erlaubte Tabellennamen abgefragt werden."""
    cleaned = (table_name or "").strip().lower()
    if cleaned not in ALLOWED_INSPECTION_TABLES:
        raise ValidationError(f"Tabellenname '{table_name}' ist nicht in der Whitelist erlaubt.")
    return cleaned


def validate_github_repo_slug(repo: str) -> str:
    """Validiert ein GitHub-Repository im Format `owner/repo`."""
    cleaned = (repo or "").strip().strip("/")
    if not _GITHUB_REPO_RE.match(cleaned):
        raise ValidationError(f"Ungültiges GitHub-Repository-Format: '{repo}'.")
    return cleaned


def validate_workflow_filename(filename: str) -> str:
    """Validiert den Dateinamen eines GitHub Actions Workflows (`*.yaml` / `*.yml`)."""
    cleaned = (filename or "").strip()
    if not _WORKFLOW_FILE_RE.match(cleaned):
        raise ValidationError(f"Ungültiger Workflow-Dateiname: '{filename}'.")
    return cleaned


def validate_user_question(question: str) -> str:
    """Bereinigt und begrenzt eine Benutzerfrage für den KI-Assistenten."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", (question or "")).strip()
    if not cleaned:
        raise ValidationError("Bitte gib eine Frage für die KI-Analyse ein.")
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise ValidationError(f"Frage ist zu lang (max. {MAX_QUESTION_LENGTH} Zeichen).")
    return cleaned


def sanitize_display_url(url: str) -> str:
    """Erlaubt ausschließlich sichere `http`/`https`-Links für die UI-Darstellung."""
    cleaned = (url or "").strip()
    if not cleaned:
        return "#"
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES or not parsed.netloc:
        return "#"
    return cleaned


def resolve_safe_sqlite_path(base_dir: Path, raw_path: str, allow_workspace_parent: bool = False) -> Path:
    """
    Löst einen SQLite-Dateipfad sicher gegen Path-Traversal auf.
    Für eigene DBs gilt `base_dir`, für Nachbar-Bots optional `base_dir.parent`.
    """
    cleaned = (raw_path or "").strip()
    if not cleaned:
        raise ValidationError("Leerer SQLite-Dateipfad.")

    anchor = base_dir.parent.resolve() if allow_workspace_parent else base_dir.resolve()
    candidate = (base_dir / cleaned).resolve()
    if not candidate.is_relative_to(anchor):
        raise ValidationError(f"Sicherheitsverletzung: Pfad '{raw_path}' liegt außerhalb des erlaubten Bereichs.")
    return candidate
