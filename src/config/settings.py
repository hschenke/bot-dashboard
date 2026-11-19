"""Sicheres Laden der Dashboard-Konfiguration aus ENV, Streamlit Secrets und YAML."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.constants import (
    DEFAULT_BRAIN_SQLITE_REL_PATH,
    DEFAULT_SQLITE_REL_PATH,
)
from src.domain.enums import AppId
from src.domain.models import ManagedAppSpec

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _lookup_in_mapping(mapping: Any, key: str) -> str | None:
    """Sucht Schlüssel flach oder verschachtelt (`FOO_BAR` -> `[foo] bar`) in Mappings."""
    if not mapping or not hasattr(mapping, "get"):
        return None
    for cand in (key, key.lower()):
        val = mapping.get(cand)
        if val is not None:
            return str(val).strip()
    parts = key.split("_")
    for idx in range(1, len(parts)):
        prefix = "_".join(parts[:idx])
        suffix = "_".join(parts[idx:])
        for p_cand in (prefix, prefix.lower()):
            sec = mapping.get(p_cand)
            if hasattr(sec, "get"):
                for s_cand in (suffix, suffix.lower()):
                    s_val = sec.get(s_cand)
                    if s_val is not None:
                        return str(s_val).strip()
    return None


def _load_local_secrets_mapping() -> dict[str, Any]:
    """Liest `.streamlit/secrets.toml` genau einmal als Mapping ein (Performance)."""
    target = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not target.is_file():
        return {}
    try:
        parsed = tomllib.loads(target.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.debug("Fehler beim Lesen von %s: %s", target, exc)
        return {}


def _read_local_secrets_toml(key: str, local_secrets: dict[str, Any] | None = None) -> str | None:
    """Liest Schlüssel aus vorberechnetem Secrets-Mapping oder direkt aus .streamlit/secrets.toml."""
    data = local_secrets if local_secrets is not None else _load_local_secrets_mapping()
    return _lookup_in_mapping(data, key) if data else None


def _read_streamlit_secret(key: str) -> str | None:
    """Liest optional einen Schlüssel aus `st.secrets` (flach oder verschachtelt)."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            return _lookup_in_mapping(st.secrets, key)
    except Exception as exc:
        logger.debug("Streamlit Secret '%s' nicht lesbar: %s", key, exc)
    return None


def get_secret_or_env(
    key: str,
    default: str = "",
    *,
    local_secrets: dict[str, Any] | None = None,
) -> str:
    """Ermittelt Konfigurationswert priorisiert aus ENV, `st.secrets` & `secrets.toml`."""
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return (
        _read_streamlit_secret(key)
        or _read_local_secrets_toml(key, local_secrets=local_secrets)
        or default
    )


def _load_yaml_defaults(yaml_path: Path | None = None) -> dict[str, Any]:
    """Lädt deklarative Standardeinstellungen sicher mit `yaml.safe_load`."""
    target = yaml_path or (PROJECT_ROOT / "config" / "settings.yaml")
    if not target.is_file():
        return {}
    try:
        parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Konnte settings.yaml nicht lesen (%s): %s", target, exc)
        return {}


def _is_placeholder(val: str) -> bool:
    clean = val.strip().lower()
    return (
        not clean
        or clean.startswith("your_")
        or clean.startswith("dein_")
        or "deine-datenbank" in clean
        or val.strip() == "AIzaSy..."
    )


def _clean_secret(key: str, local_secrets: dict[str, Any] | None = None) -> str:
    raw = get_secret_or_env(key, local_secrets=local_secrets)
    return "" if _is_placeholder(raw) else raw.strip()


@dataclass(frozen=True)
class AppSettings:
    """Zentrales Konfigurationsobjekt (`repr=False` auf allen Geheimnissen)."""

    gemini_api_key: str = field(default="", repr=False)
    gemini_free_api_key: str = field(default="", repr=False)
    gemini_paid_api_key: str = field(default="", repr=False)
    browserbase_api_key: str = field(default="", repr=False)
    browserbase_project_id: str = ""
    brave_search_api_key: str = field(default="", repr=False)
    tavily_api_key: str = field(default="", repr=False)
    serper_api_key: str = field(default="", repr=False)
    reddit_client_id: str = field(default="", repr=False)
    reddit_client_secret: str = field(default="", repr=False)
    groq_api_key: str = field(default="", repr=False)
    groq_paid_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_paid_api_key: str = field(default="", repr=False)
    cerebras_api_key: str = field(default="", repr=False)
    cerebras_paid_api_key: str = field(default="", repr=False)
    mistral_api_key: str = field(default="", repr=False)
    mistral_paid_api_key: str = field(default="", repr=False)

    github_token: str = field(default="", repr=False)
    turso_url: str = ""
    turso_auth_token: str = field(default="", repr=False)
    sqlite_path: str = DEFAULT_SQLITE_REL_PATH
    brain_turso_url: str = ""
    brain_turso_auth_token: str = field(default="", repr=False)
    brain_secret_key: str = field(default="", repr=False)
    brain_sqlite_path: str = DEFAULT_BRAIN_SQLITE_REL_PATH

    app_password: str = field(default="", repr=False)
    streamlit_app_url: str = ""
    default_language: str = "de"
    managed_apps: tuple[ManagedAppSpec, ...] = ()

    @property
    def has_gemini_configured(self) -> bool:
        return any(
            not _is_placeholder(k)
            for k in (self.gemini_free_api_key, self.gemini_paid_api_key, self.gemini_api_key)
        )

    @property
    def has_any_ai_key_configured(self) -> bool:
        return any(
            not _is_placeholder(k)
            for k in (
                self.gemini_api_key,
                self.gemini_free_api_key,
                self.gemini_paid_api_key,
                self.groq_api_key,
                self.groq_paid_api_key,
                self.openrouter_api_key,
                self.openrouter_paid_api_key,
                self.cerebras_api_key,
                self.cerebras_paid_api_key,
                self.mistral_api_key,
                self.mistral_paid_api_key,
            )
        )

    @property
    def has_github_token(self) -> bool:
        return not _is_placeholder(self.github_token)

    @property
    def has_turso_configured(self) -> bool:
        return not _is_placeholder(self.turso_url) and not _is_placeholder(self.turso_auth_token)

    def get_app_spec(self, app_id: AppId) -> ManagedAppSpec:
        for spec in self.managed_apps:
            if spec.app_id == app_id:
                return spec
        raise KeyError(f"Unbekannte App-ID: {app_id}")


def _build_app_spec(
    app_id: AppId,
    prefix: str,
    yaml_apps: dict[str, Any],
    *,
    local_secrets: dict[str, Any] | None = None,
) -> ManagedAppSpec:
    """Erstellt `ManagedAppSpec` aus ENV/Secrets mit YAML-Metadaten als Fallback."""
    cfg = yaml_apps.get(app_id.value, {}) if isinstance(yaml_apps.get(app_id.value), dict) else {}
    turso_url = get_secret_or_env(
        f"{prefix}_TURSO_URL", str(cfg.get("turso_url", "")), local_secrets=local_secrets
    )
    turso_key = get_secret_or_env(f"{prefix}_TURSO_KEY", local_secrets=local_secrets)
    sec_wf = cfg.get("secondary_workflows", [])
    secondary = (
        tuple(str(w).strip() for w in sec_wf if str(w).strip())
        if isinstance(sec_wf, list)
        else ()
    )
    exp_tbl = cfg.get("expected_tables", [])
    tables = (
        tuple(str(t).strip() for t in exp_tbl if str(t).strip())
        if isinstance(exp_tbl, list)
        else ()
    )

    return ManagedAppSpec(
        app_id=app_id,
        title=str(cfg.get("title", app_id.value)),
        icon=str(cfg.get("icon", "📦")),
        description=str(cfg.get("description", "")),
        streamlit_url=get_secret_or_env(
            f"{prefix}_URL", str(cfg.get("streamlit_url", "")), local_secrets=local_secrets
        ).strip(),
        github_repo=get_secret_or_env(
            f"{prefix}_REPO", str(cfg.get("github_repo", "")), local_secrets=local_secrets
        ).strip(),
        primary_workflow=get_secret_or_env(
            f"{prefix}_WORKFLOW",
            str(cfg.get("primary_workflow", "")),
            local_secrets=local_secrets,
        ).strip(),
        secondary_workflows=secondary,
        turso_url="" if _is_placeholder(turso_url) else turso_url.strip(),
        turso_auth_token="" if _is_placeholder(turso_key) else turso_key.strip(),
        local_sqlite_path=get_secret_or_env(
            f"{prefix}_SQLITE_PATH",
            str(cfg.get("local_sqlite_path", "")),
            local_secrets=local_secrets,
        ).strip(),
        expected_tables=tables,
    )


def load_app_settings(
    yaml_path: Path | None = None,
    allow_sibling_env_fallback: bool = False,  # noqa: ARG001
) -> AppSettings:
    """Lädt und validiert die Dashboard-Konfiguration ausschließlich aus ENV/Secrets/YAML."""
    data = _load_yaml_defaults(yaml_path)
    apps = data.get("apps", {}) if isinstance(data.get("apps"), dict) else {}
    local_secrets = _load_local_secrets_mapping()

    return AppSettings(
        gemini_api_key=_clean_secret("GEMINI_API_KEY", local_secrets),
        gemini_free_api_key=_clean_secret("GEMINI_FREE_API_KEY", local_secrets),
        gemini_paid_api_key=_clean_secret("GEMINI_PAID_API_KEY", local_secrets),
        browserbase_api_key=_clean_secret("BROWSERBASE_API_KEY", local_secrets),
        browserbase_project_id=get_secret_or_env(
            "BROWSERBASE_PROJECT_ID", local_secrets=local_secrets
        ),
        brave_search_api_key=(
            _clean_secret("BRAVE_SEARCH_API_KEY", local_secrets)
            or _clean_secret("BRAVE_API_KEY", local_secrets)
        ),
        tavily_api_key=_clean_secret("TAVILY_API_KEY", local_secrets),
        serper_api_key=_clean_secret("SERPER_API_KEY", local_secrets),
        reddit_client_id=_clean_secret("REDDIT_CLIENT_ID", local_secrets),
        reddit_client_secret=_clean_secret("REDDIT_CLIENT_SECRET", local_secrets),
        groq_api_key=_clean_secret("GROQ_API_KEY", local_secrets),
        groq_paid_api_key=_clean_secret("GROQ_PAID_API_KEY", local_secrets),
        openrouter_api_key=_clean_secret("OPENROUTER_API_KEY", local_secrets),
        openrouter_paid_api_key=_clean_secret("OPENROUTER_PAID_API_KEY", local_secrets),
        cerebras_api_key=_clean_secret("CEREBRAS_API_KEY", local_secrets),
        cerebras_paid_api_key=_clean_secret("CEREBRAS_PAID_API_KEY", local_secrets),
        mistral_api_key=_clean_secret("MISTRAL_API_KEY", local_secrets),
        mistral_paid_api_key=_clean_secret("MISTRAL_PAID_API_KEY", local_secrets),
        github_token=_clean_secret("GITHUB_TOKEN", local_secrets),
        turso_url=get_secret_or_env("TURSO_URL", local_secrets=local_secrets),
        turso_auth_token=get_secret_or_env("TURSO_KEY", local_secrets=local_secrets),
        sqlite_path=get_secret_or_env(
            "SQLITE_PATH", DEFAULT_SQLITE_REL_PATH, local_secrets=local_secrets
        ),
        brain_turso_url=get_secret_or_env("BRAIN_TURSO_URL", local_secrets=local_secrets),
        brain_turso_auth_token=get_secret_or_env("BRAIN_TURSO_KEY", local_secrets=local_secrets),
        brain_secret_key=_clean_secret("BRAIN_SECRET_KEY", local_secrets),
        brain_sqlite_path=get_secret_or_env(
            "BRAIN_SQLITE_PATH", DEFAULT_BRAIN_SQLITE_REL_PATH, local_secrets=local_secrets
        ),
        app_password=get_secret_or_env("APP_PASSWORD", local_secrets=local_secrets),
        streamlit_app_url=get_secret_or_env("STREAMLIT_APP_URL", local_secrets=local_secrets),
        default_language=str(data.get("default_language", "de")),
        managed_apps=(
            _build_app_spec(AppId.NEWS_BOT, "NEWS_BOT", apps, local_secrets=local_secrets),
            _build_app_spec(AppId.RESEARCH_BOT, "RESEARCH_BOT", apps, local_secrets=local_secrets),
            _build_app_spec(AppId.ALERT_BOT, "ALERT_BOT", apps, local_secrets=local_secrets),
        ),
    )
