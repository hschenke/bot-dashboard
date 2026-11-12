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
    CANDIDATE_MODELS,
    DEFAULT_DEEP_MODEL,
    DEFAULT_FAST_MODEL,
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


def _read_local_secrets_toml(key: str) -> str | None:
    """Liest Schlüssel direkt aus .streamlit/secrets.toml (für CLI, Worker & Tests)."""
    target = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not target.is_file():
        return None
    try:
        data = tomllib.loads(target.read_text(encoding="utf-8"))
        return _lookup_in_mapping(data, key)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.debug("Fehler beim Lesen von %s: %s", target, exc)
        return None


def _read_streamlit_secret(key: str) -> str | None:
    """Liest optional einen Schlüssel aus `st.secrets` (flach oder verschachtelt)."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            return _lookup_in_mapping(st.secrets, key)
    except Exception as exc:
        logger.debug("Streamlit Secret '%s' nicht lesbar: %s", key, exc)
    return None


def get_secret_or_env(key: str, default: str = "") -> str:
    """Ermittelt Konfigurationswert priorisiert aus ENV, `st.secrets` & `secrets.toml`."""
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    return _read_streamlit_secret(key) or _read_local_secrets_toml(key) or default


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


@dataclass(frozen=True)
class AppSettings:
    """Zentrales Konfigurationsobjekt (`repr=False` auf allen Geheimnissen)."""

    gemini_api_key: str = field(default="", repr=False)
    gemini_fast_model: str = DEFAULT_FAST_MODEL
    gemini_deep_model: str = DEFAULT_DEEP_MODEL
    allowed_models: tuple[str, ...] = CANDIDATE_MODELS

    github_token: str = field(default="", repr=False)
    turso_url: str = ""
    turso_auth_token: str = field(default="", repr=False)
    sqlite_path: str = DEFAULT_SQLITE_REL_PATH

    app_password: str = field(default="", repr=False)
    streamlit_app_url: str = ""
    default_language: str = "de"
    fleet_diagnosis_prompt: str = ""
    fleet_qa_prompt: str = ""
    managed_apps: tuple[ManagedAppSpec, ...] = ()

    @property
    def has_gemini_configured(self) -> bool:
        return not _is_placeholder(self.gemini_api_key)

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


def _build_app_spec(app_id: AppId, prefix: str, yaml_apps: dict[str, Any]) -> ManagedAppSpec:
    """Erstellt `ManagedAppSpec` aus ENV/Secrets mit YAML-Metadaten als Fallback."""
    cfg = yaml_apps.get(app_id.value, {}) if isinstance(yaml_apps.get(app_id.value), dict) else {}
    turso_url = get_secret_or_env(f"{prefix}_TURSO_URL", str(cfg.get("turso_url", "")))
    turso_key = get_secret_or_env(f"{prefix}_TURSO_KEY")
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
        streamlit_url=get_secret_or_env(f"{prefix}_URL", str(cfg.get("streamlit_url", ""))).strip(),
        github_repo=get_secret_or_env(f"{prefix}_REPO", str(cfg.get("github_repo", ""))).strip(),
        primary_workflow=get_secret_or_env(
            f"{prefix}_WORKFLOW", str(cfg.get("primary_workflow", ""))
        ).strip(),
        secondary_workflows=secondary,
        turso_url="" if _is_placeholder(turso_url) else turso_url.strip(),
        turso_auth_token="" if _is_placeholder(turso_key) else turso_key.strip(),
        local_sqlite_path=get_secret_or_env(
            f"{prefix}_SQLITE_PATH", str(cfg.get("local_sqlite_path", ""))
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
    prompts = data.get("prompts", {}) if isinstance(data.get("prompts"), dict) else {}
    raw_models = data.get("allowed_models")
    models = (
        tuple(str(m).strip() for m in raw_models if str(m).strip())
        if isinstance(raw_models, list) and raw_models
        else CANDIDATE_MODELS
    )

    return AppSettings(
        gemini_api_key="" if _is_placeholder(get_secret_or_env("GEMINI_API_KEY")) else get_secret_or_env("GEMINI_API_KEY"),
        gemini_fast_model=get_secret_or_env("GEMINI_MODEL", DEFAULT_FAST_MODEL),
        gemini_deep_model=get_secret_or_env("GEMINI_DEEP_MODEL", DEFAULT_DEEP_MODEL),
        allowed_models=models,
        github_token="" if _is_placeholder(get_secret_or_env("GITHUB_TOKEN")) else get_secret_or_env("GITHUB_TOKEN"),
        turso_url=get_secret_or_env("TURSO_URL"),
        turso_auth_token=get_secret_or_env("TURSO_KEY"),
        sqlite_path=get_secret_or_env("SQLITE_PATH", DEFAULT_SQLITE_REL_PATH),
        app_password=get_secret_or_env("APP_PASSWORD"),
        streamlit_app_url=get_secret_or_env("STREAMLIT_APP_URL"),
        default_language=str(data.get("default_language", "de")),
        fleet_diagnosis_prompt=str(prompts.get("fleet_diagnosis_system_prompt", "")),
        fleet_qa_prompt=str(prompts.get("fleet_qa_system_prompt", "")),
        managed_apps=(
            _build_app_spec(AppId.NEWS_BOT, "NEWS_BOT", apps),
            _build_app_spec(AppId.RESEARCH_BOT, "RESEARCH_BOT", apps),
            _build_app_spec(AppId.ALERT_BOT, "ALERT_BOT", apps),
        ),
    )
