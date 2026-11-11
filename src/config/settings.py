"""Sicheres Laden der Dashboard-Konfiguration aus ENV, Streamlit Secrets und YAML."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values, load_dotenv

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

_SIBLING_DIRS: dict[AppId, str] = {
    AppId.NEWS_BOT: "news-aggregator-bot",
    AppId.RESEARCH_BOT: "research-bot",
    AppId.ALERT_BOT: "alert-bot",
}


def _read_streamlit_secret(key: str) -> str | None:
    """Liest optional einen Schlüssel aus `st.secrets`, sofern Streamlit aktiv ist."""
    if "STREAMLIT_SERVER_PORT" not in os.environ and "streamlit" not in sys.modules:
        return None
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            value = st.secrets[key]
            return str(value).strip() if value is not None else None
    except Exception as exc:
        logger.debug("Streamlit Secret '%s' nicht lesbar: %s", key, exc)
    return None


def get_secret_or_env(key: str, default: str = "") -> str:
    """Ermittelt einen Konfigurationswert priorisiert aus ENV, danach `st.secrets`."""
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    secret_val = _read_streamlit_secret(key)
    return secret_val if secret_val else default


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


def _read_sibling_env(app_id: AppId, enabled: bool) -> dict[str, str]:
    """Liest lokal vorhandene Nachbar-`.env`-Dateien als komfortablen Dev-Fallback."""
    if not enabled:
        return {}
    folder_name = _SIBLING_DIRS.get(app_id, "")
    if not folder_name:
        return {}
    workspace_parent = PROJECT_ROOT.parent.resolve()
    candidate = (workspace_parent / folder_name / ".env").resolve()
    if not candidate.is_relative_to(workspace_parent) or not candidate.is_file():
        return {}
    try:
        raw = dotenv_values(candidate)
        return {k: str(v).strip() for k, v in raw.items() if k and v is not None}
    except OSError:
        return {}


def _is_placeholder(val: str) -> bool:
    clean = val.strip()
    if not clean:
        return True
    lower = clean.lower()
    return (
        lower.startswith("your_")
        or lower.startswith("dein_")
        or "deine-datenbank" in lower
        or clean == "AIzaSy..."
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


def _build_app_spec(
    app_id: AppId,
    env_prefix: str,
    yaml_apps: dict[str, Any],
    sibling_env: dict[str, str],
) -> ManagedAppSpec:
    """Erstellt die `ManagedAppSpec` für eine einzelne App aus ENV, YAML und lokalem Fallback."""
    app_yaml = yaml_apps.get(app_id.value, {})
    if not isinstance(app_yaml, dict):
        app_yaml = {}

    url = (
        get_secret_or_env(f"{env_prefix}_URL")
        or sibling_env.get("STREAMLIT_APP_URL", "")
        or str(app_yaml.get("streamlit_url", ""))
    )
    repo = (
        get_secret_or_env(f"{env_prefix}_REPO")
        or sibling_env.get("GITHUB_REPO", "")
        or str(app_yaml.get("github_repo", ""))
    )
    workflow = (
        get_secret_or_env(f"{env_prefix}_WORKFLOW")
        or sibling_env.get("GITHUB_WORKFLOW_FILE", "")
        or str(app_yaml.get("primary_workflow", ""))
    )
    turso_url = (
        get_secret_or_env(f"{env_prefix}_TURSO_URL")
        or sibling_env.get("TURSO_URL", "")
        or str(app_yaml.get("turso_url", ""))
    )
    turso_key = get_secret_or_env(f"{env_prefix}_TURSO_KEY") or sibling_env.get("TURSO_KEY", "")
    sqlite_rel = get_secret_or_env(
        f"{env_prefix}_SQLITE_PATH",
        str(app_yaml.get("local_sqlite_path", "")),
    )

    sec_wf = app_yaml.get("secondary_workflows", [])
    secondary = tuple(str(w).strip() for w in sec_wf if str(w).strip()) if isinstance(sec_wf, list) else ()
    exp_tbl = app_yaml.get("expected_tables", [])
    tables = tuple(str(t).strip() for t in exp_tbl if str(t).strip()) if isinstance(exp_tbl, list) else ()

    return ManagedAppSpec(
        app_id=app_id,
        title=str(app_yaml.get("title", app_id.value)),
        icon=str(app_yaml.get("icon", "📦")),
        description=str(app_yaml.get("description", "")),
        streamlit_url=url.strip(),
        github_repo=repo.strip(),
        primary_workflow=workflow.strip(),
        secondary_workflows=secondary,
        turso_url="" if _is_placeholder(turso_url) else turso_url.strip(),
        turso_auth_token="" if _is_placeholder(turso_key) else turso_key.strip(),
        local_sqlite_path=sqlite_rel.strip(),
        expected_tables=tables,
    )


def load_app_settings(
    yaml_path: Path | None = None,
    allow_sibling_env_fallback: bool = True,
) -> AppSettings:
    """Lädt und validiert die vollständige Dashboard-Konfiguration."""
    yaml_data = _load_yaml_defaults(yaml_path)
    yaml_apps = yaml_data.get("apps", {}) if isinstance(yaml_data.get("apps"), dict) else {}
    prompts = yaml_data.get("prompts", {}) if isinstance(yaml_data.get("prompts"), dict) else {}

    sibling_envs = {
        app_id: _read_sibling_env(app_id, allow_sibling_env_fallback) for app_id in AppId
    }

    specs = (
        _build_app_spec(AppId.NEWS_BOT, "NEWS_BOT", yaml_apps, sibling_envs[AppId.NEWS_BOT]),
        _build_app_spec(AppId.RESEARCH_BOT, "RESEARCH_BOT", yaml_apps, sibling_envs[AppId.RESEARCH_BOT]),
        _build_app_spec(AppId.ALERT_BOT, "ALERT_BOT", yaml_apps, sibling_envs[AppId.ALERT_BOT]),
    )

    gemini_key = get_secret_or_env("GEMINI_API_KEY")
    github_token = get_secret_or_env("GITHUB_TOKEN")
    for sib in sibling_envs.values():
        if _is_placeholder(gemini_key) and not _is_placeholder(sib.get("GEMINI_API_KEY", "")):
            gemini_key = sib["GEMINI_API_KEY"]
        if _is_placeholder(github_token) and not _is_placeholder(sib.get("GITHUB_TOKEN", "")):
            github_token = sib["GITHUB_TOKEN"]

    raw_models = yaml_data.get("allowed_models")
    allowed_models = (
        tuple(str(m).strip() for m in raw_models if str(m).strip())
        if isinstance(raw_models, list) and raw_models
        else CANDIDATE_MODELS
    )

    return AppSettings(
        gemini_api_key="" if _is_placeholder(gemini_key) else gemini_key,
        gemini_fast_model=get_secret_or_env("GEMINI_MODEL", DEFAULT_FAST_MODEL),
        gemini_deep_model=get_secret_or_env("GEMINI_DEEP_MODEL", DEFAULT_DEEP_MODEL),
        allowed_models=allowed_models,
        github_token="" if _is_placeholder(github_token) else github_token,
        turso_url=get_secret_or_env("TURSO_URL"),
        turso_auth_token=get_secret_or_env("TURSO_KEY"),
        sqlite_path=get_secret_or_env("SQLITE_PATH", DEFAULT_SQLITE_REL_PATH),
        app_password=get_secret_or_env("APP_PASSWORD"),
        streamlit_app_url=get_secret_or_env("STREAMLIT_APP_URL"),
        default_language=str(yaml_data.get("default_language", "de")),
        fleet_diagnosis_prompt=str(prompts.get("fleet_diagnosis_system_prompt", "")),
        fleet_qa_prompt=str(prompts.get("fleet_qa_system_prompt", "")),
        managed_apps=specs,
    )
