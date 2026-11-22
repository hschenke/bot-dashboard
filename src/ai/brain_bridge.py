"""Brücke zwischen Bot-Dashboard AppSettings und dem eigenständigen ai_brain-Framework."""

from __future__ import annotations

from typing import Any

from ai_brain import AIWebsearchBrain, BrainSecrets, create_brain_repository

from src.config.settings import PROJECT_ROOT, AppSettings


def build_brain_secrets(settings: AppSettings) -> BrainSecrets:
    """Überträgt alle optionalen Free-/Paid-Tier API-Keys sicher (`repr=False`) an das Brain."""
    return BrainSecrets(
        gemini_api_key=settings.gemini_api_key,
        gemini_free_api_key=settings.gemini_free_api_key,
        gemini_paid_api_key=settings.gemini_paid_api_key,
        browserbase_api_key=settings.browserbase_api_key,
        browserbase_project_id=settings.browserbase_project_id,
        brave_search_api_key=settings.brave_search_api_key,
        tavily_api_key=settings.tavily_api_key,
        serper_api_key=settings.serper_api_key,
        reddit_client_id=settings.reddit_client_id,
        reddit_client_secret=settings.reddit_client_secret,
        groq_api_key=settings.groq_api_key,
        groq_paid_api_key=settings.groq_paid_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_paid_api_key=settings.openrouter_paid_api_key,
        cerebras_api_key=settings.cerebras_api_key,
        cerebras_paid_api_key=settings.cerebras_paid_api_key,
        mistral_api_key=settings.mistral_api_key,
        mistral_paid_api_key=settings.mistral_paid_api_key,
    )


def create_dashboard_brain(
    settings: AppSettings,
    *,
    genai_client: Any = None,
) -> AIWebsearchBrain:
    """Erstellt die authentifizierte Dashboard-Admin-Instanz auf Basis von Turso Cloud oder SQLite."""
    use_brain_turso = bool(settings.brain_turso_url.strip() and settings.brain_turso_auth_token.strip())
    use_shared_turso = bool(settings.has_turso_configured)

    turso_url = (
        settings.brain_turso_url
        if use_brain_turso
        else (settings.turso_url if use_shared_turso else "")
    )
    turso_key = (
        settings.brain_turso_auth_token
        if use_brain_turso
        else (settings.turso_auth_token if use_shared_turso else "")
    )
    repo = create_brain_repository(
        turso_url=turso_url,
        turso_auth_token=turso_key,
        sqlite_path=settings.brain_sqlite_path,
        base_dir=PROJECT_ROOT,
    )
    return AIWebsearchBrain.for_dashboard(
        repository=repo,
        secrets=build_brain_secrets(settings),
        genai_client=genai_client,
        vault_passphrase=settings.brain_secret_key or turso_key,
        explicit_vault_key=settings.brain_secret_key,
    )
