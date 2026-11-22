"""Typisierte Enums für Stufen, AI-Provider, Tool-Kategorien und Prüfstatus."""

from __future__ import annotations

from enum import StrEnum


class BrainTier(StrEnum):
    """Leistungs- und Ausführungsstufen für sämtliche Brain-Anfragen."""

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class ModelProvider(StrEnum):
    """Unterstützte kostenfreie AI-Modell-Provider."""

    GOOGLE_GEMINI = "google_gemini"
    GEMINI_DEEP_AGENT = "gemini_deep_agent"
    OPENAI_COMPATIBLE = "openai_compatible"


class ToolCategory(StrEnum):
    """Fachliche Kategorie eines Websearch- oder Extraktions-Tools."""

    WEB_SEARCH = "web_search"
    NEWS_SEARCH = "news_search"
    COMMUNITY_SEARCH = "community_search"
    ACADEMIC_SEARCH = "academic_search"
    PAGE_EXTRACTOR = "page_extractor"
    AI_GROUNDING = "ai_grounding"
    REALTIME_API = "realtime_api"


class ProbeStatus(StrEnum):
    """Normierter Gesundheits- und Teststatus eines Modells oder Tools."""

    UNTESTED = "untested"
    OK = "ok"
    QUOTA_LIMITED = "quota_limited"
    AUTH_MISSING = "auth_missing"
    BLOCKED = "blocked"
    ERROR = "error"
