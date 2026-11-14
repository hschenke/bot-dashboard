"""Unveränderliche Datenmodelle, Schemata und Secrets für ki_brain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ki_brain.enums import BrainTier, ModelProvider, ProbeStatus, ToolCategory


def _is_placeholder_secret(value: str) -> bool:
    clean = (value or "").strip().lower()
    return not clean or clean.startswith(("your_", "dein_")) or clean == "aizasy..."


@dataclass(frozen=True)
class BrainSecrets:
    """Sicheres API-Key-Container-Objekt (`repr=False` auf allen Geheimnissen)."""

    gemini_api_key: str = field(default="", repr=False)
    browserbase_api_key: str = field(default="", repr=False)
    browserbase_project_id: str = ""
    brave_search_api_key: str = field(default="", repr=False)
    tavily_api_key: str = field(default="", repr=False)
    serper_api_key: str = field(default="", repr=False)
    reddit_client_id: str = field(default="", repr=False)
    reddit_client_secret: str = field(default="", repr=False)
    groq_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    cerebras_api_key: str = field(default="", repr=False)
    mistral_api_key: str = field(default="", repr=False)
    extra_keys: dict[str, str] = field(default_factory=dict, repr=False)

    def resolve_key(self, env_name: str) -> str:
        """Löst einen API-Key anhand des Env-Namens sicher ohne Secrets-Leak auf."""
        clean = (env_name or "").strip().upper()
        if not clean:
            return ""
        mapping = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "BROWSERBASE_API_KEY": self.browserbase_api_key,
            "BRAVE_SEARCH_API_KEY": self.brave_search_api_key or os.getenv("BRAVE_API_KEY", ""),
            "TAVILY_API_KEY": self.tavily_api_key,
            "SERPER_API_KEY": self.serper_api_key,
            "REDDIT_CLIENT_ID": self.reddit_client_id,
            "REDDIT_CLIENT_SECRET": self.reddit_client_secret,
            "GROQ_API_KEY": self.groq_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "CEREBRAS_API_KEY": self.cerebras_api_key,
            "MISTRAL_API_KEY": self.mistral_api_key,
        }
        val = self.extra_keys.get(clean) or mapping.get(clean) or os.getenv(clean, "")
        return "" if _is_placeholder_secret(val) else val.strip()

    def has_key(self, env_name: str) -> bool:
        return bool(self.resolve_key(env_name))

    def with_extra_keys(self, overrides: dict[str, str]) -> BrainSecrets:
        """Erstellt eine neue `BrainSecrets`-Instanz mit zusammengeführten verschlüsselten DB-Keys."""
        merged = {**self.extra_keys, **{k.strip().upper(): v.strip() for k, v in overrides.items() if v.strip()}}
        return BrainSecrets(
            gemini_api_key=self.gemini_api_key,
            browserbase_api_key=self.browserbase_api_key,
            browserbase_project_id=self.browserbase_project_id,
            brave_search_api_key=self.brave_search_api_key,
            tavily_api_key=self.tavily_api_key,
            serper_api_key=self.serper_api_key,
            reddit_client_id=self.reddit_client_id,
            reddit_client_secret=self.reddit_client_secret,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            cerebras_api_key=self.cerebras_api_key,
            mistral_api_key=self.mistral_api_key,
            extra_keys=merged,
        )

    @classmethod
    def from_env(cls) -> BrainSecrets:
        """Lädt bekannte optionale Free-Tier-Keys sicher aus Umgebungsvariablen."""
        brave_val = os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            browserbase_api_key=os.getenv("BROWSERBASE_API_KEY", "").strip(),
            browserbase_project_id=os.getenv("BROWSERBASE_PROJECT_ID", "").strip(),
            brave_search_api_key=brave_val,
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            serper_api_key=os.getenv("SERPER_API_KEY", "").strip(),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", "").strip(),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY", "").strip(),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
        )


class AIModelSpec(BaseModel):
    """Konfigurierbarer Eintrag eines kostenfreien KI-Modells im Brain-Katalog."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=100)
    name: str = Field(min_length=2, max_length=120)
    provider: ModelProvider = ModelProvider.GOOGLE_GEMINI
    model_id: str = Field(min_length=2, max_length=140)
    tier_hint: BrainTier = BrainTier.FAST
    is_free: bool = True
    requires_api_key: bool = True
    api_key_env: str = Field(default="GEMINI_API_KEY", max_length=64)
    base_url: str = Field(default="", max_length=300)
    supports_structured_output: bool = True
    supports_google_search: bool = True
    supports_url_context: bool = True
    is_deep_agent: bool = False
    max_rpm: int = Field(default=15, ge=1, le=240)
    enabled: bool = True
    source: str = Field(default="builtin", max_length=32)
    origin_apps: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=400)
    last_status: ProbeStatus = ProbeStatus.UNTESTED
    last_latency_ms: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    last_tested_at: str = Field(default="", max_length=64)
    last_error: str = Field(default="", max_length=400)


class ToolSpec(BaseModel):
    """Konfigurierbarer Eintrag eines kostenfreien Websearch- oder Extraktions-Tools."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=80)
    name: str = Field(min_length=2, max_length=120)
    category: ToolCategory = ToolCategory.WEB_SEARCH
    executor_type: str = Field(min_length=2, max_length=64)
    description: str = Field(default="", max_length=400)
    endpoint_url: str = Field(default="", max_length=400)
    is_free: bool = True
    requires_api_key: bool = False
    api_key_env: str = Field(default="", max_length=64)
    origin_apps: list[str] = Field(default_factory=list)
    enabled: bool = True
    timeout_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    max_results: int = Field(default=5, ge=1, le=20)
    source: str = Field(default="builtin", max_length=32)
    config: dict[str, Any] = Field(default_factory=dict)
    last_status: ProbeStatus = ProbeStatus.UNTESTED
    last_latency_ms: int = Field(default=0, ge=0)
    last_result_count: int = Field(default=0, ge=0)
    reliability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    last_tested_at: str = Field(default="", max_length=64)
    last_error: str = Field(default="", max_length=400)


class TierSequenceConfig(BaseModel):
    """Optimierte Abfolge von Tools & Modellen für Fast, Standard oder Deep."""

    model_config = ConfigDict(frozen=True)

    tier: BrainTier
    search_tool_sequence: list[str] = Field(default_factory=list)
    extractor_tool_sequence: list[str] = Field(default_factory=list)
    model_sequence: list[str] = Field(default_factory=list)
    max_search_tools: int = Field(default=3, ge=1, le=25)
    enable_page_scraping: bool = False
    max_pages_to_scrape: int = Field(default=0, ge=0, le=10)
    max_followup_rounds: int = Field(default=0, ge=0, le=4)
    single_pass: bool = True
    use_deep_agent: bool = False
    manual_override: bool = False
    optimization_rationale: str = Field(default="", max_length=1200)
    updated_at: str = Field(default="", max_length=64)


class BenchmarkEntry(BaseModel):
    """Einzelergebnis eines Tool- oder Modell-Testlaufs."""

    model_config = ConfigDict(frozen=True)

    item_type: str
    item_id: str
    name: str
    status: ProbeStatus
    latency_ms: int = 0
    result_count: int = 0
    quality_score: float = 0.0
    details: str = ""


class BenchmarkReport(BaseModel):
    """Gesamtprotokoll eines Benchmark-Testlaufs inkl. optimierter Sequenzen."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    started_at: str
    finished_at: str
    test_query: str
    test_url: str
    entries: list[BenchmarkEntry] = Field(default_factory=list)
    sequences: dict[str, TierSequenceConfig] = Field(default_factory=dict)
    healthy_tools_count: int = 0
    healthy_models_count: int = 0


class BrainCitation(BaseModel):
    """Normalisierter Web- oder Wissenstreffer aus einem Brain-Tool."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str = ""
    snippet: str = ""
    source_tool: str = ""
    published_at: str = ""


class BrainFetchedPage(BaseModel):
    """Bereinigter Volltext-Auszug einer geladenen Webseite."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str = ""
    text_content: str = ""
    extractor_used: str = ""
    latency_ms: int = 0


class BrainSearchResult(BaseModel):
    """Ergebnis einer stufenoptimierten Brain-Websuche."""

    model_config = ConfigDict(frozen=True)

    query: str
    tier: BrainTier
    citations: list[BrainCitation] = Field(default_factory=list)
    tools_executed: list[str] = Field(default_factory=list)
    tool_errors: dict[str, str] = Field(default_factory=dict)
    latency_ms: int = 0


class BrainGenerationResult(BaseModel):
    """Ergebnis einer LLM-Text-, Struktur- oder Komplettanfrage im Brain."""

    model_config = ConfigDict(frozen=True)

    text: str
    tier: BrainTier
    model_used: str
    citations: list[BrainCitation] = Field(default_factory=list)
    tools_executed: list[str] = Field(default_factory=list)
    pages_fetched: int = 0
    latency_ms: int = 0
    warnings: list[str] = Field(default_factory=list)
    used_heuristic_fallback: bool = False


class BrainConfigSnapshot(BaseModel):
    """Exportierbarer Gesamtzustand des Brain-Katalogs und der 3 Sequenzen."""

    model_config = ConfigDict(frozen=True)

    version: str = "1.0"
    models: list[AIModelSpec] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    sequences: dict[str, TierSequenceConfig] = Field(default_factory=dict)
    last_benchmark: BenchmarkReport | None = None
    exported_at: str = ""


class SecretStatusEntry(BaseModel):
    """Write-Only Status-Metadaten eines API-Keys (enthält niemals den Klartext-Key!)."""

    model_config = ConfigDict(frozen=True)

    env_name: str
    is_configured: bool
    source: str = ""
    updated_at: str = ""
    affected_items: list[str] = Field(default_factory=list)

