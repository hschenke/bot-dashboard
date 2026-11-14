"""Hauptfassade des wiederverwendbaren KI & Websearch Brain Frameworks."""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from ki_brain.benchmark import BrainBenchmarkRunner
from ki_brain.discovery import DiscoverySummary, auto_sync_catalog
from ki_brain.enums import BrainTier
from ki_brain.execution import (
    execute_tiered_ask,
    execute_tiered_fetch_page,
    execute_tiered_search,
    execute_tiered_structured,
    execute_tiered_text,
)
from ki_brain.llm_runner import UnifiedLLMRunner
from ki_brain.models import (
    AIModelSpec,
    BenchmarkReport,
    BrainConfigSnapshot,
    BrainFetchedPage,
    BrainGenerationResult,
    BrainSearchResult,
    BrainSecrets,
    TierSequenceConfig,
    ToolSpec,
)
from ki_brain.optimizer import DEFAULT_TEST_QUERY, DEFAULT_TEST_URL
from ki_brain.rate_limiter import LlmRateLimiter, get_global_rate_limiter
from ki_brain.security import validate_safe_external_url
from ki_brain.storage import BrainRepository, create_brain_repository

ModelT = TypeVar("ModelT", bound=BaseModel)


class KIWebsearchBrain:
    """Zentrale, dynamische Schaltzentrale für kostenfreie KI-Modelle & Websearch-Tools."""

    def __init__(
        self,
        repository: BrainRepository,
        secrets: BrainSecrets | None = None,
        *,
        genai_client: Any = None,
        rate_limiter: LlmRateLimiter | None = None,
    ) -> None:
        self._repo = repository
        self._secrets = secrets or BrainSecrets.from_env()
        self._genai_client = genai_client
        self._rate_limiter = rate_limiter or get_global_rate_limiter()
        self._runner = UnifiedLLMRunner(
            self._secrets, genai_client=genai_client, rate_limiter=self._rate_limiter
        )

    @classmethod
    def from_env(
        cls,
        *,
        turso_url: str = "",
        turso_auth_token: str = "",
        sqlite_path: str | Path = "data/ki_brain.db",
        base_dir: Path | None = None,
        secrets: BrainSecrets | None = None,
    ) -> KIWebsearchBrain:
        """Initialisiert das Brain direkt aus Umgebungsvariablen (Turso Cloud oder SQLite)."""
        resolved_url = turso_url or os.getenv("BRAIN_TURSO_URL") or os.getenv("TURSO_URL", "")
        resolved_tok = turso_auth_token or os.getenv("BRAIN_TURSO_KEY") or os.getenv("TURSO_KEY", "")
        repo = create_brain_repository(
            turso_url=resolved_url,
            turso_auth_token=resolved_tok,
            sqlite_path=sqlite_path,
            base_dir=base_dir,
        )
        return cls(repository=repo, secrets=secrets or BrainSecrets.from_env())

    @property
    def repository(self) -> BrainRepository:
        return self._repo

    @property
    def backend_name(self) -> str:
        return self._repo.backend_name

    # --- Dynamische Verwaltung von Modellen, Tools & Sequenzen ---
    def list_models(self, enabled_only: bool = False) -> list[AIModelSpec]:
        return self._repo.list_models(enabled_only=enabled_only)

    def get_model(self, model_id: str) -> AIModelSpec | None:
        return self._repo.get_model(model_id)

    def add_or_update_model(self, spec: AIModelSpec) -> AIModelSpec:
        if spec.base_url.strip():
            validate_safe_external_url(spec.base_url.strip(), resolve_dns=False)
        return self._repo.upsert_model(spec)

    def set_model_enabled(self, model_id: str, enabled: bool) -> AIModelSpec | None:
        current = self._repo.get_model(model_id)
        return self._repo.upsert_model(current.model_copy(update={"enabled": enabled})) if current else None

    def remove_model(self, model_id: str) -> bool:
        return self._repo.delete_model(model_id)

    def list_tools(self, enabled_only: bool = False) -> list[ToolSpec]:
        return self._repo.list_tools(enabled_only=enabled_only)

    def get_tool(self, tool_id: str) -> ToolSpec | None:
        return self._repo.get_tool(tool_id)

    def add_or_update_tool(self, spec: ToolSpec) -> ToolSpec:
        if spec.endpoint_url.strip():
            validate_safe_external_url(spec.endpoint_url.strip(), resolve_dns=False)
        return self._repo.upsert_tool(spec)

    def set_tool_enabled(self, tool_id: str, enabled: bool) -> ToolSpec | None:
        current = self._repo.get_tool(tool_id)
        return self._repo.upsert_tool(current.model_copy(update={"enabled": enabled})) if current else None

    def remove_tool(self, tool_id: str) -> bool:
        return self._repo.delete_tool(tool_id)

    def list_sequences(self) -> dict[BrainTier, TierSequenceConfig]:
        return self._repo.list_sequences()

    def get_sequence(self, tier: BrainTier | str) -> TierSequenceConfig:
        return self._repo.get_sequence(tier)

    def update_sequence(self, config: TierSequenceConfig) -> TierSequenceConfig:
        return self._repo.upsert_sequence(config)

    def reset_defaults(self) -> None:
        self._repo.reset_to_defaults()

    # --- Auto-Discovery & Testlauf (Fast, Standard, Deep) ---
    def auto_sync(self, manifest_url: str = "") -> DiscoverySummary:
        return auto_sync_catalog(
            self._repo, self._secrets, genai_client=self._genai_client, manifest_url=manifest_url
        )

    def run_benchmark(
        self,
        *,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        force_overwrite_sequences: bool = False,
    ) -> BenchmarkReport:
        runner = BrainBenchmarkRunner(self._repo, self._secrets, self._runner)
        return runner.run_benchmark(
            test_query=test_query,
            test_url=test_url,
            force_overwrite_sequences=force_overwrite_sequences,
        )

    def get_latest_benchmark(self) -> BenchmarkReport | None:
        return self._repo.get_latest_benchmark()

    # --- Einheitliche Ausführungs-API für alle Anfragen ---
    def search(
        self,
        query: str,
        *,
        tier: BrainTier | str = BrainTier.STANDARD,
        language: str = "de",
        max_results: int = 8,
        on_action: Callable[[str], None] | None = None,
    ) -> BrainSearchResult:
        return execute_tiered_search(
            self._repo, self._secrets, query, BrainTier(tier), language=language, max_results=max_results, on_action=on_action
        )

    def fetch_page(self, url: str) -> BrainFetchedPage:
        return execute_tiered_fetch_page(self._repo, self._secrets, url)

    def generate_text(
        self,
        prompt: str,
        *,
        tier: BrainTier | str = BrainTier.FAST,
        system_instruction: str = "",
        on_action: Callable[[str], None] | None = None,
    ) -> BrainGenerationResult:
        return execute_tiered_text(
            self._repo, self._secrets, self._runner, prompt, tier=BrainTier(tier), system_instruction=system_instruction, on_action=on_action
        )

    def generate_structured(
        self,
        prompt: str,
        schema_cls: type[ModelT],
        *,
        tier: BrainTier | str = BrainTier.FAST,
        system_instruction: str = "",
    ) -> tuple[ModelT, str]:
        return execute_tiered_structured(
            self._repo, self._secrets, self._runner, prompt, schema_cls, tier=BrainTier(tier), system_instruction=system_instruction
        )

    def ask(
        self,
        query: str,
        *,
        tier: BrainTier | str = BrainTier.STANDARD,
        language: str = "de",
        system_instruction: str = "",
        on_action: Callable[[str], None] | None = None,
    ) -> BrainGenerationResult:
        return execute_tiered_ask(
            self._repo,
            self._secrets,
            self._runner,
            query,
            tier=BrainTier(tier),
            language=language,
            system_instruction=system_instruction,
            on_action=on_action,
        )

    # --- Export / Import ---
    def export_snapshot(self) -> BrainConfigSnapshot:
        seqs = {t.value: cfg for t, cfg in self._repo.list_sequences().items()}
        return BrainConfigSnapshot(
            models=self._repo.list_models(),
            tools=self._repo.list_tools(),
            sequences=seqs,
            last_benchmark=self._repo.get_latest_benchmark(),
            exported_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        )

    def export_snapshot_json(self) -> str:
        return self.export_snapshot().model_dump_json(indent=2)

    def export_snapshot_yaml(self) -> str:
        payload = self.export_snapshot().model_dump(mode="json")
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def import_snapshot_json(self, raw_json: str) -> None:
        snap = BrainConfigSnapshot.model_validate_json(raw_json)
        for m in snap.models:
            self.add_or_update_model(m)
        for t in snap.tools:
            self.add_or_update_tool(t)
        for seq in snap.sequences.values():
            self.update_sequence(seq)

    def close(self) -> None:
        self._repo.close()

    def __enter__(self) -> KIWebsearchBrain:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
