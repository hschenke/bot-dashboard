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
from ki_brain.enums import BrainTier, ProbeStatus
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
    BenchmarkEntry,
    BenchmarkReport,
    BrainConfigSnapshot,
    BrainFetchedPage,
    BrainGenerationResult,
    BrainSearchResult,
    BrainSecrets,
    SecretStatusEntry,
    TierSequenceConfig,
    ToolSpec,
)
from ki_brain.optimizer import DEFAULT_TEST_QUERY, DEFAULT_TEST_URL, optimize_tier_sequences
from ki_brain.rate_limiter import LlmRateLimiter, get_global_rate_limiter
from ki_brain.secrets_vault import encrypt_secret_value, validate_secret_env_name
from ki_brain.security import validate_safe_external_url
from ki_brain.storage import BrainRepository, create_brain_repository
from ki_brain.unblock import UnblockSummary
from ki_brain.unblock import resolve_blocked_tools as _resolve_blocked_tools_impl

ModelT = TypeVar("ModelT", bound=BaseModel)

_DEFAULT_SECRET_NAMES: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
    "MISTRAL_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "BROWSERBASE_API_KEY",
)


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
        self._base_secrets = secrets or BrainSecrets.from_env()
        self._secrets = self._base_secrets.with_extra_keys(self._repo.load_decrypted_secrets())
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
        saved = self._repo.upsert_model(spec)
        self._repo.remove_excluded_id(model_id=saved.id)
        return saved

    def set_model_enabled(self, model_id: str, enabled: bool) -> AIModelSpec | None:
        current = self._repo.get_model(model_id)
        return self._repo.upsert_model(current.model_copy(update={"enabled": enabled})) if current else None

    def _prune_ids_from_sequences(
        self, *, model_ids: set[str] | None = None, tool_ids: set[str] | None = None
    ) -> None:
        drop_m = model_ids or set()
        drop_t = tool_ids or set()
        if not drop_m and not drop_t:
            return
        for seq in self._repo.list_sequences().values():
            new_s = [tid for tid in seq.search_tool_sequence if tid not in drop_t]
            new_e = [tid for tid in seq.extractor_tool_sequence if tid not in drop_t]
            new_m = [mid for mid in seq.model_sequence if mid not in drop_m]
            if (
                new_s != seq.search_tool_sequence
                or new_e != seq.extractor_tool_sequence
                or new_m != seq.model_sequence
            ):
                self._repo.upsert_sequence(
                    seq.model_copy(
                        update={
                            "search_tool_sequence": new_s,
                            "extractor_tool_sequence": new_e,
                            "model_sequence": new_m,
                        }
                    )
                )

    def remove_model(self, model_id: str) -> bool:
        removed = self._repo.delete_model(model_id)
        if removed:
            self._repo.add_excluded_ids(model_ids=[model_id])
            self._prune_ids_from_sequences(model_ids={model_id.strip().lower()})
        return removed

    def list_tools(self, enabled_only: bool = False) -> list[ToolSpec]:
        return self._repo.list_tools(enabled_only=enabled_only)

    def get_tool(self, tool_id: str) -> ToolSpec | None:
        return self._repo.get_tool(tool_id)

    def add_or_update_tool(self, spec: ToolSpec) -> ToolSpec:
        if spec.endpoint_url.strip():
            validate_safe_external_url(spec.endpoint_url.strip(), resolve_dns=False)
        saved = self._repo.upsert_tool(spec)
        self._repo.remove_excluded_id(tool_id=saved.id)
        return saved

    def set_tool_enabled(self, tool_id: str, enabled: bool) -> ToolSpec | None:
        current = self._repo.get_tool(tool_id)
        return self._repo.upsert_tool(current.model_copy(update={"enabled": enabled})) if current else None

    def remove_tool(self, tool_id: str) -> bool:
        removed = self._repo.delete_tool(tool_id)
        if removed:
            self._repo.add_excluded_ids(tool_ids=[tool_id])
            self._prune_ids_from_sequences(tool_ids={tool_id.strip().lower()})
        return removed

    def list_sequences(self) -> dict[BrainTier, TierSequenceConfig]:
        return self._repo.list_sequences()

    def get_sequence(self, tier: BrainTier | str) -> TierSequenceConfig:
        return self._repo.get_sequence(tier)

    def update_sequence(self, config: TierSequenceConfig) -> TierSequenceConfig:
        return self._repo.upsert_sequence(config)

    def reset_defaults(self) -> None:
        self._repo.reset_to_defaults()

    def cleanup_unhealthy_items_with_status(
        self,
        statuses: tuple[ProbeStatus, ...] = (
            ProbeStatus.ERROR,
            ProbeStatus.QUOTA_LIMITED,
        ),
    ) -> tuple[list[tuple[str, ProbeStatus]], list[tuple[str, ProbeStatus]]]:
        """Entfernt Modelle & Tools mit `Error` oder `Quota 429` dauerhaft inkl. Status-Protokoll; `Blocked` bleibt erhalten."""
        bad = set(statuses)
        removed_models = [
            (m.id, m.last_status)
            for m in self._repo.list_models()
            if m.last_status in bad and self._repo.delete_model(m.id)
        ]
        removed_tools = [
            (t.id, t.last_status)
            for t in self._repo.list_tools()
            if t.last_status in bad and self._repo.delete_tool(t.id)
        ]
        if removed_models or removed_tools:
            rem_m_ids = [mid for mid, _ in removed_models]
            rem_t_ids = [tid for tid, _ in removed_tools]
            self._repo.add_excluded_ids(model_ids=rem_m_ids, tool_ids=rem_t_ids)
            self._prune_ids_from_sequences(model_ids=set(rem_m_ids), tool_ids=set(rem_t_ids))
            seqs = optimize_tier_sequences(
                self._repo.list_models(),
                self._repo.list_tools(),
                self._repo.list_sequences(),
                force_overwrite=False,
            )
            for seq in seqs.values():
                self._repo.upsert_sequence(seq)
        return removed_models, removed_tools

    def cleanup_unhealthy_items(
        self,
        statuses: tuple[ProbeStatus, ...] = (
            ProbeStatus.ERROR,
            ProbeStatus.QUOTA_LIMITED,
        ),
    ) -> tuple[list[str], list[str]]:
        """Entfernt Modelle & Tools mit `Error` oder `Quota 429` dauerhaft; `Blocked` bleibt zur Analyse erhalten."""
        rem_m, rem_t = self.cleanup_unhealthy_items_with_status(statuses=statuses)
        return [mid for mid, _ in rem_m], [tid for tid, _ in rem_t]

    def remove_preview_models(self) -> list[str]:
        """Entfernt veraltete Preview-/Experimental-Modelle (außer Deep Research) dauerhaft aus Katalog & Sequenzen."""
        from ki_brain.discovery import is_preview_or_experimental_model

        removed: list[str] = []
        for m in self._repo.list_models():
            if not m.is_deep_agent and is_preview_or_experimental_model(m.model_id, m.name) and self._repo.delete_model(m.id):
                removed.append(m.id)
        if removed:
            self._repo.add_excluded_ids(model_ids=removed)
            self._prune_ids_from_sequences(model_ids=set(removed))
            seqs = optimize_tier_sequences(
                self._repo.list_models(),
                self._repo.list_tools(),
                self._repo.list_sequences(),
                force_overwrite=False,
            )
            for seq in seqs.values():
                self._repo.upsert_sequence(seq)
        return removed

    def resolve_blocked_tools(
        self,
        *,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> UnblockSummary:
        """Wendet Rules, searx.space-Pool & Endpoint-Fallbacks auf `Blocked`-Tools an und testet live nach."""
        return _resolve_blocked_tools_impl(
            self._repo,
            self._secrets,
            self._runner,
            test_query=test_query,
            test_url=test_url,
            on_progress=on_progress,
        )

    # --- Write-Only Secret-Management (wie GitHub Secrets) ---
    def _refresh_runtime_secrets(self) -> None:
        self._secrets = self._base_secrets.with_extra_keys(self._repo.load_decrypted_secrets())
        self._runner = UnifiedLLMRunner(
            self._secrets, genai_client=self._genai_client, rate_limiter=self._rate_limiter
        )

    def set_secret(self, env_name: str, new_secret_value: str) -> str:
        """Speichert oder überschreibt ein Secret komplett verschlüsselt (Write-Only) und entsperrt Missing-Key-Einträge."""
        clean_name = validate_secret_env_name(env_name)
        encrypted = encrypt_secret_value(new_secret_value)
        updated_at = self._repo.upsert_encrypted_secret(clean_name, encrypted)
        self._refresh_runtime_secrets()
        for m in self._repo.list_models():
            if m.api_key_env.upper() == clean_name and m.last_status == ProbeStatus.AUTH_MISSING:
                self._repo.upsert_model(m.model_copy(update={"last_status": ProbeStatus.UNTESTED, "last_error": ""}))
        for t in self._repo.list_tools():
            if t.api_key_env.upper() == clean_name and t.last_status == ProbeStatus.AUTH_MISSING:
                self._repo.upsert_tool(t.model_copy(update={"last_status": ProbeStatus.UNTESTED, "last_error": ""}))
        return updated_at

    def delete_secret(self, env_name: str) -> bool:
        """Entfernt ein im Brain-Vault gespeichertes Secret."""
        removed = self._repo.delete_encrypted_secret(env_name)
        self._refresh_runtime_secrets()
        return removed

    def probe_items_for_secret(self, env_name: str) -> list[BenchmarkEntry]:
        """Testet alle zu einem Secret gehörenden Modelle/Tools live und aktualisiert Tier-Sequenzen bei Erfolg."""
        clean_name = validate_secret_env_name(env_name)
        self._refresh_runtime_secrets()
        bench = BrainBenchmarkRunner(self._repo, self._secrets, self._runner)
        entries: list[BenchmarkEntry] = []

        target_models = [m for m in self._repo.list_models(enabled_only=True) if m.api_key_env.strip().upper() == clean_name]
        for model in target_models:
            um, entry = bench.probe_model(model)
            self._repo.upsert_model(um)
            entries.append(entry)

        all_models = self._repo.list_models(enabled_only=True)
        grounding_model = next(
            (m for m in all_models if not m.is_deep_agent and m.provider.value == "google_gemini" and m.last_status == ProbeStatus.OK),
            next((m for m in all_models if not m.is_deep_agent and m.provider.value == "google_gemini"), None),
        )
        target_tools = [t for t in self._repo.list_tools(enabled_only=True) if t.api_key_env.strip().upper() == clean_name]
        for tool in target_tools:
            ut, entry = bench.probe_tool(tool, DEFAULT_TEST_QUERY, DEFAULT_TEST_URL, grounding_model)
            self._repo.upsert_tool(ut)
            entries.append(entry)

        updated_seqs = self._repo.list_sequences()
        if any(e.status == ProbeStatus.OK for e in entries):
            seqs = optimize_tier_sequences(
                self._repo.list_models(),
                self._repo.list_tools(),
                updated_seqs,
                force_overwrite=False,
            )
            updated_seqs = {t: self._repo.upsert_sequence(seq) for t, seq in seqs.items()}

        latest = self._repo.get_latest_benchmark()
        if latest is not None and entries:
            by_key = {(e.item_type, e.item_id): e for e in latest.entries}
            for new_entry in entries:
                by_key[(new_entry.item_type, new_entry.item_id)] = new_entry
            merged = list(by_key.values())
            refreshed = latest.model_copy(
                update={
                    "entries": merged,
                    "sequences": {t.value: cfg for t, cfg in updated_seqs.items()},
                    "healthy_tools_count": sum(1 for e in merged if e.item_type == "tool" and e.status == ProbeStatus.OK),
                    "healthy_models_count": sum(1 for e in merged if e.item_type == "model" and e.status == ProbeStatus.OK),
                }
            )
            self._repo.save_benchmark_run(refreshed)
        return entries

    def list_secret_statuses(
        self,
        *,
        models: list[AIModelSpec] | None = None,
        tools: list[ToolSpec] | None = None,
        secret_timestamps: dict[str, str] | None = None,
    ) -> list[SecretStatusEntry]:
        """Liefert Write-Only-Metadaten aller relevanten API-Keys (NIEMALS den Klartext-Wert!)."""
        vault_ts = secret_timestamps if secret_timestamps is not None else self._repo.list_secret_timestamps()
        models_list = models if models is not None else self._repo.list_models()
        tools_list = tools if tools is not None else self._repo.list_tools()
        all_names = list(_DEFAULT_SECRET_NAMES)
        for item_env in [*(m.api_key_env for m in models_list), *(t.api_key_env for t in tools_list), *vault_ts.keys()]:
            clean = item_env.strip().upper()
            if clean and clean not in all_names:
                all_names.append(clean)
        entries: list[SecretStatusEntry] = []
        for name in all_names:
            in_vault = name in vault_ts
            in_env = self._base_secrets.has_key(name)
            source = "Brain Vault (verschlüsselt)" if in_vault else ("ENV / Streamlit Secrets" if in_env else "Nicht gesetzt")
            affected = [
                *[f"Model: {m.id}" for m in models_list if m.api_key_env.strip().upper() == name],
                *[f"Tool: {t.id}" for t in tools_list if t.api_key_env.strip().upper() == name],
            ]
            entries.append(
                SecretStatusEntry(
                    env_name=name,
                    is_configured=(in_vault or in_env),
                    source=source,
                    updated_at=vault_ts.get(name, ""),
                    affected_items=affected,
                )
            )
        return entries

    def load_ui_state(self) -> tuple[Any, list[SecretStatusEntry]]:
        """Lädt den gesamten Dashboard-Zustand in einem DB-Batch plus abgeleitete Secret-Metadaten."""
        state = self._repo.load_dashboard_state()
        statuses = self.list_secret_statuses(
            models=state.models,
            tools=state.tools,
            secret_timestamps=state.secret_timestamps,
        )
        return state, statuses

    # --- Auto-Discovery & Testlauf (Fast, Standard, Deep) ---
    def auto_sync(
        self,
        manifest_url: str = "",
        *,
        verify_discovered: bool = True,
        exclude_previews: bool = True,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> DiscoverySummary:
        return auto_sync_catalog(
            self._repo,
            self._secrets,
            genai_client=self._genai_client,
            manifest_url=manifest_url,
            runner=self._runner,
            verify_discovered=verify_discovered,
            exclude_previews=exclude_previews,
            on_progress=on_progress,
        )

    def run_benchmark(
        self,
        *,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        force_overwrite_sequences: bool = False,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> BenchmarkReport:
        runner = BrainBenchmarkRunner(self._repo, self._secrets, self._runner)
        return runner.run_benchmark(
            test_query=test_query,
            test_url=test_url,
            force_overwrite_sequences=force_overwrite_sequences,
            on_progress=on_progress,
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
