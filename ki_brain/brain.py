"""Hauptfassade des wiederverwendbaren KI & Websearch Brain Frameworks."""

from __future__ import annotations

import datetime
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from ki_brain.access import (
    INTERNAL_DASHBOARD_APP,
    AppClientStatus,
    authorize_client_app,
    build_app_client_statuses,
    generate_app_access_token,
)
from ki_brain.benchmark import BrainBenchmarkRunner
from ki_brain.discovery import DiscoverySummary, auto_sync_catalog
from ki_brain.enums import BrainTier, ProbeStatus
from ki_brain.exceptions import BrainAuthError
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
    derive_tier_key_envs,
)
from ki_brain.optimizer import DEFAULT_TEST_QUERY, DEFAULT_TEST_URL, optimize_tier_sequences
from ki_brain.rate_limiter import LlmRateLimiter, get_global_rate_limiter
from ki_brain.secrets_vault import encrypt_secret_value, validate_secret_env_name
from ki_brain.security import validate_safe_external_url
from ki_brain.storage import BrainRepository, create_brain_repository
from ki_brain.unblock import UnblockSummary
from ki_brain.unblock import resolve_blocked_tools as _resolve_blocked_tools_impl

ModelT = TypeVar("ModelT", bound=BaseModel)

_DEFAULT_AI_SECRET_SPECS: tuple[tuple[str, str], ...] = (
    ("GEMINI_FREE_API_KEY", "free"),
    ("GEMINI_PAID_API_KEY", "paid"),
    ("GEMINI_API_KEY", "legacy"),
    ("GROQ_API_KEY", "free"),
    ("GROQ_PAID_API_KEY", "paid"),
    ("OPENROUTER_API_KEY", "free"),
    ("OPENROUTER_PAID_API_KEY", "paid"),
    ("CEREBRAS_API_KEY", "free"),
    ("MISTRAL_API_KEY", "free"),
)
_DEFAULT_TOOL_SECRET_NAMES: tuple[str, ...] = (
    "BRAVE_SEARCH_API_KEY",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "BROWSERBASE_API_KEY",
)
_DEFAULT_SECRET_NAMES: tuple[str, ...] = (
    *(name for name, _ in _DEFAULT_AI_SECRET_SPECS),
    *_DEFAULT_TOOL_SECRET_NAMES,
)
_AI_SECRET_TIERS: dict[str, str] = dict(_DEFAULT_AI_SECRET_SPECS)
_HTTP_202_RE = re.compile(r"(?:(?:http|status)\s*202\b|\b202\s*\(anti-bot)", re.IGNORECASE)


@dataclass(frozen=True)
class PipelineSummary:
    """Konsolidiertes Ergebnis des vereinten KI-Brain-Ablaufs (Discover → Benchmark → Unblock → Prune)."""

    executed_steps: tuple[str, ...] = ()
    discovery: DiscoverySummary | None = None
    benchmark: BenchmarkReport | None = None
    unblock: UnblockSummary | None = None
    pruned_models: tuple[tuple[str, ProbeStatus], ...] = ()
    pruned_tools: tuple[tuple[str, ProbeStatus], ...] = ()
    warnings: tuple[str, ...] = ()


def _should_clean_item(status: ProbeStatus, last_error: str, bad: set[ProbeStatus], include_http_202: bool) -> bool:
    if status in bad:
        return True
    return include_http_202 and status == ProbeStatus.BLOCKED and bool(_HTTP_202_RE.search(last_error or ""))


def _model_matches_secret_env(model: AIModelSpec, clean_env: str) -> bool:
    if not model.requires_api_key:
        return False
    free_envs, paid_envs = derive_tier_key_envs(model.api_key_env, model.paid_api_key_env)
    return clean_env in free_envs or clean_env in paid_envs


def _tool_matches_secret_env(tool: ToolSpec, clean_env: str) -> bool:
    if not tool.requires_api_key:
        return False
    t_env = tool.api_key_env.strip().upper()
    if t_env == clean_env:
        return True
    return t_env == "GEMINI_API_KEY" and clean_env in {"GEMINI_FREE_API_KEY", "GEMINI_PAID_API_KEY"}


def _classify_secret_env(
    env_name: str, models_list: list[AIModelSpec], tools_list: list[ToolSpec]
) -> tuple[str, str]:
    if env_name in _AI_SECRET_TIERS:
        return "ai", _AI_SECRET_TIERS[env_name]
    if env_name in _DEFAULT_TOOL_SECRET_NAMES:
        return "tool", "tool"
    if any(_model_matches_secret_env(m, env_name) for m in models_list):
        tier = "paid" if env_name.endswith("_PAID_API_KEY") else "free"
        return "ai", tier
    if any(_tool_matches_secret_env(t, env_name) for t in tools_list):
        return "tool", "tool"
    if env_name.endswith("_PAID_API_KEY"):
        return "ai", "paid"
    if env_name.endswith("_FREE_API_KEY"):
        return "ai", "free"
    return "tool", "tool"


class KIWebsearchBrain:
    """Zentrale, dynamische Schaltzentrale für kostenfreie KI-Modelle & Websearch-Tools."""

    def __init__(
        self,
        repository: BrainRepository,
        secrets: BrainSecrets | None = None,
        *,
        app_id: str = "",
        app_token: str = "",
        genai_client: Any = None,
        rate_limiter: LlmRateLimiter | None = None,
        _internal_admin: bool = False,
    ) -> None:
        self._repo = repository
        if _internal_admin:
            self._client_app_id = INTERNAL_DASHBOARD_APP
            self._is_admin_client = True
        else:
            self._client_app_id = authorize_client_app(repository, app_id, app_token)
            self._is_admin_client = False
        self._base_secrets = secrets or BrainSecrets.from_env()
        self._secrets = self._base_secrets.with_extra_keys(self._repo.load_decrypted_secrets())
        self._genai_client = genai_client
        self._rate_limiter = rate_limiter or get_global_rate_limiter()
        self._runner = UnifiedLLMRunner(
            self._secrets, genai_client=genai_client, rate_limiter=self._rate_limiter
        )

    @classmethod
    def for_dashboard(
        cls,
        repository: BrainRepository,
        secrets: BrainSecrets | None = None,
        *,
        genai_client: Any = None,
        rate_limiter: LlmRateLimiter | None = None,
    ) -> KIWebsearchBrain:
        """Erstellt die interne Admin-Instanz für das authentifizierte Holli Dashboard."""
        return cls(
            repository=repository,
            secrets=secrets,
            genai_client=genai_client,
            rate_limiter=rate_limiter,
            _internal_admin=True,
        )

    @classmethod
    def from_env(
        cls,
        *,
        app_id: str = "",
        app_token: str = "",
        turso_url: str = "",
        turso_auth_token: str = "",
        sqlite_path: str | Path = "data/ki_brain.db",
        base_dir: Path | None = None,
        secrets: BrainSecrets | None = None,
    ) -> KIWebsearchBrain:
        """Initialisiert das Brain für eine autorisierte App (`news_bot`, `research_bot`, `alert_bot`)."""
        resolved_app_id = app_id.strip() or os.getenv("BRAIN_APP_ID", "").strip()
        resolved_app_token = app_token.strip() or os.getenv("BRAIN_APP_TOKEN", "").strip()
        resolved_url = turso_url or os.getenv("BRAIN_TURSO_URL") or os.getenv("TURSO_URL", "")
        resolved_tok = turso_auth_token or os.getenv("BRAIN_TURSO_KEY") or os.getenv("TURSO_KEY", "")
        repo = create_brain_repository(
            turso_url=resolved_url,
            turso_auth_token=resolved_tok,
            sqlite_path=sqlite_path,
            base_dir=base_dir,
        )
        try:
            return cls(
                repository=repo,
                secrets=secrets or BrainSecrets.from_env(),
                app_id=resolved_app_id,
                app_token=resolved_app_token,
            )
        except Exception:
            repo.close()
            raise

    @property
    def repository(self) -> BrainRepository:
        return self._repo

    @property
    def backend_name(self) -> str:
        return self._repo.backend_name

    @property
    def client_app_id(self) -> str:
        return self._client_app_id

    @property
    def is_admin_client(self) -> bool:
        return self._is_admin_client

    def _require_admin(self) -> None:
        if not self._is_admin_client:
            raise BrainAuthError(
                f"Zugriff verweigert: App '{self._client_app_id}' besitzt keine Admin-Rechte für diese Vault-Aktion."
            )

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
        from ki_brain.optimizer import _format_rationale

        drop_m = model_ids or set()
        drop_t = tool_ids or set()
        if not drop_m and not drop_t:
            return
        all_models = self._repo.list_models()
        all_tools = self._repo.list_tools()
        deep_ids = {m.id for m in all_models if m.is_deep_agent}
        usage_state = self._repo.load_usage_quotas()
        for seq in self._repo.list_sequences().values():
            new_s = [tid for tid in seq.search_tool_sequence if tid not in drop_t]
            new_e = [tid for tid in seq.extractor_tool_sequence if tid not in drop_t]
            new_m = [mid for mid in seq.model_sequence if mid not in drop_m]
            new_ps = [tid for tid in seq.paid_search_tool_sequence if tid not in drop_t]
            new_pe = [tid for tid in seq.paid_extractor_tool_sequence if tid not in drop_t]
            new_pm = [mid for mid in seq.paid_model_sequence if mid not in drop_m]
            new_use_deep = seq.tier == BrainTier.DEEP and any(mid in deep_ids for mid in new_m)
            new_paid_deep = seq.tier == BrainTier.DEEP and any(mid in deep_ids for mid in new_pm)
            rationale_stale = any(
                dropped in (seq.optimization_rationale or "") for dropped in (*drop_m, *drop_t)
            )
            if (
                new_s != seq.search_tool_sequence
                or new_e != seq.extractor_tool_sequence
                or new_m != seq.model_sequence
                or new_ps != seq.paid_search_tool_sequence
                or new_pe != seq.paid_extractor_tool_sequence
                or new_pm != seq.paid_model_sequence
                or new_use_deep != seq.use_deep_agent
                or new_paid_deep != seq.paid_use_deep_agent
                or rationale_stale
            ):
                new_rationale = (
                    seq.optimization_rationale
                    if seq.manual_override and not rationale_stale
                    else _format_rationale(seq.tier, new_s, new_m, all_tools, all_models, usage_state)
                )
                self._repo.upsert_sequence(
                    seq.model_copy(
                        update={
                            "search_tool_sequence": new_s,
                            "extractor_tool_sequence": new_e,
                            "model_sequence": new_m,
                            "paid_search_tool_sequence": new_ps,
                            "paid_extractor_tool_sequence": new_pe,
                            "paid_model_sequence": new_pm,
                            "use_deep_agent": new_use_deep,
                            "paid_use_deep_agent": new_paid_deep,
                            "optimization_rationale": new_rationale,
                        }
                    )
                )

    def remove_model(self, model_id: str) -> bool:
        removed = self._repo.delete_model(model_id)
        if removed:
            self._repo.add_excluded_ids(model_ids=[model_id], reason="manual")
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
            self._repo.add_excluded_ids(tool_ids=[tool_id], reason="manual")
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

    def load_usage_quotas(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Liefert den periodenbasierten Free-Kontingent-Verbrauch (Monat/Tag/Lifetime)."""
        return self._repo.load_usage_quotas()

    def reset_usage_quotas(self) -> None:
        """Setzt gespeicherte Verbrauchs-Zähler der Free-Kontingente zurück."""
        self._repo.reset_usage_quotas()

    def cleanup_unhealthy_items_with_status(
        self,
        statuses: tuple[ProbeStatus, ...] = (
            ProbeStatus.ERROR,
            ProbeStatus.QUOTA_LIMITED,
        ),
        *,
        include_http_202: bool = True,
    ) -> tuple[list[tuple[str, ProbeStatus]], list[tuple[str, ProbeStatus]]]:
        """Entfernt Modelle & Tools mit `Error`, `Quota 429` oder `HTTP 202` dauerhaft; übrige `Blocked` bleiben erhalten."""
        bad = set(statuses)
        removed_models = [
            (m.id, m.last_status)
            for m in self._repo.list_models()
            if _should_clean_item(m.last_status, m.last_error, bad, include_http_202) and self._repo.delete_model(m.id)
        ]
        removed_tools = [
            (t.id, t.last_status)
            for t in self._repo.list_tools()
            if _should_clean_item(t.last_status, t.last_error, bad, include_http_202) and self._repo.delete_tool(t.id)
        ]
        if removed_models or removed_tools:
            rem_m_ids = [mid for mid, _ in removed_models]
            rem_t_ids = [tid for tid, _ in removed_tools]
            model_reasons = {
                mid: ("quota_limited" if st == ProbeStatus.QUOTA_LIMITED else "error")
                for mid, st in removed_models
            }
            tool_reasons = {
                tid: ("quota_limited" if st == ProbeStatus.QUOTA_LIMITED else "error")
                for tid, st in removed_tools
            }
            self._repo.add_excluded_ids(
                model_ids=rem_m_ids,
                tool_ids=rem_t_ids,
                model_reasons=model_reasons,
                tool_reasons=tool_reasons,
            )
            self._prune_ids_from_sequences(model_ids=set(rem_m_ids), tool_ids=set(rem_t_ids))
            seqs = optimize_tier_sequences(
                self._repo.list_models(),
                self._repo.list_tools(),
                self._repo.list_sequences(),
                force_overwrite=False,
                usage_state=self._repo.load_usage_quotas(),
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
        *,
        include_http_202: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Entfernt Modelle & Tools mit `Error`, `Quota 429` oder `HTTP 202` dauerhaft; übrige `Blocked` bleiben erhalten."""
        rem_m, rem_t = self.cleanup_unhealthy_items_with_status(statuses=statuses, include_http_202=include_http_202)
        return [mid for mid, _ in rem_m], [tid for tid, _ in rem_t]

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

    def _reset_items_for_secret(
        self, clean_name: str, from_statuses: set[ProbeStatus], target_status: ProbeStatus, error_msg: str = ""
    ) -> None:
        is_paid_key = clean_name.endswith("_PAID_API_KEY")
        models_to_save: list[AIModelSpec] = []
        for m in self._repo.list_models():
            if not _model_matches_secret_env(m, clean_name):
                continue
            if is_paid_key and m.is_free and self._secrets.has_model_key(m, allow_paid=False):
                continue
            if m.last_status in from_statuses:
                models_to_save.append(m.model_copy(update={"last_status": target_status, "last_error": error_msg}))
        tools_to_save: list[ToolSpec] = [
            t.model_copy(update={"last_status": target_status, "last_error": error_msg})
            for t in self._repo.list_tools()
            if _tool_matches_secret_env(t, clean_name) and t.last_status in from_statuses
        ]
        self._repo.upsert_models_batch(models_to_save)
        self._repo.upsert_tools_batch(tools_to_save)

    def set_secret(self, env_name: str, new_secret_value: str) -> str:
        """Speichert oder überschreibt ein Secret komplett verschlüsselt (Write-Only) und entsperrt Key-/Quota-Einträge."""
        self._require_admin()
        clean_name = validate_secret_env_name(env_name)
        encrypted = encrypt_secret_value(new_secret_value)
        updated_at = self._repo.upsert_encrypted_secret(clean_name, encrypted)
        self._refresh_runtime_secrets()
        self._reset_items_for_secret(
            clean_name,
            {ProbeStatus.AUTH_MISSING, ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR},
            ProbeStatus.UNTESTED,
        )
        return updated_at

    def delete_secret(self, env_name: str) -> bool:
        """Entfernt ein im Brain-Vault gespeichertes Secret und aktualisiert abhängige Statuswerte."""
        self._require_admin()
        clean_name = validate_secret_env_name(env_name)
        removed = self._repo.delete_encrypted_secret(clean_name)
        self._refresh_runtime_secrets()
        if removed:
            models_to_save: list[AIModelSpec] = []
            for m in self._repo.list_models():
                if not _model_matches_secret_env(m, clean_name):
                    continue
                has_key = self._secrets.has_model_key(m, allow_paid=not m.is_free)
                if has_key and m.last_status in {ProbeStatus.OK, ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR, ProbeStatus.AUTH_MISSING}:
                    models_to_save.append(m.model_copy(update={"last_status": ProbeStatus.UNTESTED, "last_error": ""}))
                elif not has_key and m.last_status in {ProbeStatus.OK, ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR, ProbeStatus.UNTESTED}:
                    models_to_save.append(
                        m.model_copy(update={"last_status": ProbeStatus.AUTH_MISSING, "last_error": f"Key {clean_name} fehlt."})
                    )
            tools_to_save: list[ToolSpec] = []
            for t in self._repo.list_tools():
                if not _tool_matches_secret_env(t, clean_name):
                    continue
                has_key = self._secrets.has_tool_key(t, allow_paid=False)
                if has_key and t.last_status in {ProbeStatus.OK, ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR, ProbeStatus.AUTH_MISSING}:
                    tools_to_save.append(t.model_copy(update={"last_status": ProbeStatus.UNTESTED, "last_error": ""}))
                elif not has_key and t.last_status in {ProbeStatus.OK, ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR, ProbeStatus.UNTESTED}:
                    tools_to_save.append(
                        t.model_copy(update={"last_status": ProbeStatus.AUTH_MISSING, "last_error": f"Key {clean_name} fehlt."})
                    )
            self._repo.upsert_models_batch(models_to_save)
            self._repo.upsert_tools_batch(tools_to_save)
        return removed

    # --- App-Zugriffskontrolle (Whitelist: news_bot, research_bot, alert_bot) ---
    def list_app_clients(
        self, *, client_records: dict[str, dict[str, Any]] | None = None
    ) -> list[AppClientStatus]:
        """Liefert den sicheren Zugriffsstatus der 3 exklusiv zugelassenen Apps."""
        recs = client_records if client_records is not None else self._repo.list_app_client_records()
        return build_app_client_statuses(recs)

    def issue_app_token(self, app_id: str) -> str:
        """Generiert/rotiert ein 256-Bit-Access-Token für eine zugelassene App (Klartext nur 1× zurückgegeben!)."""
        self._require_admin()
        token, token_hash = generate_app_access_token(app_id)
        self._repo.upsert_app_client(app_id, token_hash, enabled=True)
        return token

    def revoke_app_token(self, app_id: str) -> bool:
        """Widerruft das Access-Token einer zugelassenen App sofort."""
        self._require_admin()
        return self._repo.revoke_app_client(app_id)

    def set_app_enabled(self, app_id: str, enabled: bool) -> bool:
        """Aktiviert oder sperrt den Zugriff einer bereits registrierten App."""
        self._require_admin()
        return self._repo.set_app_client_enabled(app_id, enabled)

    @staticmethod
    def _select_quick_verify_model(models: list[AIModelSpec]) -> AIModelSpec:
        def _rank(m: AIModelSpec) -> tuple[bool, int, int, float, str]:
            mid = m.model_id.lower()
            tier_rank = 0 if "lite" in mid else (1 if "flash" in mid else 2)
            lat = m.last_latency_ms if m.last_latency_ms > 0 else 999_999
            return (m.is_deep_agent, tier_rank, lat, -m.quality_score, m.id)

        return min(models, key=_rank)

    def probe_items_for_secret(self, env_name: str, *, quick_verify: bool = False) -> list[BenchmarkEntry]:
        """Testet zu einem Secret gehörende Modelle/Tools live (`quick_verify=True` prüft 1 schnelles Referenzmodell)."""
        from ki_brain.quotas import select_grounding_model

        clean_name = validate_secret_env_name(env_name)
        is_paid_key = clean_name.endswith("_PAID_API_KEY")
        self._refresh_runtime_secrets()
        bench = BrainBenchmarkRunner(self._repo, self._secrets, self._runner)
        usage_state = self._repo.load_usage_quotas()
        entries: list[BenchmarkEntry] = []

        target_models = [
            m
            for m in self._repo.list_models(enabled_only=True)
            if _model_matches_secret_env(m, clean_name) and (is_paid_key or m.is_free)
        ]
        if quick_verify and target_models:
            target_models = [self._select_quick_verify_model(target_models)]

        probe_timeout = 25.0 if quick_verify else 30.0
        for model in target_models:
            um, entry = bench.probe_model(
                model, usage_state=usage_state, include_paid=is_paid_key, timeout_seconds=probe_timeout
            )
            if um != model:
                self._repo.upsert_model(um)
            entries.append(entry)

        target_tools = (
            []
            if (quick_verify and target_models)
            else [t for t in self._repo.list_tools(enabled_only=True) if _tool_matches_secret_env(t, clean_name)]
        )
        if target_tools:
            all_models = self._repo.list_models(enabled_only=True)
            for tool in target_tools:
                g_model = select_grounding_model(
                    all_models, self._secrets, executor_type=tool.executor_type, allow_paid=is_paid_key
                )
                ut, entry = bench.probe_tool(
                    tool, DEFAULT_TEST_QUERY, DEFAULT_TEST_URL, g_model, usage_state=usage_state, include_paid=is_paid_key
                )
                if ut != tool:
                    self._repo.upsert_tool(ut)
                entries.append(entry)

        updated_seqs = self._repo.list_sequences()
        if any(e.status == ProbeStatus.OK for e in entries):
            seqs = optimize_tier_sequences(
                self._repo.list_models(),
                self._repo.list_tools(),
                updated_seqs,
                force_overwrite=False,
                usage_state=self._repo.load_usage_quotas(),
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
        extra_envs = [
            *(m.api_key_env for m in models_list),
            *(m.paid_api_key_env for m in models_list),
            *(t.api_key_env for t in tools_list),
            *vault_ts.keys(),
        ]
        for item_env in extra_envs:
            clean = item_env.strip().upper()
            if clean and clean not in all_names:
                all_names.append(clean)
        entries: list[SecretStatusEntry] = []
        for name in all_names:
            in_vault = name in vault_ts
            in_env = self._base_secrets.has_key(name)
            if name == "GEMINI_API_KEY" and not in_vault and not in_env:
                continue
            if in_vault and in_env:
                source = "Brain Vault (verschlüsselt, überschreibt ENV/Secrets)"
            elif in_vault:
                source = "Brain Vault (verschlüsselt)"
            elif in_env:
                source = "ENV / Streamlit Secrets"
            else:
                source = "Nicht gesetzt"
            cat, key_tier = _classify_secret_env(name, models_list, tools_list)
            affected = [
                *[
                    f"Model: {m.id}"
                    for m in models_list
                    if _model_matches_secret_env(m, name)
                    and (key_tier != "free" or m.is_free)
                ],
                *[f"Tool: {t.id}" for t in tools_list if _tool_matches_secret_env(t, name)],
            ]
            entries.append(
                SecretStatusEntry(
                    env_name=name,
                    is_configured=(in_vault or in_env),
                    source=source,
                    updated_at=vault_ts.get(name, ""),
                    affected_items=affected,
                    category=cat,
                    key_tier=key_tier,
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

    # --- Auto-Discovery, Testlauf & Vereinte Pipeline (Fast, Standard, Deep) ---
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
        include_paid: bool = False,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> BenchmarkReport:
        runner = BrainBenchmarkRunner(self._repo, self._secrets, self._runner)
        return runner.run_benchmark(
            test_query=test_query,
            test_url=test_url,
            force_overwrite_sequences=force_overwrite_sequences,
            include_paid=include_paid,
            on_progress=on_progress,
        )

    def run_full_pipeline(
        self,
        *,
        run_discover: bool = True,
        run_benchmark: bool = True,
        run_unblock: bool = True,
        run_prune: bool = True,
        exclude_previews: bool = True,
        include_paid: bool = False,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> PipelineSummary:
        """Führt den kombinierten KI-Brain-Ablauf aus (Discover → Benchmark → Unblock → Prune)."""
        steps: list[tuple[str, float]] = [
            (name, weight)
            for name, enabled, weight in (
                ("discover", run_discover, 0.30),
                ("benchmark", run_benchmark, 0.42),
                ("unblock", run_unblock, 0.18),
                ("prune", run_prune, 0.10),
            )
            if enabled
        ]
        if not steps:
            return PipelineSummary()

        total_w = sum(w for _, w in steps)
        cursor = 0.0
        executed: list[str] = []
        warnings: list[str] = []
        discovery: DiscoverySummary | None = None
        benchmark: BenchmarkReport | None = None
        unblock: UnblockSummary | None = None
        pruned_models: list[tuple[str, ProbeStatus]] = []
        pruned_tools: list[tuple[str, ProbeStatus]] = []

        def _slice_cb(start: float, span: float, label: str) -> Callable[[float, str], None]:
            def _inner(frac: float, msg: str) -> None:
                if on_progress:
                    clamped = max(0.0, min(1.0, float(frac)))
                    on_progress(min(0.99, start + span * clamped), f"[{label}] {msg}")

            return _inner

        for step_name, weight in steps:
            span = weight / total_w
            if step_name == "discover":
                cb = _slice_cb(cursor, span, "1/4 Discover")
                cb(0.05, "Prüfe Quota-Recovery & neue Modelle/Tools...")
                discovery = self.auto_sync(
                    verify_discovered=True,
                    exclude_previews=exclude_previews,
                    on_progress=cb,
                )
                warnings.extend(discovery.warnings)
                executed.append("discover")
            elif step_name == "benchmark":
                cb = _slice_cb(cursor, span, "2/4 Benchmark")
                cb(0.05, "Teste aktive Modelle & Tools live...")
                benchmark = self.run_benchmark(include_paid=include_paid, on_progress=cb)
                executed.append("benchmark")
            elif step_name == "unblock":
                cb = _slice_cb(cursor, span, "3/4 Unblock")
                cb(0.05, "Prüfe & entsperre Blocked-Tools...")
                unblock = self.resolve_blocked_tools(on_progress=cb)
                executed.append("unblock")
            elif step_name == "prune":
                cb = _slice_cb(cursor, span, "4/4 Prune")
                cb(0.25, "Bereinige Error-, Quota-429- & HTTP-202-Einträge...")
                pruned_models, pruned_tools = self.cleanup_unhealthy_items_with_status()
                cb(1.0, "Bereinigung & Sequenz-Optimierung abgeschlossen.")
                executed.append("prune")
            cursor += span

        if on_progress:
            on_progress(1.0, "KI-Brain Ablauf abgeschlossen.")
        return PipelineSummary(
            executed_steps=tuple(executed),
            discovery=discovery,
            benchmark=benchmark,
            unblock=unblock,
            pruned_models=tuple(pruned_models),
            pruned_tools=tuple(pruned_tools),
            warnings=tuple(warnings),
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
        allow_paid: bool = False,
        use_maps: bool | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        on_action: Callable[[str], None] | None = None,
    ) -> BrainSearchResult:
        return execute_tiered_search(
            self._repo,
            self._secrets,
            query,
            BrainTier(tier),
            language=language,
            max_results=max_results,
            runner=self._runner,
            allow_paid=allow_paid,
            use_maps=use_maps,
            latitude=latitude,
            longitude=longitude,
            on_action=on_action,
        )

    def fetch_page(self, url: str, *, allow_paid: bool = False) -> BrainFetchedPage:
        return execute_tiered_fetch_page(self._repo, self._secrets, url, allow_paid=allow_paid)

    def generate_text(
        self,
        prompt: str,
        *,
        tier: BrainTier | str = BrainTier.FAST,
        system_instruction: str = "",
        allow_paid: bool = False,
        on_action: Callable[[str], None] | None = None,
    ) -> BrainGenerationResult:
        return execute_tiered_text(
            self._repo,
            self._secrets,
            self._runner,
            prompt,
            tier=BrainTier(tier),
            system_instruction=system_instruction,
            allow_paid=allow_paid,
            on_action=on_action,
        )

    def generate_structured(
        self,
        prompt: str,
        schema_cls: type[ModelT],
        *,
        tier: BrainTier | str = BrainTier.FAST,
        system_instruction: str = "",
        allow_paid: bool = False,
    ) -> tuple[ModelT, str]:
        return execute_tiered_structured(
            self._repo,
            self._secrets,
            self._runner,
            prompt,
            schema_cls,
            tier=BrainTier(tier),
            system_instruction=system_instruction,
            allow_paid=allow_paid,
        )

    def ask(
        self,
        query: str,
        *,
        tier: BrainTier | str = BrainTier.STANDARD,
        language: str = "de",
        system_instruction: str = "",
        allow_paid: bool = False,
        use_maps: bool | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
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
            allow_paid=allow_paid,
            use_maps=use_maps,
            latitude=latitude,
            longitude=longitude,
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
