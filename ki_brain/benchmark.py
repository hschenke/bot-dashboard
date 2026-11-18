"""Benchmark- & Testlauf-Engine zur Messung aller Tools & Modelle im Brain."""

from __future__ import annotations

import datetime
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ki_brain.enums import ProbeStatus, ToolCategory
from ki_brain.llm_runner import UnifiedLLMRunner, is_quota_error, pop_last_grounding_counts
from ki_brain.models import (
    AIModelSpec,
    BenchmarkEntry,
    BenchmarkReport,
    BrainSecrets,
    ToolSpec,
)
from ki_brain.optimizer import (
    DEFAULT_TEST_QUERY,
    DEFAULT_TEST_URL,
    compute_model_quality,
    compute_tool_quality,
    optimize_tier_sequences,
)
from ki_brain.quotas import (
    evaluate_model_quota,
    evaluate_tool_quota,
    record_quota_usage,
    select_grounding_model,
)
from ki_brain.storage import BrainRepository
from ki_brain.tools.http_utils import consume_probe_trace, diagnose_empty_probe, start_probe_trace
from ki_brain.tools.registry import (
    execute_extractor_tool,
    execute_search_tool,
    is_extractor_capable,
)

_PROBE_RULE_QUERIES: dict[str, str] = {
    "duckduckgo_instant": "Python",
    "open_meteo": "Berlin",
    "wikidata_api": "Python",
    "gemini_google_maps": "Cafés am Alexanderplatz in Berlin",
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _get_probe_query(tool: ToolSpec, default_query: str) -> str:
    cfg_query = str((tool.config or {}).get("probe_query") or "").strip()
    if cfg_query:
        return cfg_query
    return _PROBE_RULE_QUERIES.get(tool.executor_type.strip().lower(), default_query)


def _compute_grounding_usage(executor_type: str, counts: dict[str, int], *, is_quota_capped: bool) -> int:
    clean = executor_type.strip().lower()
    if clean == "gemini_google_search":
        return max(1, int(counts.get("google_search", 1))) if is_quota_capped else int(counts.get("google_search", 0))
    if clean == "gemini_google_maps":
        return max(1, int(counts.get("google_maps", 1))) if is_quota_capped else int(counts.get("google_maps", 0))
    return 0


class BrainBenchmarkRunner:
    """Testet alle aktivierten Tools & Modelle live und aktualisiert Fast/Standard/Deep."""

    def __init__(self, repo: BrainRepository, secrets: BrainSecrets, runner: UnifiedLLMRunner) -> None:
        self._repo = repo
        self._secrets = secrets
        self._runner = runner

    def probe_tool(
        self,
        tool: ToolSpec,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        grounding_model: AIModelSpec | None = None,
        *,
        usage_state: dict[str, Any] | None = None,
        include_paid: bool = False,
    ) -> tuple[ToolSpec, BenchmarkEntry]:
        updated, entry, used_units = self._probe_tool_with_usage(
            tool,
            test_query=test_query,
            test_url=test_url,
            grounding_model=grounding_model,
            usage_state=usage_state,
            include_paid=include_paid,
        )
        if used_units > 0:
            record_quota_usage(self._repo, tool_counts={tool.id: used_units})
        return updated, entry

    def _probe_tool_with_usage(
        self,
        tool: ToolSpec,
        *,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        grounding_model: AIModelSpec | None = None,
        usage_state: dict[str, Any] | None = None,
        include_paid: bool = False,
    ) -> tuple[ToolSpec, BenchmarkEntry, int]:
        now = _utc_now_iso()
        if tool.requires_api_key and not self._secrets.has_key(tool.api_key_env):
            u, e = self._record_tool_result(tool, ProbeStatus.AUTH_MISSING, 0, 0, 0, f"Key {tool.api_key_env} fehlt.", now)
            return u, e, 0

        if not include_paid and usage_state is not None:
            q_snap = evaluate_tool_quota(tool, usage_state)
            if not q_snap.is_free_available:
                note = "0%-Schutz: Free-Kontingent erschöpft oder Paid-Only (Opt-In für Live-Probe nötig)."
                entry = BenchmarkEntry(
                    item_type="tool", item_id=tool.id, name=tool.name, status=tool.last_status,
                    latency_ms=tool.last_latency_ms, result_count=0, quality_score=tool.quality_score, details=note,
                )
                return tool, entry, 0

        t0 = time.monotonic()
        start_probe_trace()
        try:
            if tool.category == ToolCategory.AI_GROUNDING:
                consume_probe_trace()
                return self._probe_ai_grounding_tool(tool, test_query, test_url, grounding_model, t0, now, include_paid=include_paid)
            if is_extractor_capable(tool):
                page = execute_extractor_tool(tool, test_url, secrets=self._secrets)
                consume_probe_trace()
                lat = int((time.monotonic() - t0) * 1000)
                chars = len(page.text_content)
                used = 1 if tool.free_quota_limit > 0 else 0
                u, e = self._record_tool_result(tool, ProbeStatus.OK, lat, 1, chars, f"{chars} Zeichen extrahiert.", now)
                return u, e, used

            query_to_use = _get_probe_query(tool, test_query)
            hits = execute_search_tool(tool, query_to_use, language="de", secrets=self._secrets, limit=4)
            trace_events = consume_probe_trace()
            lat = int((time.monotonic() - t0) * 1000)
            if not hits:
                is_quota, diag = diagnose_empty_probe(trace_events, query_to_use)
                status = ProbeStatus.QUOTA_LIMITED if is_quota else ProbeStatus.BLOCKED
                u, e = self._record_tool_result(tool, status, lat, 0, 0, diag, now)
                return u, e, 0
            used = 1 if tool.free_quota_limit > 0 else 0
            u, e = self._record_tool_result(
                tool, ProbeStatus.OK, lat, len(hits), 0, f"{len(hits)} Treffer ({lat} ms, Query='{query_to_use}').", now
            )
            return u, e, used
        except Exception as exc:
            consume_probe_trace()
            lat = int((time.monotonic() - t0) * 1000)
            status = ProbeStatus.QUOTA_LIMITED if is_quota_error(exc) else ProbeStatus.ERROR
            u, e = self._record_tool_result(tool, status, lat, 0, 0, str(exc)[:300], now)
            return u, e, 0

    def _probe_ai_grounding_tool(
        self,
        tool: ToolSpec,
        test_query: str,
        test_url: str,
        grounding_model: AIModelSpec | None,
        t0: float,
        now: str,
        *,
        include_paid: bool = False,
    ) -> tuple[ToolSpec, BenchmarkEntry, int]:
        etype = tool.executor_type.strip().lower()
        if grounding_model is None:
            msg = (
                "Gemini 3+ Modell für kostenfreies Maps-Grounding erforderlich (Gemini 2.5 ab $25/1k)."
                if etype == "gemini_google_maps" and not include_paid
                else "Kein aktives Gemini-Modell."
            )
            u, e = self._record_tool_result(tool, ProbeStatus.ERROR, 0, 0, 0, msg, now)
            return u, e, 0
        query_to_use = _get_probe_query(tool, test_query)
        if etype == "gemini_url_context":
            t_type, prompt = "url_context", f"Fasse kurz zusammen: {test_url}"
        elif etype == "gemini_google_maps":
            t_type, prompt = "google_maps", f"Finde passende Orte und Details auf Google Maps zu: {query_to_use}"
        else:
            t_type, prompt = "google_search", f"Suche aktuell nach: {query_to_use}"

        text, citations = self._runner.run_text(
            grounding_model, prompt, timeout_seconds=min(15.0, tool.timeout_seconds), tools=[{"type": t_type}]
        )
        g_counts = pop_last_grounding_counts()
        used = _compute_grounding_usage(etype, g_counts, is_quota_capped=tool.free_quota_limit > 0)
        lat = int((time.monotonic() - t0) * 1000)
        count = max(1 if text else 0, len(citations))
        u, e = self._record_tool_result(tool, ProbeStatus.OK, lat, count, len(text), f"{len(citations)} Zitate ({lat} ms).", now)
        return u, e, used

    @staticmethod
    def _record_tool_result(
        tool: ToolSpec,
        status: ProbeStatus,
        latency_ms: int,
        result_count: int,
        text_chars: int,
        details: str,
        now: str,
    ) -> tuple[ToolSpec, BenchmarkEntry]:
        quality = compute_tool_quality(status, latency_ms, result_count, text_chars=text_chars)
        prev_rel = tool.reliability_score or (100.0 if status == ProbeStatus.OK else 0.0)
        curr_rel = 100.0 if status == ProbeStatus.OK else (20.0 if status == ProbeStatus.QUOTA_LIMITED else 0.0)
        reliability = round(0.6 * curr_rel + 0.4 * prev_rel, 1) if tool.last_tested_at else curr_rel
        updated = tool.model_copy(
            update={
                "last_status": status,
                "last_latency_ms": latency_ms,
                "last_result_count": result_count,
                "reliability_score": reliability,
                "quality_score": quality,
                "last_tested_at": now,
                "last_error": "" if status == ProbeStatus.OK else details[:380],
            }
        )
        entry = BenchmarkEntry(
            item_type="tool",
            item_id=tool.id,
            name=tool.name,
            status=status,
            latency_ms=latency_ms,
            result_count=result_count,
            quality_score=quality,
            details=details,
        )
        return updated, entry

    def probe_model(
        self,
        model: AIModelSpec,
        *,
        usage_state: dict[str, Any] | None = None,
        include_paid: bool = False,
    ) -> tuple[AIModelSpec, BenchmarkEntry]:
        now = _utc_now_iso()
        if model.requires_api_key and not self._secrets.has_key(model.api_key_env):
            return self._record_model_result(model, ProbeStatus.AUTH_MISSING, 0, f"Key {model.api_key_env} fehlt.", now)
        if not include_paid and not model.is_deep_agent and usage_state is not None:
            m_snap = evaluate_model_quota(model, usage_state)
            if not m_snap.is_free_available:
                note = "0%-Schutz: Paid-Only / Free-Kontingent erschöpft (Test nur mit Paid-Opt-In)."
                entry = BenchmarkEntry(
                    item_type="model", item_id=model.id, name=model.name, status=model.last_status,
                    latency_ms=model.last_latency_ms, result_count=0, quality_score=model.quality_score, details=note,
                )
                return model, entry
        t0 = time.monotonic()
        try:
            if model.is_deep_agent:
                client = self._runner._get_genai_client(timeout_seconds=12.0)  # noqa: SLF001
                listed = list(client.models.list())
                lat = int((time.monotonic() - t0) * 1000)
                details = f"Deep-Research-Agent API erreichbar ({len(listed)} Modelle, {lat} ms)."
                return self._record_model_result(model, ProbeStatus.OK, lat, details, now)

            text, _ = self._runner.run_text(model, "Antworte nur mit OK.", timeout_seconds=18.0)
            lat = int((time.monotonic() - t0) * 1000)
            return self._record_model_result(model, ProbeStatus.OK, lat, f"Antwort '{text[:30]}' ({lat} ms).", now)
        except Exception as exc:
            lat = int((time.monotonic() - t0) * 1000)
            status = ProbeStatus.QUOTA_LIMITED if is_quota_error(exc) else ProbeStatus.ERROR
            return self._record_model_result(model, status, lat, str(exc)[:300], now)

    @staticmethod
    def _record_model_result(
        model: AIModelSpec, status: ProbeStatus, latency_ms: int, details: str, now: str
    ) -> tuple[AIModelSpec, BenchmarkEntry]:
        quality = compute_model_quality(model, status, latency_ms)
        updated = model.model_copy(
            update={
                "last_status": status,
                "last_latency_ms": latency_ms,
                "quality_score": quality,
                "last_tested_at": now,
                "last_error": "" if status == ProbeStatus.OK else details[:380],
            }
        )
        entry = BenchmarkEntry(
            item_type="model",
            item_id=model.id,
            name=model.name,
            status=status,
            latency_ms=latency_ms,
            result_count=1 if status == ProbeStatus.OK else 0,
            quality_score=quality,
            details=details,
        )
        return updated, entry

    def run_benchmark(
        self,
        *,
        test_query: str = DEFAULT_TEST_QUERY,
        test_url: str = DEFAULT_TEST_URL,
        force_overwrite_sequences: bool = False,
        include_paid: bool = False,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> BenchmarkReport:
        """Testet alle aktivierten Tools & Modelle (Standard: strikt 0% Kosten) und optimiert Fast, Standard & Deep."""
        started = _utc_now_iso()
        models = self._repo.list_models(enabled_only=True)
        tools = self._repo.list_tools(enabled_only=True)
        usage_state = self._repo.load_usage_quotas()
        total_steps = max(1, len(models) + len(tools) + 1)
        done = 0

        entries: list[BenchmarkEntry] = []
        updated_models: list[AIModelSpec] = []
        for model in models:
            if on_progress:
                on_progress(min(0.95, max(0.03, done / total_steps)), f"Teste KI-Modell ({done + 1}/{len(models)}): `{model.id}`...")
            um, entry = self.probe_model(model, usage_state=usage_state, include_paid=include_paid)
            if um != model:
                self._repo.upsert_model(um)
            updated_models.append(um)
            entries.append(entry)
            done += 1

        g_models = {
            t.id: select_grounding_model(
                updated_models,
                self._secrets,
                executor_type=t.executor_type,
                allow_paid=include_paid,
            )
            for t in tools
            if t.category == ToolCategory.AI_GROUNDING
        }

        updated_tools: list[ToolSpec] = []
        usage_deltas: dict[str, int] = {}
        if on_progress and tools:
            on_progress(min(0.95, done / total_steps), f"Teste {len(tools)} Websearch- & Extraktions-Tools parallel...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [
                pool.submit(
                    self._probe_tool_with_usage,
                    tool,
                    test_query=test_query,
                    test_url=test_url,
                    grounding_model=g_models.get(tool.id),
                    usage_state=usage_state,
                    include_paid=include_paid,
                )
                for tool in tools
            ]
            for fut in as_completed(futs):
                ut, entry, used_units = fut.result()
                if ut != next((orig for orig in tools if orig.id == ut.id), None):
                    self._repo.upsert_tool(ut)
                if used_units > 0:
                    usage_deltas[ut.id] = usage_deltas.get(ut.id, 0) + used_units
                updated_tools.append(ut)
                entries.append(entry)
                done += 1
                if on_progress:
                    on_progress(min(0.94, done / total_steps), f"Tool geprüft ({done}/{len(models) + len(tools)}): `{ut.id}` → {ut.last_status.value}")

        if usage_deltas:
            usage_state = record_quota_usage(self._repo, tool_counts=usage_deltas)

        if on_progress:
            on_progress(0.96, "Optimiere 0%-Kosten-Sequenzen für Fast, Standard & Deep...")
        all_models = self._repo.list_models()
        all_tools = self._repo.list_tools()
        sequences = optimize_tier_sequences(
            all_models,
            all_tools,
            self._repo.list_sequences(),
            force_overwrite=force_overwrite_sequences,
            usage_state=usage_state,
        )
        saved_seqs = {tier.value: self._repo.upsert_sequence(seq) for tier, seq in sequences.items()}

        report = BenchmarkReport(
            run_id=str(uuid.uuid4())[:12],
            started_at=started,
            finished_at=_utc_now_iso(),
            test_query=test_query,
            test_url=test_url,
            entries=entries,
            sequences=saved_seqs,
            healthy_tools_count=sum(1 for t in updated_tools if t.last_status == ProbeStatus.OK),
            healthy_models_count=sum(1 for m in updated_models if m.last_status == ProbeStatus.OK),
        )
        self._repo.save_benchmark_run(report)
        if on_progress:
            on_progress(1.0, f"Testlauf `{report.run_id}` fertig.")
        return report
