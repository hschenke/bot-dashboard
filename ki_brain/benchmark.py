"""Benchmark- & Testlauf-Engine zur Messung aller Tools & Modelle im Brain."""

from __future__ import annotations

import datetime
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from ki_brain.enums import ProbeStatus, ToolCategory
from ki_brain.llm_runner import UnifiedLLMRunner, is_quota_error
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
from ki_brain.storage import BrainRepository
from ki_brain.tools.registry import (
    execute_extractor_tool,
    execute_search_tool,
    is_extractor_capable,
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


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
    ) -> tuple[ToolSpec, BenchmarkEntry]:
        now = _utc_now_iso()
        if tool.requires_api_key and not self._secrets.has_key(tool.api_key_env):
            return self._record_tool_result(tool, ProbeStatus.AUTH_MISSING, 0, 0, 0, f"Key {tool.api_key_env} fehlt.", now)

        t0 = time.monotonic()
        try:
            if tool.category == ToolCategory.AI_GROUNDING:
                return self._probe_ai_grounding_tool(tool, test_query, test_url, grounding_model, t0, now)
            if is_extractor_capable(tool):
                page = execute_extractor_tool(tool, test_url, secrets=self._secrets)
                lat = int((time.monotonic() - t0) * 1000)
                chars = len(page.text_content)
                return self._record_tool_result(tool, ProbeStatus.OK, lat, 1, chars, f"{chars} Zeichen extrahiert.", now)

            query_to_use = "Berlin" if tool.executor_type == "open_meteo" else test_query
            hits = execute_search_tool(tool, query_to_use, language="de", secrets=self._secrets, limit=4)
            lat = int((time.monotonic() - t0) * 1000)
            if not hits:
                return self._record_tool_result(tool, ProbeStatus.BLOCKED, lat, 0, 0, "0 Treffer erhalten (leer/blockiert).", now)
            return self._record_tool_result(tool, ProbeStatus.OK, lat, len(hits), 0, f"{len(hits)} Treffer ({lat} ms).", now)
        except Exception as exc:
            lat = int((time.monotonic() - t0) * 1000)
            status = ProbeStatus.QUOTA_LIMITED if is_quota_error(exc) else ProbeStatus.ERROR
            return self._record_tool_result(tool, status, lat, 0, 0, str(exc)[:300], now)

    def _probe_ai_grounding_tool(
        self,
        tool: ToolSpec,
        test_query: str,
        test_url: str,
        grounding_model: AIModelSpec | None,
        t0: float,
        now: str,
    ) -> tuple[ToolSpec, BenchmarkEntry]:
        if grounding_model is None:
            return self._record_tool_result(tool, ProbeStatus.ERROR, 0, 0, 0, "Kein aktives Gemini-Modell.", now)
        t_type = "url_context" if tool.executor_type == "gemini_url_context" else "google_search"
        prompt = f"Fasse kurz zusammen: {test_url}" if t_type == "url_context" else f"Suche aktuell nach: {test_query}"
        text, citations = self._runner.run_text(
            grounding_model, prompt, timeout_seconds=min(15.0, tool.timeout_seconds), tools=[{"type": t_type}]
        )
        lat = int((time.monotonic() - t0) * 1000)
        count = max(1 if text else 0, len(citations))
        return self._record_tool_result(tool, ProbeStatus.OK, lat, count, len(text), f"{len(citations)} Zitate ({lat} ms).", now)

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

    def probe_model(self, model: AIModelSpec) -> tuple[AIModelSpec, BenchmarkEntry]:
        now = _utc_now_iso()
        if model.requires_api_key and not self._secrets.has_key(model.api_key_env):
            return self._record_model_result(model, ProbeStatus.AUTH_MISSING, 0, f"Key {model.api_key_env} fehlt.", now)
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
    ) -> BenchmarkReport:
        """Testet alle aktivierten Tools & Modelle und optimiert Fast, Standard & Deep."""
        started = _utc_now_iso()
        models = self._repo.list_models(enabled_only=True)
        tools = self._repo.list_tools(enabled_only=True)

        entries: list[BenchmarkEntry] = []
        updated_models: list[AIModelSpec] = []
        for model in models:
            um, entry = self.probe_model(model)
            self._repo.upsert_model(um)
            updated_models.append(um)
            entries.append(entry)

        grounding_model = next(
            (m for m in updated_models if not m.is_deep_agent and m.provider.value == "google_gemini" and m.last_status == ProbeStatus.OK),
            next((m for m in updated_models if not m.is_deep_agent and m.provider.value == "google_gemini"), None),
        )

        updated_tools: list[ToolSpec] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(self.probe_tool, tool, test_query, test_url, grounding_model) for tool in tools]
            for fut in futs:
                ut, entry = fut.result()
                self._repo.upsert_tool(ut)
                updated_tools.append(ut)
                entries.append(entry)

        all_models = self._repo.list_models()
        all_tools = self._repo.list_tools()
        sequences = optimize_tier_sequences(
            all_models, all_tools, self._repo.list_sequences(), force_overwrite=force_overwrite_sequences
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
        return report
