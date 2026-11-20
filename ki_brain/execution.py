"""Stufengesteuerte Ausführungs-Pipeline für Websearch, Extraktion & KI-Synthese."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from ki_brain.enums import BrainTier, ProbeStatus
from ki_brain.exceptions import BrainProviderError, BrainSSRFError
from ki_brain.llm_runner import UnifiedLLMRunner, is_quota_error, pop_last_grounding_counts
from ki_brain.models import (
    AIModelSpec,
    BrainCitation,
    BrainFetchedPage,
    BrainGenerationResult,
    BrainSearchResult,
    BrainSecrets,
    TierSequenceConfig,
    ToolSpec,
)
from ki_brain.optimizer import compute_tool_quality
from ki_brain.quotas import (
    evaluate_model_quota,
    evaluate_tool_quota,
    is_geo_or_maps_query,
    record_quota_usage,
    select_grounding_model,
)
from ki_brain.storage import BrainRepository
from ki_brain.tools.registry import (
    execute_extractor_tool,
    execute_search_tool,
    is_extractor_capable,
    is_search_capable,
)

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)
_AI_GROUNDING_SEARCH = frozenset({"gemini_google_search", "gemini_google_maps"})


def _build_passive_tool_update(
    tool: ToolSpec,
    *,
    ok: bool,
    latency_ms: int,
    result_count: int = 0,
    text_chars: int = 0,
    exc: Exception | None = None,
) -> ToolSpec | None:
    """Erstellt ein passives Health-Update aus produktivem Traffic (0 Extra-Calls)."""
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    if ok:
        quality = compute_tool_quality(ProbeStatus.OK, latency_ms, result_count, text_chars=text_chars)
        prev_rel = tool.reliability_score or 100.0
        reliability = round(0.6 * 100.0 + 0.4 * prev_rel, 1) if tool.last_tested_at else 100.0
        return tool.model_copy(
            update={
                "last_status": ProbeStatus.OK,
                "last_latency_ms": max(1, int(latency_ms)),
                "last_result_count": max(0, int(result_count)),
                "reliability_score": reliability,
                "quality_score": quality,
                "last_tested_at": now,
                "last_error": "",
            }
        )
    if exc is not None and is_quota_error(exc):
        prev_rel = tool.reliability_score or 20.0
        reliability = round(0.6 * 20.0 + 0.4 * prev_rel, 1) if tool.last_tested_at else 20.0
        return tool.model_copy(
            update={
                "last_status": ProbeStatus.QUOTA_LIMITED,
                "last_latency_ms": max(0, int(latency_ms)),
                "reliability_score": reliability,
                "quality_score": compute_tool_quality(ProbeStatus.QUOTA_LIMITED, latency_ms, 0),
                "last_tested_at": now,
                "last_error": str(exc)[:380],
            }
        )
    return None


def _persist_passive_tool_health(repo: BrainRepository, updates: list[ToolSpec]) -> None:
    if not updates:
        return
    try:
        repo.upsert_tools_batch(updates)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Passives Tool-Health-Update übersprungen: %s", exc)


def _ordered_models(
    repo: BrainRepository,
    tier: BrainTier,
    *,
    allow_paid: bool = False,
    usage_state: dict[str, Any] | None = None,
) -> list[AIModelSpec]:
    enabled = repo.list_models(enabled_only=True)
    allowed = {
        m.id: m
        for m in enabled
        if allow_paid or evaluate_model_quota(m, usage_state).is_free_available
    }
    seq = repo.get_sequence(tier)
    seq_ids = seq.paid_model_sequence if (allow_paid and seq.paid_model_sequence) else seq.model_sequence
    ordered = [allowed[mid] for mid in seq_ids if mid in allowed]
    seen = {m.id for m in ordered}
    for m in allowed.values():
        if m.id not in seen and (tier == BrainTier.DEEP or not m.is_deep_agent):
            ordered.append(m)
    return ordered


def _ordered_tools(
    repo: BrainRepository,
    seq: TierSequenceConfig,
    capability_fn: Callable[[ToolSpec], bool],
    *,
    is_search: bool = True,
    allow_paid: bool = False,
    usage_state: dict[str, Any] | None = None,
) -> list[ToolSpec]:
    enabled = [t for t in repo.list_tools(enabled_only=True) if capability_fn(t)]
    allowed = {
        t.id: t
        for t in enabled
        if allow_paid or evaluate_tool_quota(t, usage_state).is_free_available
    }
    seq_ids = (
        (seq.paid_search_tool_sequence if (allow_paid and seq.paid_search_tool_sequence) else seq.search_tool_sequence)
        if is_search
        else (
            seq.paid_extractor_tool_sequence
            if (allow_paid and seq.paid_extractor_tool_sequence)
            else seq.extractor_tool_sequence
        )
    )
    ordered = [allowed[tid] for tid in seq_ids if tid in allowed]
    seen = {t.id for t in ordered}
    for t in allowed.values():
        if t.id not in seen:
            ordered.append(t)
    return ordered


def _prioritize_maps_tools(
    candidates: list[ToolSpec], *, wants_maps: bool, max_search_tools: int
) -> list[ToolSpec]:
    maps_tools = [t for t in candidates if t.executor_type.strip().lower() == "gemini_google_maps"]
    others = [t for t in candidates if t.executor_type.strip().lower() != "gemini_google_maps"]
    if not wants_maps or not maps_tools:
        return others
    insert_at = 0 if (max_search_tools <= 1 or not others) else 1
    return others[:insert_at] + maps_tools + others[insert_at:]


def _resolve_tool_usage_delta(tool: ToolSpec) -> int:
    etype = tool.executor_type.strip().lower()
    if etype in _AI_GROUNDING_SEARCH:
        counts = pop_last_grounding_counts()
        key = "google_maps" if etype == "gemini_google_maps" else "google_search"
        raw = int(counts.get(key, 0))
        return max(1, raw) if tool.free_quota_limit > 0 else raw
    return 1 if tool.free_quota_limit > 0 else 0


def execute_tiered_search(
    repo: BrainRepository,
    secrets: BrainSecrets,
    query: str,
    tier: BrainTier = BrainTier.STANDARD,
    *,
    language: str = "de",
    max_results: int = 8,
    runner: Any = None,
    allow_paid: bool = False,
    use_maps: bool | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    on_action: Callable[[str], None] | None = None,
) -> BrainSearchResult:
    """Durchsucht Tools in optimierter Sequenz (Standard: strikt 0% Kosten, Maps Geo-Smart)."""
    t0 = time.monotonic()
    clean_q = " ".join((query or "").split())[:240]
    if not clean_q:
        return BrainSearchResult(query="", tier=tier, latency_ms=0)

    seq = repo.get_sequence(tier)
    usage_state = repo.load_usage_quotas()
    wants_maps = (use_maps is True) or (
        use_maps is None and (is_geo_or_maps_query(clean_q) or (latitude is not None and longitude is not None))
    )
    ordered = _ordered_tools(repo, seq, is_search_capable, is_search=True, allow_paid=allow_paid, usage_state=usage_state)
    candidates = _prioritize_maps_tools(ordered, wants_maps=wants_maps, max_search_tools=seq.max_search_tools)
    enabled_models = repo.list_models(enabled_only=True)

    citations: list[BrainCitation] = []
    seen_urls: set[str] = set()
    executed: list[str] = []
    errors: dict[str, str] = {}
    usage_deltas: dict[str, int] = {}
    health_updates: list[ToolSpec] = []
    tools_with_hits = 0

    for tool in candidates:
        if tools_with_hits >= seq.max_search_tools:
            break
        if tool.requires_api_key and not secrets.has_tool_key(tool, allow_paid=allow_paid):
            continue
        etype = tool.executor_type.strip().lower()
        g_model: AIModelSpec | None = None
        if etype in _AI_GROUNDING_SEARCH:
            g_model = select_grounding_model(enabled_models, secrets, executor_type=etype, allow_paid=allow_paid)
            if g_model is None:
                continue
            pop_last_grounding_counts()
        if on_action:
            on_action(f"🔎 Suche via {tool.name}...")
        executed.append(tool.id)
        t_call = time.monotonic()
        try:
            if etype in _AI_GROUNDING_SEARCH:
                hits = execute_search_tool(
                    tool, clean_q, language=language, secrets=secrets, limit=tool.max_results,
                    runner=runner, grounding_model=g_model, latitude=latitude, longitude=longitude,
                    allow_paid=allow_paid,
                )
            else:
                hits = execute_search_tool(tool, clean_q, language=language, secrets=secrets, limit=tool.max_results)
            call_ms = int((time.monotonic() - t_call) * 1000)
            delta = _resolve_tool_usage_delta(tool)
            if delta > 0:
                usage_deltas[tool.id] = usage_deltas.get(tool.id, 0) + delta
            if hits:
                upd = _build_passive_tool_update(tool, ok=True, latency_ms=call_ms, result_count=len(hits))
                if upd is not None:
                    health_updates.append(upd)
            added_here = 0
            for hit in hits:
                if hit.url not in seen_urls:
                    seen_urls.add(hit.url)
                    citations.append(hit)
                    added_here += 1
            if added_here > 0:
                tools_with_hits += 1
        except Exception as exc:
            pop_last_grounding_counts()
            call_ms = int((time.monotonic() - t_call) * 1000)
            upd = _build_passive_tool_update(tool, ok=False, latency_ms=call_ms, exc=exc)
            if upd is not None:
                health_updates.append(upd)
            errors[tool.id] = str(exc)[:220]

    if usage_deltas:
        record_quota_usage(repo, tool_counts=usage_deltas)
    _persist_passive_tool_health(repo, health_updates)

    return BrainSearchResult(
        query=clean_q,
        tier=tier,
        citations=citations[: max(max_results, len(citations)) if tier == BrainTier.DEEP else max_results],
        tools_executed=executed,
        tool_errors=errors,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def execute_tiered_fetch_page(
    repo: BrainRepository,
    secrets: BrainSecrets,
    url: str,
    *,
    allow_paid: bool = False,
) -> BrainFetchedPage:
    """Extrahiert Seiteninhalt über die priorisierte Extractor-Sequenz (Standard: 0% Kosten)."""
    seq = repo.get_sequence(BrainTier.DEEP)
    usage_state = repo.load_usage_quotas()
    candidates = _ordered_tools(
        repo, seq, is_extractor_capable, is_search=False, allow_paid=allow_paid, usage_state=usage_state
    )
    last_exc: Exception | None = None
    health_updates: list[ToolSpec] = []
    for tool in candidates:
        if tool.requires_api_key and not secrets.has_key(tool.api_key_env):
            continue
        t_call = time.monotonic()
        try:
            page = execute_extractor_tool(tool, url, secrets=secrets)
            call_ms = int((time.monotonic() - t_call) * 1000)
            if tool.free_quota_limit > 0:
                record_quota_usage(repo, tool_counts={tool.id: 1})
            upd = _build_passive_tool_update(
                tool, ok=True, latency_ms=call_ms, result_count=1, text_chars=len(page.text_content)
            )
            if upd is not None:
                health_updates.append(upd)
            _persist_passive_tool_health(repo, health_updates)
            return page
        except BrainSSRFError:
            _persist_passive_tool_health(repo, health_updates)
            raise
        except Exception as exc:
            call_ms = int((time.monotonic() - t_call) * 1000)
            upd = _build_passive_tool_update(tool, ok=False, latency_ms=call_ms, exc=exc)
            if upd is not None:
                health_updates.append(upd)
            last_exc = exc
    _persist_passive_tool_health(repo, health_updates)
    raise BrainProviderError(f"Alle Seiten-Extraktoren für '{url}' fehlgeschlagen: {last_exc}")


def _build_heuristic_answer(
    query: str,
    tier: BrainTier,
    citations: list[BrainCitation],
    pages: list[BrainFetchedPage],
    warnings: list[str],
    latency_ms: int,
    tools_executed: list[str],
) -> BrainGenerationResult:
    if not citations and not pages:
        text = (
            f"Für **{query}** ist aktuell kein aktives KI-Modell verfügbar "
            "und es konnten keine externen Webquellen abgerufen werden."
        )
    else:
        lines = [f"### Recherche-Ergebnis ({tier.value.upper()}) zu: {query}", ""]
        for c in citations[:8]:
            snip = f": {c.snippet}" if c.snippet else ""
            lines.append(f"- **[{c.title or c.url}]({c.url})** (`{c.source_tool}`){snip}")
        if pages:
            lines.extend(["", "#### Extrahierte Auszüge"])
            for p in pages[:2]:
                lines.append(f"**{p.title}** (`{p.extractor_used}`):\n{p.text_content[:500]}...")
        text = "\n".join(lines)
    return BrainGenerationResult(
        text=text,
        tier=tier,
        model_used="heuristic-extractive",
        citations=citations,
        tools_executed=tools_executed,
        pages_fetched=len(pages),
        latency_ms=latency_ms,
        warnings=warnings,
        used_heuristic_fallback=True,
    )


def execute_tiered_text(
    repo: BrainRepository,
    secrets: BrainSecrets,
    runner: UnifiedLLMRunner,
    prompt: str,
    *,
    tier: BrainTier = BrainTier.FAST,
    system_instruction: str = "",
    allow_paid: bool = False,
    on_action: Callable[[str], None] | None = None,
) -> BrainGenerationResult:
    t0 = time.monotonic()
    usage_state = repo.load_usage_quotas()
    warnings: list[str] = []
    for model in _ordered_models(repo, tier, allow_paid=allow_paid, usage_state=usage_state):
        if model.is_deep_agent or not secrets.has_model_key(model, allow_paid=allow_paid):
            continue
        if on_action:
            on_action(f"🤖 KI-Synthese via {model.name}...")
        try:
            text, citations = runner.run_text(
                model, prompt, system_instruction=system_instruction, allow_paid=allow_paid
            )
            if model.free_quota_limit > 0:
                record_quota_usage(repo, model_counts={model.id: 1})
            return BrainGenerationResult(
                text=text,
                tier=tier,
                model_used=model.id,
                citations=citations,
                latency_ms=int((time.monotonic() - t0) * 1000),
                warnings=warnings,
            )
        except Exception as exc:
            kind = "Quota 429" if is_quota_error(exc) else "Fehler"
            warnings.append(f"{model.id}: {kind} ({str(exc)[:120]})")
    raise BrainProviderError(f"Alle Modelle für Tier '{tier.value}' fehlgeschlagen: {'; '.join(warnings)}")


def execute_tiered_structured(
    repo: BrainRepository,
    secrets: BrainSecrets,
    runner: UnifiedLLMRunner,
    prompt: str,
    schema_cls: type[ModelT],
    *,
    tier: BrainTier = BrainTier.FAST,
    system_instruction: str = "",
    allow_paid: bool = False,
) -> tuple[ModelT, str]:
    usage_state = repo.load_usage_quotas()
    warnings: list[str] = []
    for model in _ordered_models(repo, tier, allow_paid=allow_paid, usage_state=usage_state):
        if not model.supports_structured_output or not secrets.has_model_key(model, allow_paid=allow_paid):
            continue
        try:
            parsed = runner.run_structured(
                model, prompt, schema_cls, system_instruction=system_instruction, allow_paid=allow_paid
            )
            if model.free_quota_limit > 0:
                record_quota_usage(repo, model_counts={model.id: 1})
            return parsed, model.id
        except Exception as exc:
            warnings.append(f"{model.id}: {str(exc)[:120]}")
    raise BrainProviderError(f"Structured Output fehlgeschlagen: {'; '.join(warnings)}")


def execute_tiered_ask(
    repo: BrainRepository,
    secrets: BrainSecrets,
    runner: UnifiedLLMRunner,
    query: str,
    *,
    tier: BrainTier = BrainTier.STANDARD,
    language: str = "de",
    system_instruction: str = "",
    allow_paid: bool = False,
    use_maps: bool | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    on_action: Callable[[str], None] | None = None,
) -> BrainGenerationResult:
    """Vollständige Brain-Anfrage: Websuche (+ Geo-Smart Maps) -> Scraping -> KI-Synthese (0%-Standard)."""
    t0 = time.monotonic()
    seq = repo.get_sequence(tier)
    max_hits = 12 if tier == BrainTier.DEEP else (8 if tier == BrainTier.STANDARD else 5)
    search_res = execute_tiered_search(
        repo,
        secrets,
        query,
        tier,
        language=language,
        max_results=max_hits,
        runner=runner,
        allow_paid=allow_paid,
        use_maps=use_maps,
        latitude=latitude,
        longitude=longitude,
        on_action=on_action,
    )

    pages: list[BrainFetchedPage] = []
    if seq.enable_page_scraping and seq.max_pages_to_scrape > 0 and search_res.citations:
        for cite in search_res.citations[: seq.max_pages_to_scrape]:
            try:
                if on_action:
                    on_action(f"📄 Extrahiere Volltext: {cite.url[:55]}...")
                pages.append(execute_tiered_fetch_page(repo, secrets, cite.url, allow_paid=allow_paid))
            except Exception:  # noqa: S110
                pass

    sources_block = "\n".join(
        f"{i}. [{c.title}]({c.url}) ({c.source_tool}): {c.snippet}"
        for i, c in enumerate(search_res.citations[:12], start=1)
    )
    pages_block = "\n\n".join(f"### Quelle: {p.title} ({p.url})\n{p.text_content[:2200]}" for p in pages)
    full_prompt = (
        f"Anfrage ({tier.value.upper()}):\n{query}\n\n"
        f"Gefundene Quellen:\n{sources_block or 'Keine externen Suchtreffer.'}"
        + (f"\n\nExtrahierte Seiteninhalte:\n{pages_block}" if pages_block else "")
    )
    sys_msg = system_instruction or "Beantworte die Anfrage präzise, faktenbasiert und zitiere relevante Quellen als `[Titel](URL)`."

    usage_state = repo.load_usage_quotas()
    warnings: list[str] = [f"{k}: {v}" for k, v in search_res.tool_errors.items()]
    for model in _ordered_models(repo, tier, allow_paid=allow_paid, usage_state=usage_state):
        if not secrets.has_model_key(model, allow_paid=allow_paid):
            continue
        if model.is_deep_agent and tier != BrainTier.DEEP:
            continue
        if on_action:
            on_action(f"🤖 Synthese via {model.name}...")
        try:
            text, extra_cites = runner.run_text(
                model, full_prompt, system_instruction=sys_msg, allow_paid=allow_paid
            )
            if model.free_quota_limit > 0:
                record_quota_usage(repo, model_counts={model.id: 1})
            all_cites = list(search_res.citations)
            seen = {c.url for c in all_cites}
            for ec in extra_cites:
                if ec.url not in seen:
                    all_cites.append(ec)
                    seen.add(ec.url)
            return BrainGenerationResult(
                text=text,
                tier=tier,
                model_used=model.id,
                citations=all_cites,
                tools_executed=search_res.tools_executed,
                pages_fetched=len(pages),
                latency_ms=int((time.monotonic() - t0) * 1000),
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"{model.id}: {str(exc)[:140]}")

    return _build_heuristic_answer(
        query, tier, search_res.citations, pages, warnings, int((time.monotonic() - t0) * 1000), search_res.tools_executed
    )
