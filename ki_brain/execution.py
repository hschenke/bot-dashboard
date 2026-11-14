"""Stufengesteuerte Ausführungs-Pipeline für Websearch, Extraktion & KI-Synthese."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from ki_brain.enums import BrainTier
from ki_brain.exceptions import BrainProviderError, BrainSSRFError
from ki_brain.llm_runner import UnifiedLLMRunner, is_quota_error
from ki_brain.models import (
    AIModelSpec,
    BrainCitation,
    BrainFetchedPage,
    BrainGenerationResult,
    BrainSearchResult,
    BrainSecrets,
    ToolSpec,
)
from ki_brain.storage import BrainRepository
from ki_brain.tools.registry import (
    execute_extractor_tool,
    execute_search_tool,
    is_extractor_capable,
    is_search_capable,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _ordered_models(repo: BrainRepository, tier: BrainTier) -> list[AIModelSpec]:
    enabled = {m.id: m for m in repo.list_models(enabled_only=True)}
    seq = repo.get_sequence(tier)
    ordered = [enabled[mid] for mid in seq.model_sequence if mid in enabled]
    seen = {m.id for m in ordered}
    for m in enabled.values():
        if m.id not in seen and (tier == BrainTier.DEEP or not m.is_deep_agent):
            ordered.append(m)
    return ordered


def _ordered_tools(repo: BrainRepository, seq_ids: list[str], capability_fn: Callable[[ToolSpec], bool]) -> list[ToolSpec]:
    enabled = {t.id: t for t in repo.list_tools(enabled_only=True) if capability_fn(t)}
    ordered = [enabled[tid] for tid in seq_ids if tid in enabled]
    seen = {t.id for t in ordered}
    for t in enabled.values():
        if t.id not in seen:
            ordered.append(t)
    return ordered


def execute_tiered_search(
    repo: BrainRepository,
    secrets: BrainSecrets,
    query: str,
    tier: BrainTier = BrainTier.STANDARD,
    *,
    language: str = "de",
    max_results: int = 8,
    on_action: Callable[[str], None] | None = None,
) -> BrainSearchResult:
    """Durchsucht Tools exakt in der für `tier` optimierten Sequenz inkl. Fallback."""
    t0 = time.monotonic()
    clean_q = " ".join((query or "").split())[:240]
    if not clean_q:
        return BrainSearchResult(query="", tier=tier, latency_ms=0)

    seq = repo.get_sequence(tier)
    candidates = _ordered_tools(repo, seq.search_tool_sequence, is_search_capable)
    citations: list[BrainCitation] = []
    seen_urls: set[str] = set()
    executed: list[str] = []
    errors: dict[str, str] = {}
    tools_with_hits = 0

    for tool in candidates:
        if tools_with_hits >= seq.max_search_tools:
            break
        if tool.requires_api_key and not secrets.has_key(tool.api_key_env):
            continue
        if on_action:
            on_action(f"🔎 Suche via {tool.name}...")
        executed.append(tool.id)
        try:
            hits = execute_search_tool(tool, clean_q, language=language, secrets=secrets, limit=tool.max_results)
            added_here = 0
            for hit in hits:
                if hit.url not in seen_urls:
                    seen_urls.add(hit.url)
                    citations.append(hit)
                    added_here += 1
            if added_here > 0:
                tools_with_hits += 1
        except Exception as exc:
            errors[tool.id] = str(exc)[:220]

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
) -> BrainFetchedPage:
    """Extrahiert Seiteninhalt über die priorisierte Extractor-Sequenz."""
    seq = repo.get_sequence(BrainTier.DEEP)
    candidates = _ordered_tools(repo, seq.extractor_tool_sequence, is_extractor_capable)
    last_exc: Exception | None = None
    for tool in candidates:
        if tool.requires_api_key and not secrets.has_key(tool.api_key_env):
            continue
        try:
            return execute_extractor_tool(tool, url, secrets=secrets)
        except BrainSSRFError:
            raise
        except Exception as exc:
            last_exc = exc
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
    on_action: Callable[[str], None] | None = None,
) -> BrainGenerationResult:
    t0 = time.monotonic()
    warnings: list[str] = []
    for model in _ordered_models(repo, tier):
        if model.is_deep_agent or (model.requires_api_key and not secrets.has_key(model.api_key_env)):
            continue
        if on_action:
            on_action(f"🤖 KI-Synthese via {model.name}...")
        try:
            text, citations = runner.run_text(model, prompt, system_instruction=system_instruction)
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
) -> tuple[ModelT, str]:
    warnings: list[str] = []
    for model in _ordered_models(repo, tier):
        if not model.supports_structured_output or (model.requires_api_key and not secrets.has_key(model.api_key_env)):
            continue
        try:
            parsed = runner.run_structured(model, prompt, schema_cls, system_instruction=system_instruction)
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
    on_action: Callable[[str], None] | None = None,
) -> BrainGenerationResult:
    """Vollständige Brain-Anfrage: Websuche -> optionaler Seitenabruf -> KI-Synthese -> Heuristik-Fallback."""
    t0 = time.monotonic()
    seq = repo.get_sequence(tier)
    max_hits = 12 if tier == BrainTier.DEEP else (8 if tier == BrainTier.STANDARD else 5)
    search_res = execute_tiered_search(repo, secrets, query, tier, language=language, max_results=max_hits, on_action=on_action)

    pages: list[BrainFetchedPage] = []
    if seq.enable_page_scraping and seq.max_pages_to_scrape > 0 and search_res.citations:
        for cite in search_res.citations[: seq.max_pages_to_scrape]:
            try:
                if on_action:
                    on_action(f"📄 Extrahiere Volltext: {cite.url[:55]}...")
                pages.append(execute_tiered_fetch_page(repo, secrets, cite.url))
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

    warnings: list[str] = [f"{k}: {v}" for k, v in search_res.tool_errors.items()]
    for model in _ordered_models(repo, tier):
        if model.requires_api_key and not secrets.has_key(model.api_key_env):
            continue
        if model.is_deep_agent and tier != BrainTier.DEEP:
            continue
        if on_action:
            on_action(f"🤖 Synthese via {model.name}...")
        try:
            text, extra_cites = runner.run_text(model, full_prompt, system_instruction=sys_msg)
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
