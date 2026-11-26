"""Zentrale Tool-Ausführungs-Registry für Websearch, Wissen & Seitenextraktion."""

from __future__ import annotations

from typing import Any

from ai_brain.enums import ToolCategory
from ai_brain.exceptions import BrainProviderError
from ai_brain.models import AIModelSpec, BrainCitation, BrainFetchedPage, BrainSecrets, ToolSpec
from ai_brain.tools.extractors import (
    execute_custom_json_api,
    execute_custom_rss_feed,
    fetch_direct_http,
    fetch_via_browserbase,
    fetch_via_firecrawl,
    fetch_via_jina_reader,
    query_open_meteo,
)
from ai_brain.tools.search_apis import (
    search_brave,
    search_firecrawl,
    search_searxng,
    search_serper,
    search_tavily,
)
from ai_brain.tools.search_knowledge import (
    search_arxiv,
    search_hackernews,
    search_reddit,
    search_wikidata,
    search_wikipedia,
)
from ai_brain.tools.search_web import (
    search_duckduckgo_html,
    search_duckduckgo_instant,
    search_duckduckgo_lite,
    search_google_news_rss,
    search_yahoo,
)

_AI_SEARCH_EXECUTORS = frozenset({"gemini_google_search", "gemini_google_maps"})


def is_search_capable(tool: ToolSpec) -> bool:
    """Gibt an, ob ein Tool Suchanfragen mit Zitaten beantwortet."""
    if tool.category in {
        ToolCategory.WEB_SEARCH,
        ToolCategory.NEWS_SEARCH,
        ToolCategory.COMMUNITY_SEARCH,
        ToolCategory.ACADEMIC_SEARCH,
        ToolCategory.REALTIME_API,
    }:
        return True
    return tool.category == ToolCategory.AI_GROUNDING and tool.executor_type.strip().lower() in _AI_SEARCH_EXECUTORS


def is_extractor_capable(tool: ToolSpec) -> bool:
    """Gibt an, ob ein Tool Webseiten-Volltexte extrahiert."""
    return tool.category == ToolCategory.PAGE_EXTRACTOR


def _execute_ai_grounding_search(
    tool: ToolSpec,
    query: str,
    *,
    secrets: BrainSecrets,
    max_res: int,
    runner: Any = None,
    grounding_model: AIModelSpec | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    allow_paid: bool = False,
) -> list[BrainCitation]:
    from ai_brain.llm_runner import UnifiedLLMRunner

    active_runner = runner if (runner is not None and hasattr(runner, "run_text")) else UnifiedLLMRunner(secrets)
    model = grounding_model or AIModelSpec(
        id="gemini-3.5-flash-lite", name="Gemini 3.5 Flash-Lite", model_id="gemini-3.5-flash-lite"
    )
    etype = tool.executor_type.strip().lower()
    if etype == "gemini_google_maps":
        tool_spec: dict[str, Any] = {"type": "google_maps"}
        if latitude is not None and longitude is not None:
            tool_spec["latitude"] = float(latitude)
            tool_spec["longitude"] = float(longitude)
        prompt = f"Finde passende Orte und Details auf Google Maps zu: {query}"
    else:
        tool_spec = {"type": "google_search"}
        prompt = f"Suche aktuell im Web nach: {query}"

    _, citations = active_runner.run_text(
        model,
        prompt,
        timeout_seconds=min(20.0, max(5.0, tool.timeout_seconds)),
        tools=[tool_spec],
        allow_paid=allow_paid,
    )
    return citations[:max_res]


def execute_search_tool(
    tool: ToolSpec,
    query: str,
    *,
    language: str = "de",
    secrets: BrainSecrets | None = None,
    limit: int | None = None,
    runner: Any = None,
    grounding_model: AIModelSpec | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    allow_paid: bool = False,
) -> list[BrainCitation]:
    """Führt ein beliebiges konfiguriertes Such- oder Custom-Tool typsicher aus."""
    sec = secrets or BrainSecrets.from_env()
    max_res = max(1, min(limit or tool.max_results, 20))
    etype = tool.executor_type.strip().lower()
    tmo = tool.timeout_seconds

    if etype in _AI_SEARCH_EXECUTORS:
        return _execute_ai_grounding_search(
            tool, query, secrets=sec, max_res=max_res, runner=runner,
            grounding_model=grounding_model, latitude=latitude, longitude=longitude, allow_paid=allow_paid,
        )
    if etype == "brave_search":
        return search_brave(query, secrets=sec, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "tavily_search":
        return search_tavily(query, secrets=sec, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "serper_search":
        return search_serper(
            query, secrets=sec, endpoint_url=tool.endpoint_url, language=language, max_results=max_res, timeout_seconds=tmo
        )
    if etype == "firecrawl_search":
        return search_firecrawl(
            query, secrets=sec, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo
        )
    if etype == "searxng_search":
        return search_searxng(tool, query, language=language, max_results=max_res)
    if etype == "yahoo_search":
        return search_yahoo(
            query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo, language=language
        )
    if etype == "duckduckgo_html":
        return search_duckduckgo_html(query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "duckduckgo_lite":
        return search_duckduckgo_lite(query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "duckduckgo_instant":
        return search_duckduckgo_instant(query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "google_news_rss":
        return search_google_news_rss(
            query, endpoint_url=tool.endpoint_url, language=language, max_results=max_res, timeout_seconds=tmo
        )
    if etype == "wikipedia_api":
        return search_wikipedia(query, language=language, max_results=max_res, timeout_seconds=tmo)
    if etype == "wikidata_api":
        return search_wikidata(query, language=language, max_results=max_res, timeout_seconds=tmo)
    if etype == "hackernews_algolia":
        return search_hackernews(query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "reddit_search":
        return search_reddit(query, secrets=sec, max_results=max_res, timeout_seconds=tmo)
    if etype == "arxiv_api":
        return search_arxiv(query, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "open_meteo":
        return query_open_meteo(query, timeout_seconds=tmo)
    if etype == "custom_json_api":
        return execute_custom_json_api(tool, query, secrets=sec, limit=max_res)
    if etype == "custom_rss_feed":
        return execute_custom_rss_feed(tool, query, limit=max_res)

    raise BrainProviderError(f"Unbekannter Such-Executor-Typ '{tool.executor_type}'.")


def execute_extractor_tool(
    tool: ToolSpec,
    url: str,
    *,
    secrets: BrainSecrets | None = None,
) -> BrainFetchedPage:
    """Extrahiert den bereinigten Seiteninhalt einer URL via passendem Extractor-Tool."""
    sec = secrets or BrainSecrets.from_env()
    etype = tool.executor_type.strip().lower()
    tmo = tool.timeout_seconds

    if etype == "direct_http_fetch":
        return fetch_direct_http(url, timeout_seconds=tmo)
    if etype == "jina_reader":
        return fetch_via_jina_reader(url, timeout_seconds=tmo)
    if etype == "firecrawl_scrape":
        return fetch_via_firecrawl(url, secrets=sec, endpoint_url=tool.endpoint_url, timeout_seconds=tmo)
    if etype == "browserbase_fetch":
        return fetch_via_browserbase(url, secrets=sec, timeout_seconds=tmo)

    raise BrainProviderError(f"Unbekannter Extractor-Typ '{tool.executor_type}'.")
