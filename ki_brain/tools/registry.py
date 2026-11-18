"""Zentrale Tool-Ausführungs-Registry für Websearch, Wissen & Seitenextraktion."""

from __future__ import annotations

from ki_brain.enums import ToolCategory
from ki_brain.exceptions import BrainProviderError
from ki_brain.models import BrainCitation, BrainFetchedPage, BrainSecrets, ToolSpec
from ki_brain.tools.extractors import (
    execute_custom_json_api,
    execute_custom_rss_feed,
    fetch_direct_http,
    fetch_via_browserbase,
    fetch_via_jina_reader,
    query_open_meteo,
)
from ki_brain.tools.search_apis import (
    search_brave,
    search_searxng,
    search_serper,
    search_tavily,
)
from ki_brain.tools.search_knowledge import (
    search_arxiv,
    search_hackernews,
    search_reddit,
    search_wikidata,
    search_wikipedia,
)
from ki_brain.tools.search_web import (
    search_duckduckgo_html,
    search_duckduckgo_instant,
    search_duckduckgo_lite,
    search_google_news_rss,
    search_yahoo,
)


def is_search_capable(tool: ToolSpec) -> bool:
    """Gibt an, ob ein Tool Suchanfragen mit Zitaten beantwortet."""
    return tool.category in {
        ToolCategory.WEB_SEARCH,
        ToolCategory.NEWS_SEARCH,
        ToolCategory.COMMUNITY_SEARCH,
        ToolCategory.ACADEMIC_SEARCH,
        ToolCategory.REALTIME_API,
    }


def is_extractor_capable(tool: ToolSpec) -> bool:
    """Gibt an, ob ein Tool Webseiten-Volltexte extrahiert."""
    return tool.category == ToolCategory.PAGE_EXTRACTOR


def execute_search_tool(
    tool: ToolSpec,
    query: str,
    *,
    language: str = "de",
    secrets: BrainSecrets | None = None,
    limit: int | None = None,
) -> list[BrainCitation]:
    """Führt ein beliebiges konfiguriertes Such- oder Custom-Tool typsicher aus."""
    sec = secrets or BrainSecrets.from_env()
    max_res = max(1, min(limit or tool.max_results, 20))
    etype = tool.executor_type.strip().lower()
    tmo = tool.timeout_seconds

    if etype == "brave_search":
        return search_brave(query, secrets=sec, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "tavily_search":
        return search_tavily(query, secrets=sec, endpoint_url=tool.endpoint_url, max_results=max_res, timeout_seconds=tmo)
    if etype == "serper_search":
        return search_serper(
            query, secrets=sec, endpoint_url=tool.endpoint_url, language=language, max_results=max_res, timeout_seconds=tmo
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
    if etype == "browserbase_fetch":
        return fetch_via_browserbase(url, secrets=sec, timeout_seconds=tmo)

    raise BrainProviderError(f"Unbekannter Extractor-Typ '{tool.executor_type}'.")
