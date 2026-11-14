"""API-basierte Web-Suchmaschinen (Brave Search, Tavily, Serper.dev) & SearXNG-Rotation."""

from __future__ import annotations

import json
import logging

from ki_brain.exceptions import BrainProviderError
from ki_brain.models import BrainCitation, BrainSecrets, ToolSpec
from ki_brain.security import sanitize_display_url
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request
from ki_brain.tools.searxng import parse_searxng_results

logger = logging.getLogger(__name__)


def _raise_api_error(provider: str, status: int, body: bytes) -> None:
    snippet = body.decode("utf-8", errors="replace")[:180].strip()
    if status == 429:
        raise BrainProviderError(f"{provider} HTTP 429 Rate-Limit/Quota erschöpft: {snippet}")
    raise BrainProviderError(f"{provider} HTTP {status}: {snippet}")


def search_brave(
    query: str,
    secrets: BrainSecrets,
    *,
    endpoint_url: str = "https://api.search.brave.com/res/v1/web/search",
    max_results: int = 6,
    timeout_seconds: float = 6.0,
) -> list[BrainCitation]:
    """Führt eine Websuche über die offizielle Brave Search API (`X-Subscription-Token`) aus."""
    api_key = secrets.resolve_key("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise BrainProviderError("BRAVE_SEARCH_API_KEY fehlt.")
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://api.search.brave.com/res/v1/web/search",
        params={"q": clean_q, "count": min(max(1, max_results), 20)},
        headers={**API_HEADERS, "Accept": "application/json", "X-Subscription-Token": api_key},
        timeout_seconds=timeout_seconds,
        max_bytes=450_000,
    )
    if status != 200:
        _raise_api_error("Brave Search", status, body)
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for item in payload.get("web", {}).get("results", []):
        safe = sanitize_display_url(str(item.get("url") or ""))
        title = str(item.get("title") or "").strip()[:180]
        if safe != "#" and title:
            results.append(
                BrainCitation(
                    url=safe,
                    title=title,
                    snippet=str(item.get("description") or "").strip()[:320],
                    source_tool="brave_search",
                )
            )
            if len(results) >= max_results:
                break
    return results


def search_tavily(
    query: str,
    secrets: BrainSecrets,
    *,
    endpoint_url: str = "https://api.tavily.com/search",
    max_results: int = 6,
    timeout_seconds: float = 7.0,
) -> list[BrainCitation]:
    """Führt eine KI-optimierte Suche über Tavily (`api.tavily.com/search`) aus."""
    api_key = secrets.resolve_key("TAVILY_API_KEY")
    if not api_key:
        raise BrainProviderError("TAVILY_API_KEY fehlt.")
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "POST",
        endpoint_url or "https://api.tavily.com/search",
        json_body={"api_key": api_key, "query": clean_q, "max_results": min(max(1, max_results), 20), "search_depth": "basic"},
        headers={**API_HEADERS, "Content-Type": "application/json", "Accept": "application/json"},
        timeout_seconds=timeout_seconds,
        max_bytes=450_000,
    )
    if status != 200:
        _raise_api_error("Tavily", status, body)
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for item in payload.get("results", []):
        safe = sanitize_display_url(str(item.get("url") or ""))
        title = str(item.get("title") or "").strip()[:180]
        if safe != "#" and title:
            results.append(
                BrainCitation(
                    url=safe,
                    title=title,
                    snippet=str(item.get("content") or "").strip()[:340],
                    source_tool="tavily_search",
                )
            )
            if len(results) >= max_results:
                break
    return results


def search_serper(
    query: str,
    secrets: BrainSecrets,
    *,
    endpoint_url: str = "https://google.serper.dev/search",
    language: str = "de",
    max_results: int = 6,
    timeout_seconds: float = 6.0,
) -> list[BrainCitation]:
    """Führt eine Google-SERP-Suche über Serper.dev (`X-API-KEY`) aus."""
    api_key = secrets.resolve_key("SERPER_API_KEY")
    if not api_key:
        raise BrainProviderError("SERPER_API_KEY fehlt.")
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    hl, gl = ("en", "us") if language.lower() == "en" else ("de", "de")
    status, _, _, body = safe_http_request(
        "POST",
        endpoint_url or "https://google.serper.dev/search",
        json_body={"q": clean_q, "num": min(max(1, max_results), 20), "hl": hl, "gl": gl},
        headers={**API_HEADERS, "Content-Type": "application/json", "Accept": "application/json", "X-API-KEY": api_key},
        timeout_seconds=timeout_seconds,
        max_bytes=450_000,
    )
    if status != 200:
        _raise_api_error("Serper", status, body)
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for item in payload.get("organic", []):
        safe = sanitize_display_url(str(item.get("link") or item.get("url") or ""))
        title = str(item.get("title") or "").strip()[:180]
        if safe != "#" and title:
            results.append(
                BrainCitation(
                    url=safe,
                    title=title,
                    snippet=str(item.get("snippet") or "").strip()[:320],
                    source_tool="serper_search",
                )
            )
            if len(results) >= max_results:
                break
    return results


def search_searxng(
    tool: ToolSpec,
    query: str,
    *,
    language: str = "de",
    max_results: int = 6,
) -> list[BrainCitation]:
    """Durchsucht einen rotierenden Pool öffentlicher `searx.space`-Instanzen (key-frei)."""
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    candidates = [tool.endpoint_url, *(tool.config.get("fallback_endpoints") or [])]
    endpoints = [str(u).strip() for u in dict.fromkeys(candidates) if str(u).strip()]
    lang_code = "en-US" if language.lower() == "en" else "de-DE"

    for ep in endpoints:
        try:
            status, _, _, body = safe_http_request(
                "GET",
                ep,
                params={"q": clean_q, "format": "json", "language": lang_code},
                headers={**API_HEADERS, "Accept": "application/json"},
                timeout_seconds=tool.timeout_seconds,
                max_bytes=450_000,
            )
            if status == 200 and body:
                hits = parse_searxng_results(body, source_tool=tool.id, max_results=max_results)
                if hits:
                    return hits
        except Exception as exc:  # noqa: BLE001
            logger.debug("SearXNG Endpoint '%s' fehlgeschlagen: %s", ep, exc)
    return []
