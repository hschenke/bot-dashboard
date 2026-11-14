"""Öffentliche SearXNG-Instanz-Discovery & SSRF-sichere Verifikation via searx.space."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from ki_brain.enums import ToolCategory
from ki_brain.exceptions import BrainSSRFError
from ki_brain.models import BrainCitation, ToolSpec
from ki_brain.security import sanitize_display_url, validate_safe_external_url
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request

logger = logging.getLogger(__name__)

SEARX_SPACE_INSTANCES_URL = "https://searx.space/data/instances.json"
_FALLBACK_SEARXNG_HOSTS: tuple[str, ...] = (
    "https://searx.tiekoetter.com",
    "https://search.inetol.net",
    "https://searx.ononoki.org",
    "https://search.rhscz.eu",
    "https://search.hbubli.cc",
    "https://search.utilibre.org",
)


def _normalize_search_endpoint(raw_url: str) -> str | None:
    """Normalisiert und validiert HTTPS-Basis-URLs auf einen sicheren `/search`-Endpunkt."""
    clean = (raw_url or "").strip().rstrip("/")
    if not clean:
        return None
    parsed = urlparse(clean)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return None
    endpoint = clean if clean.endswith("/search") else f"{clean}/search"
    try:
        return validate_safe_external_url(endpoint, resolve_dns=False)
    except BrainSSRFError:
        return None


def parse_searx_space_instances(payload: dict[str, Any], max_candidates: int = 15) -> list[str]:
    """Extrahiert & sortiert sichere HTTPS-Normalnetz-Instanzen aus `searx.space/data/instances.json`."""
    raw_map = payload.get("instances")
    if not isinstance(raw_map, dict):
        return []
    ranked: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for base_url, meta in raw_map.items():
        if not isinstance(meta, dict):
            continue
        if str(meta.get("network_type") or "normal").lower() != "normal":
            continue
        http_info = meta.get("http") or {}
        status = http_info.get("status_code")
        if status is not None and int(status) != 200:
            continue
        endpoint = _normalize_search_endpoint(str(base_url))
        if not endpoint or endpoint in seen:
            continue
        timing = (meta.get("timing") or {}).get("search", {}).get("all", {})
        uptime = float(timing.get("uptime") or 0.0)
        mean_lat = float(timing.get("mean") or timing.get("median") or 9.9)
        if uptime and uptime < 90.0:
            continue
        seen.add(endpoint)
        ranked.append((uptime, -mean_lat, endpoint))
    ranked.sort(reverse=True)
    return [item[2] for item in ranked[:max_candidates]]


def probe_searxng_json_endpoint(
    endpoint_url: str,
    query: str = "Python",
    timeout_seconds: float = 4.5,
    *,
    http_request_fn: Any = None,
) -> bool:
    """Prüft live, ob eine SearXNG-Instanz `?format=json` aktiv hat und Treffer liefert."""
    safe_ep = _normalize_search_endpoint(endpoint_url)
    if not safe_ep:
        return False
    req_fn = http_request_fn or safe_http_request
    try:
        status, _, _, body = req_fn(
            "GET",
            safe_ep,
            params={"q": query, "format": "json"},
            headers={**API_HEADERS, "Accept": "application/json"},
            timeout_seconds=timeout_seconds,
            max_bytes=250_000,
        )
        if status != 200 or not body:
            return False
        data = json.loads(body.decode("utf-8", errors="replace"))
        for row in data.get("results", []):
            if (
                isinstance(row, dict)
                and row.get("url")
                and row.get("title")
                and sanitize_display_url(str(row["url"])) != "#"
            ):
                return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("SearXNG-Probe fehlgeschlagen (%s): %s", safe_ep, exc)
    return False


def discover_verified_searxng_pool(
    max_verify: int = 10,
    max_keep: int = 4,
    *,
    http_request_fn: Any = None,
) -> list[str]:
    """Lädt öffentliche Instanzen von `searx.space` und prüft `?format=json` live."""
    req_fn = http_request_fn or safe_http_request
    candidates: list[str] = []
    searx_space_reachable = False
    try:
        status, _, _, body = req_fn(
            "GET",
            SEARX_SPACE_INSTANCES_URL,
            headers={**API_HEADERS, "Accept": "application/json"},
            timeout_seconds=5.5,
            max_bytes=2_000_000,
        )
        if status == 200 and body:
            searx_space_reachable = True
            data = json.loads(body.decode("utf-8", errors="replace"))
            candidates = parse_searx_space_instances(data, max_candidates=max_verify)
        elif status == 404 and http_request_fn is not None:
            return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("searx.space Abruf fehlgeschlagen: %s", exc)

    if not candidates and not searx_space_reachable:
        candidates = [ep for host in _FALLBACK_SEARXNG_HOSTS if (ep := _normalize_search_endpoint(host))]

    verified: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda ep: probe_searxng_json_endpoint(ep, http_request_fn=req_fn),
                candidates[:max_verify],
            )
        )
    for ep, ok in zip(candidates[:max_verify], results, strict=True):
        if ok:
            verified.append(ep)
            if len(verified) >= max_keep:
                break
    return verified


def build_searxng_tool(verified_endpoints: list[str]) -> ToolSpec:
    """Erstellt ein fertiges `ToolSpec` für verifizierte `searx.space`-Instanzen inkl. Fallback-Pool."""
    primary = verified_endpoints[0]
    fallbacks = verified_endpoints[1:]
    return ToolSpec(
        id="searxng_public",
        name="SearXNG Public Instances (searx.space)",
        category=ToolCategory.WEB_SEARCH,
        executor_type="searxng_search",
        endpoint_url=primary,
        requires_api_key=False,
        origin_apps=["searx.space"],
        timeout_seconds=6.0,
        max_results=6,
        source="auto_discovered",
        config={"fallback_endpoints": fallbacks, "probe_query": "Python"},
        description=(
            f"Dezentrale Metasuchmaschine über searx.space ({len(verified_endpoints)} verifizierte JSON-Instanzen im Pool)."
        ),
    )


def parse_searxng_results(raw_bytes: bytes, source_tool: str, max_results: int) -> list[BrainCitation]:
    """Parst SearXNG JSON-Suchergebnisse sicher in normalisierte Zitate."""
    data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for row in data.get("results", []):
        if not isinstance(row, dict):
            continue
        safe = sanitize_display_url(str(row.get("url") or ""))
        title = str(row.get("title") or "").strip()[:180]
        if safe == "#" or not title or safe in seen:
            continue
        seen.add(safe)
        snippet = str(row.get("content") or row.get("snippet") or "").strip()[:320]
        results.append(BrainCitation(url=safe, title=title, snippet=snippet, source_tool=source_tool))
        if len(results) >= max_results:
            break
    return results
