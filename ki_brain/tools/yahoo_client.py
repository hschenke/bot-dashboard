"""Resilienter Yahoo-Websearch-Client mit Session/YBV-Cookie-Handling, Dual-DOM-Parser & Endpunkt-Fallback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

from ki_brain.exceptions import BrainProviderError, BrainSSRFError
from ki_brain.models import BrainCitation
from ki_brain.tools.http_utils import BROWSER_HEADERS, extract_redirect_target

YAHOO_FALLBACK_ENDPOINTS: tuple[str, ...] = (
    "https://de.search.yahoo.com/search",
    "https://search.yahoo.com/search",
    "https://ch.search.yahoo.com/search",
    "https://at.search.yahoo.com/search",
)
_YBV_TTL_SECONDS = 43_200.0
_YBV_LOCK = threading.Lock()
_YBV_CACHE: dict[str, tuple[str, float]] = {}


def _get_cached_ybv(host: str) -> str:
    if not host:
        return ""
    now = time.monotonic()
    with _YBV_LOCK:
        entry = _YBV_CACHE.get(host)
        if not entry:
            return ""
        val, expires_at = entry
        if expires_at <= now:
            _YBV_CACHE.pop(host, None)
            return ""
        return val


def _store_cached_ybv(host: str, value: str) -> None:
    clean = (value or "").strip()
    if not host or not clean:
        return
    with _YBV_LOCK:
        if len(_YBV_CACHE) >= 32 and host not in _YBV_CACHE:
            _YBV_CACHE.pop(next(iter(_YBV_CACHE)), None)
        _YBV_CACHE[host] = (clean, time.monotonic() + _YBV_TTL_SECONDS)


def _evict_cached_ybv(host: str) -> None:
    if not host:
        return
    with _YBV_LOCK:
        _YBV_CACHE.pop(host, None)


def _build_yahoo_headers(language: str) -> dict[str, str]:
    lang_hdr = (
        "en-US,en;q=0.9"
        if language.strip().lower() == "en"
        else "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
    )
    return {
        **BROWSER_HEADERS,
        "Accept-Language": lang_hdr,
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _build_yahoo_cookies(host: str, language: str) -> dict[str, str]:
    vl = "lang_en" if language.strip().lower() == "en" else "lang_de"
    cookies: dict[str, str] = {"sB": f"v=1&vm=p&fl=1&vl={vl}&pn=10&rw=new&userset=1"}
    ybv = _get_cached_ybv(host)
    if ybv:
        cookies["YBV"] = ybv
    return cookies


def _extract_clean_title(anchor: Tag) -> str:
    aria = str(anchor.get("aria-label") or "").strip()
    if aria:
        return aria[:180]
    inner_h3 = anchor.select_one("h3")
    if inner_h3:
        h3_txt = inner_h3.get_text(" ", strip=True)
        if h3_txt:
            return h3_txt[:180]
    direct = " ".join(s.strip() for s in anchor.find_all(string=True, recursive=False) if s.strip())
    if direct:
        return direct[:180]
    return anchor.get_text(" ", strip=True)[:180]


def parse_yahoo_html(body: bytes, max_results: int = 6) -> list[BrainCitation]:
    """Parst organische Yahoo-Treffer für DE- (`h3 a`) und US-DOM-Layouts (`a:has(h3)`)."""
    soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "html.parser")
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for node in soup.select("div.algo"):
        anchor = (
            node.select_one("div.compTitle a")
            or node.select_one("h3 a")
            or node.select_one("a:has(h3)")
        )
        if not anchor:
            continue
        target = extract_redirect_target(str(anchor.get("href") or ""))
        if not target or target in seen:
            continue
        seen.add(target)
        comp = node.select_one(".compText, .compText p, .fc-falcon")
        snippet = comp.get_text(" ", strip=True)[:320] if comp else ""
        title = _extract_clean_title(anchor)
        if title:
            results.append(BrainCitation(url=target, title=title, snippet=snippet, source_tool="yahoo_search"))
        if len(results) >= max_results:
            break
    return results


def _candidate_endpoints(endpoint_url: str) -> list[str]:
    primary = (endpoint_url or "").strip() or YAHOO_FALLBACK_ENDPOINTS[0]
    candidates = [primary]
    for fallback in YAHOO_FALLBACK_ENDPOINTS:
        if fallback != primary:
            candidates.append(fallback)
        if len(candidates) >= 3:
            break
    return candidates


def execute_yahoo_search(
    query: str,
    *,
    endpoint_url: str = YAHOO_FALLBACK_ENDPOINTS[0],
    max_results: int = 6,
    timeout_seconds: float = 6.0,
    language: str = "de",
    http_request_fn: Callable[..., tuple[int, str, str, bytes]],
) -> list[BrainCitation]:
    """Führt Yahoo-Suche mit Cookie-Session-Handshake, YBV-Cache & Endpunkt-Rotation aus."""
    clean_q = " ".join((query or "").split())[:200]
    if not clean_q:
        return []

    headers = _build_yahoo_headers(language)
    got_http_response = False
    last_err: Exception | None = None

    for ep in _candidate_endpoints(endpoint_url):
        host = (urlparse(ep).hostname or "").lower()
        init_cookies = _build_yahoo_cookies(host, language)
        collected: dict[str, Any] = {}
        try:
            status, _, _, body = http_request_fn(
                "GET",
                ep,
                params={"p": clean_q},
                headers=headers,
                cookies=init_cookies,
                collected_cookies=collected,
                timeout_seconds=timeout_seconds,
                max_bytes=550_000,
            )
        except BrainSSRFError:
            raise
        except (BrainProviderError, requests.RequestException) as exc:
            _evict_cached_ybv(host)
            last_err = exc
            continue

        got_http_response = True
        if status == 429:
            _evict_cached_ybv(host)
            return []
        if status == 200 and body:
            hits = parse_yahoo_html(body, max_results=max_results)
            if hits:
                _store_cached_ybv(host, str(collected.get("YBV") or ""))
                return hits
        _evict_cached_ybv(host)

    if last_err is not None and not got_http_response:
        raise last_err
    return []
