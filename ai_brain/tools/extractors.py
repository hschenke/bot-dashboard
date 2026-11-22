"""Webseiten-Extraktoren, Echtzeit-APIs und dynamische Custom-Tool-Ausführer."""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import urlparse

import defusedxml.ElementTree as ET
from bs4 import BeautifulSoup

from ai_brain.exceptions import BrainProviderError
from ai_brain.models import BrainCitation, BrainFetchedPage, BrainSecrets, ToolSpec
from ai_brain.security import sanitize_display_url, validate_safe_external_url
from ai_brain.tools.http_utils import API_HEADERS, BROWSER_HEADERS, safe_http_request

_ALLOWED_TEXT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "nav", "footer", "header", "aside", "form")
_BLANK_LINES = re.compile(r"\n{3,}")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def extract_clean_text_from_html(html_text: str, max_chars: int = 8000) -> tuple[str, str]:
    """Entfernt Navigation/Skripte und liefert `(Titel, bereinigter_Text)`."""
    soup = BeautifulSoup(html_text or "", "html.parser")
    title = soup.title.get_text(" ", strip=True)[:240] if soup.title else ""
    for node in soup(_NOISE_TAGS):
        node.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    non_empty = [line for line in lines if len(line) > 1]
    clean = _BLANK_LINES.sub("\n\n", "\n".join(non_empty)).strip()
    return title, clean[:max_chars]


def fetch_direct_http(url: str, timeout_seconds: float = 7.0, max_chars: int = 8000) -> BrainFetchedPage:
    """Lädt eine Webseite direkt mit Redirect-SSRF-Prüfung und HTML-Textextraktion."""
    t0 = time.monotonic()
    status, final_url, ctype, body = safe_http_request(
        "GET", url, headers=BROWSER_HEADERS, timeout_seconds=timeout_seconds, max_bytes=1_500_000
    )
    if status != 200:
        raise BrainProviderError(f"HTTP-Status {status} bei '{url}'.")
    if ctype and not any(allowed in ctype for allowed in _ALLOWED_TEXT_TYPES):
        raise BrainProviderError(f"Nicht unterstützter Content-Type '{ctype}'.")
    title, text = extract_clean_text_from_html(body.decode("utf-8", errors="replace"), max_chars=max_chars)
    if not text:
        raise BrainProviderError(f"Kein Textinhalt auf '{final_url}' gefunden.")
    return BrainFetchedPage(
        url=final_url,
        title=title or (urlparse(final_url).netloc or final_url),
        text_content=text,
        extractor_used="direct_http_fetch",
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def fetch_via_jina_reader(url: str, timeout_seconds: float = 7.0, max_chars: int = 8000) -> BrainFetchedPage:
    """Extrahiert sauberes Markdown über Jina AI Reader nach vorheriger SSRF-Prüfung."""
    t0 = time.monotonic()
    safe_target = validate_safe_external_url(url)
    endpoint = f"https://r.jina.ai/{safe_target}"
    status, _, _, body = safe_http_request(
        "GET",
        endpoint,
        headers={**API_HEADERS, "Accept": "text/markdown", "X-Return-Format": "markdown"},
        timeout_seconds=timeout_seconds,
        max_bytes=1_000_000,
    )
    if status != 200 or not body:
        raise BrainProviderError(f"Jina Reader lieferte HTTP {status}.")
    text = body.decode("utf-8", errors="replace").strip()[:max_chars]
    if not text:
        raise BrainProviderError("Jina Reader lieferte leeren Inhalt.")
    host = urlparse(safe_target).netloc or safe_target
    return BrainFetchedPage(
        url=safe_target,
        title=f"[Jina Reader] {host}",
        text_content=text,
        extractor_used="jina_reader",
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def fetch_via_browserbase(
    url: str,
    secrets: BrainSecrets,
    timeout_seconds: float = 9.0,
    max_chars: int = 8000,
) -> BrainFetchedPage:
    """Extrahiert JS-lastige Webseiten über die Browserbase Cloud Fetch API."""
    api_key = secrets.resolve_key("BROWSERBASE_API_KEY")
    if not api_key:
        raise BrainProviderError("BROWSERBASE_API_KEY fehlt.")
    t0 = time.monotonic()
    safe_target = validate_safe_external_url(url)
    payload: dict[str, Any] = {"url": safe_target}
    if secrets.browserbase_project_id:
        payload["projectId"] = secrets.browserbase_project_id
    status, _, _, body = safe_http_request(
        "POST",
        "https://api.browserbase.com/v1/fetch",
        json_body=payload,
        headers={**API_HEADERS, "X-BB-API-Key": api_key, "Content-Type": "application/json"},
        timeout_seconds=timeout_seconds,
        max_bytes=1_000_000,
    )
    if status not in {200, 201}:
        raise BrainProviderError(f"Browserbase HTTP {status}.")
    data = json.loads(body.decode("utf-8", errors="replace"))
    content = str(data.get("content") or data.get("markdown") or data.get("text") or "").strip()[:max_chars]
    if not content:
        raise BrainProviderError("Browserbase lieferte keinen verwertbaren Text.")
    host = urlparse(safe_target).netloc or safe_target
    return BrainFetchedPage(
        url=safe_target,
        title=f"[Browserbase] {host}",
        text_content=content,
        extractor_used="browserbase_fetch",
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


_BUILTIN_CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "berlin": (52.52, 13.405, "Berlin, Deutschland"),
    "münchen": (48.137, 11.576, "München, Deutschland"),
    "munich": (48.137, 11.576, "München, Deutschland"),
    "hamburg": (53.551, 9.994, "Hamburg, Deutschland"),
    "köln": (50.938, 6.960, "Köln, Deutschland"),
    "frankfurt": (50.111, 8.682, "Frankfurt, Deutschland"),
}


def query_open_meteo(query: str, timeout_seconds: float = 6.0) -> list[BrainCitation]:
    """Fragt Ortskoordinaten und Live-Wetterdaten über Open-Meteo ohne Key ab."""
    tokens = [tok.strip(" ,.?!:;") for tok in query.split() if len(tok.strip(" ,.?!:;")) >= 2]
    ignore = {"wetter", "temperatur", "heute", "aktuell", "weather", "forecast", "wie", "ist", "das", "in"}
    candidates = [t for t in tokens if t.lower() not in ignore]
    location = candidates[-1] if candidates else "Berlin"

    lat: float | None = None
    lon: float | None = None
    place = location
    try:
        st, _, _, gbody = safe_http_request(
            "GET",
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "de", "format": "json"},
            timeout_seconds=max(timeout_seconds, 5.5),
            max_bytes=100_000,
        )
        if st == 200 and gbody:
            geo = json.loads(gbody.decode("utf-8", errors="replace")).get("results") or []
            if geo:
                first = geo[0]
                lat, lon = float(first["latitude"]), float(first["longitude"])
                place = f"{first.get('name', location)}, {first.get('country', '')}".strip(", ")
    except Exception:  # noqa: S110
        pass

    if lat is None or lon is None:
        fallback = _BUILTIN_CITY_COORDS.get(location.lower())
        if not fallback:
            return []
        lat, lon, place = fallback

    st2, _, _, wbody = safe_http_request(
        "GET",
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m,precipitation", "timezone": "auto"},
        timeout_seconds=max(timeout_seconds, 5.5),
        max_bytes=100_000,
    )
    if st2 != 200 or not wbody:
        return []
    cur = json.loads(wbody.decode("utf-8", errors="replace")).get("current", {})
    snippet = (
        f"Temperatur: {cur.get('temperature_2m')} °C | "
        f"Wind: {cur.get('wind_speed_10m')} km/h | "
        f"Niederschlag: {cur.get('precipitation')} mm"
    )
    return [BrainCitation(url="https://open-meteo.com/", title=f"Open-Meteo Live: {place}", snippet=snippet, source_tool="open_meteo")]


def _resolve_dotted_path(payload: Any, path: str) -> Any:
    cur = payload
    for part in [p.strip() for p in path.split(".") if p.strip()]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def execute_custom_json_api(
    tool: ToolSpec,
    query: str,
    secrets: BrainSecrets,
    limit: int = 5,
) -> list[BrainCitation]:
    """Führt ein dynamisch hinzugefügtes JSON-Suchtool aus (inkl. verschachtelter Felder)."""
    if not tool.endpoint_url:
        raise BrainProviderError(f"Tool '{tool.id}' hat keine endpoint_url.")
    cfg = tool.config or {}
    q_param = str(cfg.get("query_param", "q")).strip()
    items_path = str(cfg.get("items_path") or "results")
    title_field = str(cfg.get("title_field") or "title")
    url_field = str(cfg.get("url_field") or "url")
    snippet_field = str(cfg.get("snippet_field") or "snippet")
    headers = dict(API_HEADERS)
    if tool.requires_api_key and tool.api_key_env:
        key_val = secrets.resolve_key(tool.api_key_env)
        if not key_val:
            raise BrainProviderError(f"API-Key '{tool.api_key_env}' fehlt.")
        key_header = str(cfg.get("api_key_header") or "Authorization")
        headers[key_header] = f"Bearer {key_val}" if key_header == "Authorization" else key_val
    req_params: dict[str, Any] = dict(cfg.get("extra_params") or {})
    if q_param:
        req_params[q_param] = " ".join(query.split())[:180]
    status, _, _, body = safe_http_request(
        "GET",
        tool.endpoint_url,
        params=req_params or None,
        headers=headers,
        timeout_seconds=tool.timeout_seconds,
        max_bytes=500_000,
    )
    if status != 200 or not body:
        return []
    raw_items = _resolve_dotted_path(json.loads(body.decode("utf-8", errors="replace")), items_path)
    if not isinstance(raw_items, list):
        return []
    results: list[BrainCitation] = []
    for entry in raw_items[:limit]:
        if not isinstance(entry, dict):
            continue
        raw_url = _resolve_dotted_path(entry, url_field) if "." in url_field else entry.get(url_field)
        raw_title = _resolve_dotted_path(entry, title_field) if "." in title_field else entry.get(title_field)
        raw_snip = _resolve_dotted_path(entry, snippet_field) if "." in snippet_field else entry.get(snippet_field)
        link = sanitize_display_url(str(raw_url or tool.endpoint_url))
        title = str(raw_title or "").strip()[:180]
        snip = str(raw_snip or "").strip()[:300]
        if link != "#" and title:
            results.append(BrainCitation(url=link, title=title, snippet=snip, source_tool=tool.id))
    return results


def execute_custom_rss_feed(tool: ToolSpec, query: str, limit: int = 5) -> list[BrainCitation]:
    """Liest einen dynamisch hinzugefügten RSS- oder Atom-Feed SSRF-sicher aus."""
    if not tool.endpoint_url:
        raise BrainProviderError(f"Tool '{tool.id}' hat keine endpoint_url.")
    status, _, _, body = safe_http_request(
        "GET", tool.endpoint_url, headers=BROWSER_HEADERS, timeout_seconds=tool.timeout_seconds, max_bytes=500_000
    )
    if status != 200 or not body:
        return []
    root = ET.fromstring(body.decode("utf-8", errors="replace"))
    tokens = [t.lower() for t in query.split() if len(t) >= 3]
    all_entries: list[BrainCitation] = []
    for node in root.findall(".//item"):
        link = sanitize_display_url((node.findtext("link") or "").strip())
        title = (node.findtext("title") or "").strip()[:180]
        desc = BeautifulSoup(node.findtext("description") or "", "html.parser").get_text(" ", strip=True)[:280]
        if link != "#" and title:
            all_entries.append(BrainCitation(url=link, title=title, snippet=desc, source_tool=tool.id))
    for entry in root.findall("atom:entry", _ATOM_NS):
        tnode = entry.find("atom:title", _ATOM_NS)
        lnode = entry.find("atom:link", _ATOM_NS)
        href = str(lnode.attrib.get("href") or "") if lnode is not None else ""
        link = sanitize_display_url(href or ((entry.findtext("atom:id", namespaces=_ATOM_NS) or "").strip()))
        title = (tnode.text or "").strip()[:180] if tnode is not None else ""
        if link != "#" and title:
            all_entries.append(BrainCitation(url=link, title=title, snippet="", source_tool=tool.id))
    matched = [c for c in all_entries if not tokens or any(tok in f"{c.title} {c.snippet}".lower() for tok in tokens)]
    return (matched or all_entries)[:limit]
