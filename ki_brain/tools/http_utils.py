"""Gemeinsame SSRF-sichere HTTP-, Parsing- und Probe-Diagnose-Hilfsfunktionen."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests

from ki_brain.exceptions import BrainProviderError
from ki_brain.security import (
    DnsResolver,
    read_bounded_response,
    sanitize_display_url,
    validate_safe_external_url,
)

logger = logging.getLogger(__name__)

MAX_REDIRECT_HOPS = 5
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
}
API_HEADERS: dict[str, str] = {
    "User-Agent": "KIWebsearchBrain/1.0 (+https://github.com/hschenke/bot-dashboard)",
    "Accept": "application/json,application/xml,text/plain;q=0.9,*/*;q=0.8",
}

_PROBE_TRACE_LOCAL = threading.local()
_STATUS_LABELS: dict[int, str] = {
    202: "Anti-Bot-Challenge / Leerantwort",
    401: "Auth / API-Key ungültig",
    403: "Zugriff / Bot-Schutz blockiert",
    404: "Endpunkt nicht gefunden",
    429: "Rate-Limit / Quota 429",
    500: "Serverfehler (500)",
    502: "Bad Gateway (502)",
    503: "Service Unavailable (503)",
}


@dataclass(frozen=True)
class HttpProbeEvent:
    """Protokolliert Status & Größe eines einzelnen HTTP-Probe-Aufrufs (ohne Secrets)."""

    host: str
    status_code: int
    body_bytes: int


def start_probe_trace() -> None:
    """Aktiviert thread-lokales HTTP-Diagnose-Tracing für den aktuellen Tool-Testlauf."""
    _PROBE_TRACE_LOCAL.events = []


def consume_probe_trace() -> list[HttpProbeEvent]:
    """Liest und leert alle gesammelten HTTP-Diagnose-Events des aktuellen Threads."""
    events = getattr(_PROBE_TRACE_LOCAL, "events", None)
    _PROBE_TRACE_LOCAL.events = None
    return list(events) if isinstance(events, list) else []


def record_http_probe(url: str, status_code: int, body_bytes: int) -> None:
    """Zeichnet ein HTTP-Probe-Ergebnis thread-lokal auf (nur Host & Status)."""
    events = getattr(_PROBE_TRACE_LOCAL, "events", None)
    if isinstance(events, list) and len(events) < 20:
        host = (urlparse(url).hostname or url)[:80]
        events.append(HttpProbeEvent(host=host, status_code=status_code, body_bytes=body_bytes))


def diagnose_empty_probe(events: list[HttpProbeEvent], query: str) -> tuple[bool, str]:
    """Erstellt eine präzise HTTP-Diagnose für 0-Treffer-Läufe (inkl. 429-Erkennung)."""
    if not events:
        return False, f"0 Treffer erhalten für Query '{query}' (leer/blockiert)."
    if all(ev.status_code == 429 for ev in events):
        hosts = ", ".join(dict.fromkeys(ev.host for ev in events))
        return True, f"HTTP 429 (Rate-Limit auf {hosts}) — 0 Treffer."

    parts: list[str] = []
    for ev in events[-3:]:
        if ev.status_code == 200:
            kb = ev.body_bytes / 1024.0
            parts.append(f"HTTP 200 ({ev.host}, {kb:.1f} KB) — 0 Treffer für '{query}'")
        else:
            label = _STATUS_LABELS.get(ev.status_code, f"Status {ev.status_code}")
            parts.append(f"HTTP {ev.status_code} ({label} auf {ev.host})")
    is_quota = events[-1].status_code == 429
    return is_quota, " | ".join(parts)[:300]


def extract_redirect_target(raw_href: str) -> str:
    """Entpackt DuckDuckGo- (`/l/?uddg=...`) oder Yahoo-Weiterleitungen (`/RU=...`)."""
    clean = (raw_href or "").strip()
    if not clean:
        return ""
    if "uddg=" in clean:
        parsed = parse_qs(urlparse(clean).query)
        vals = parsed.get("uddg", [])
        if vals:
            clean = unquote(vals[0])
    elif "/RU=" in clean:
        part = clean.split("/RU=", 1)[1].split("/RK=", 1)[0]
        clean = unquote(part)
    elif clean.startswith("//"):
        clean = "https:" + clean
    safe = sanitize_display_url(clean)
    return "" if safe == "#" else safe


def safe_http_request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    collected_cookies: dict[str, str] | None = None,
    auth: Any = None,
    timeout_seconds: float = 5.0,
    max_bytes: int = 1_500_000,
    dns_resolver: DnsResolver | None = None,
) -> tuple[int, str, str, bytes]:
    """Führt HTTP GET/POST inkl. SSRF-Redirect-Prüfung, Session-Cookies und Probe-Trace aus."""
    validate_fn = (
        (lambda u: validate_safe_external_url(u, dns_resolver=dns_resolver))
        if dns_resolver is not None
        else validate_safe_external_url
    )
    current_url = validate_fn(url)
    timeout_tuple = (min(3.0, timeout_seconds), max(1.0, timeout_seconds))
    req_method = method.upper()
    req_headers = dict(headers or API_HEADERS)

    with requests.Session() as session:
        init_host = urlparse(current_url).hostname or ""
        if cookies and init_host:
            for k, v in cookies.items():
                session.cookies.set(str(k), str(v), domain=init_host, path="/")

        for _ in range(MAX_REDIRECT_HOPS + 1):
            with session.request(
                method=req_method,
                url=current_url,
                params=params if req_method == "GET" else None,
                data=data,
                json=json_body,
                headers=req_headers,
                auth=auth,
                timeout=timeout_tuple,
                allow_redirects=False,
                stream=True,
            ) as resp:
                if resp.status_code in {301, 302, 303, 307, 308}:
                    location = str(resp.headers.get("Location") or "").strip()
                    if not location:
                        raise BrainProviderError("HTTP-Redirect ohne Zieladresse erhalten.")
                    current_url = validate_fn(urljoin(current_url, location))
                    params = None
                    req_method = "GET"
                    data = None
                    json_body = None
                    continue
                body = read_bounded_response(resp, max_bytes=max_bytes)
                ctype = str(resp.headers.get("Content-Type") or "").lower()
                if collected_cookies is not None:
                    for cookie in session.cookies:
                        if cookie.name and cookie.value is not None:
                            collected_cookies[cookie.name] = cookie.value
                record_http_probe(current_url, resp.status_code, len(body))
                return resp.status_code, current_url, ctype, body

    raise BrainProviderError(f"Zu viele Redirects (> {MAX_REDIRECT_HOPS}) für '{url}'.")
