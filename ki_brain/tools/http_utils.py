"""Gemeinsame SSRF-sichere HTTP- und Parsing-Hilfsfunktionen für Brain-Tools."""

from __future__ import annotations

import logging
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

MAX_REDIRECT_HOPS = 4
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
}
API_HEADERS: dict[str, str] = {
    "User-Agent": "KIWebsearchBrain/1.0 (+https://github.com/hschenke/bot-dashboard)",
    "Accept": "application/json,application/xml,text/plain;q=0.9,*/*;q=0.8",
}


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
    auth: Any = None,
    timeout_seconds: float = 5.0,
    max_bytes: int = 1_500_000,
    dns_resolver: DnsResolver | None = None,
) -> tuple[int, str, str, bytes]:
    """Führt HTTP GET/POST inkl. SSRF-Redirect-Prüfung und Speicherlimit aus.

    Returns:
        (status_code, final_url, content_type, body_bytes)
    """
    validate_fn = (
        (lambda u: validate_safe_external_url(u, dns_resolver=dns_resolver))
        if dns_resolver is not None
        else validate_safe_external_url
    )
    current_url = validate_fn(url)
    timeout_tuple = (min(3.0, timeout_seconds), max(1.0, timeout_seconds))
    req_method = method.upper()
    req_headers = dict(headers or API_HEADERS)

    for _ in range(MAX_REDIRECT_HOPS + 1):
        with requests.request(
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
            return resp.status_code, current_url, ctype, body

    raise BrainProviderError(f"Zu viele Redirects (> {MAX_REDIRECT_HOPS}) für '{url}'.")
