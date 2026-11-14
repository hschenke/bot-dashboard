"""Kostenfreie Web- & News-Suchmaschinen ohne API-Key (Yahoo, DuckDuckGo, Google News)."""

from __future__ import annotations

import datetime
import email.utils
from urllib.parse import quote

import defusedxml.ElementTree as ET
from bs4 import BeautifulSoup

from ki_brain.models import BrainCitation
from ki_brain.security import sanitize_display_url
from ki_brain.tools.http_utils import BROWSER_HEADERS, extract_redirect_target, safe_http_request


def search_yahoo(
    query: str,
    endpoint_url: str = "https://de.search.yahoo.com/search",
    max_results: int = 6,
    timeout_seconds: float = 6.0,
) -> list[BrainCitation]:
    """Key-freie Yahoo-Websuche (resilient auf Cloud-Servern)."""
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://de.search.yahoo.com/search",
        params={"p": clean_q},
        headers=BROWSER_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=500_000,
    )
    if status != 200 or not body:
        return []
    soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "html.parser")
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for node in soup.select("div.algo"):
        anchor = node.select_one("h3 a")
        if not anchor:
            continue
        target = extract_redirect_target(str(anchor.get("href") or ""))
        if not target or target in seen:
            continue
        seen.add(target)
        comp = node.select_one(".compText, .compText p, .fc-falcon")
        snippet = comp.get_text(" ", strip=True)[:320] if comp else ""
        title = anchor.get_text(" ", strip=True)[:180]
        if title:
            results.append(BrainCitation(url=target, title=title, snippet=snippet, source_tool="yahoo_search"))
        if len(results) >= max_results:
            break
    return results


def search_duckduckgo_html(
    query: str,
    endpoint_url: str = "https://html.duckduckgo.com/html/",
    max_results: int = 6,
    timeout_seconds: float = 5.0,
) -> list[BrainCitation]:
    """Key-freie DuckDuckGo HTML-Suche."""
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://html.duckduckgo.com/html/",
        params={"q": clean_q},
        headers=BROWSER_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=400_000,
    )
    if status != 200 or not body:
        return []
    soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "html.parser")
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for anchor in soup.select("a.result__a"):
        target = extract_redirect_target(str(anchor.get("href") or ""))
        if not target or target in seen:
            continue
        seen.add(target)
        parent = anchor.find_parent(class_="result") or anchor.find_parent("div")
        snip_node = parent.select_one(".result__snippet") if parent else None
        snippet = snip_node.get_text(" ", strip=True)[:320] if snip_node else ""
        title = anchor.get_text(" ", strip=True)[:180]
        if title:
            results.append(BrainCitation(url=target, title=title, snippet=snippet, source_tool="duckduckgo_html"))
        if len(results) >= max_results:
            break
    return results


def search_duckduckgo_lite(
    query: str,
    endpoint_url: str = "https://lite.duckduckgo.com/lite/",
    max_results: int = 6,
    timeout_seconds: float = 5.0,
) -> list[BrainCitation]:
    """Kompakte DuckDuckGo Lite POST-Suche ohne API-Key."""
    clean_q = " ".join(query.split())[:200]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "POST",
        endpoint_url or "https://lite.duckduckgo.com/lite/",
        data={"q": clean_q},
        headers=BROWSER_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=350_000,
    )
    if status != 200 or not body:
        return []
    soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "html.parser")
    links = soup.select("a.result-link")
    snippets = soup.select(".result-snippet")
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for idx, anchor in enumerate(links):
        target = extract_redirect_target(str(anchor.get("href") or ""))
        if not target or target in seen:
            continue
        seen.add(target)
        title = anchor.get_text(" ", strip=True)[:180]
        snippet = snippets[idx].get_text(" ", strip=True)[:320] if idx < len(snippets) else ""
        if title:
            results.append(BrainCitation(url=target, title=title, snippet=snippet, source_tool="duckduckgo_lite"))
        if len(results) >= max_results:
            break
    return results


def search_duckduckgo_instant(
    query: str,
    endpoint_url: str = "https://api.duckduckgo.com/",
    max_results: int = 5,
    timeout_seconds: float = 4.5,
) -> list[BrainCitation]:
    """Offizielle JSON-Instant-Answer-API von DuckDuckGo."""
    import json

    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://api.duckduckgo.com/",
        params={"q": clean_q, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout_seconds=timeout_seconds,
        max_bytes=250_000,
    )
    if status != 200 or not body:
        return []
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    abstract_url = sanitize_display_url(str(payload.get("AbstractURL") or ""))
    abstract_text = str(payload.get("AbstractText") or "").strip()
    if abstract_url != "#" and abstract_text:
        heading = str(payload.get("Heading") or clean_q)[:180]
        results.append(
            BrainCitation(url=abstract_url, title=heading, snippet=abstract_text[:350], source_tool="duckduckgo_instant")
        )
    for topic in payload.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
            safe = sanitize_display_url(str(topic["FirstURL"]))
            if safe != "#":
                text = str(topic["Text"]).strip()
                results.append(
                    BrainCitation(url=safe, title=text.split(" - ", 1)[0][:150], snippet=text[:300], source_tool="duckduckgo_instant")
                )
    if not results:
        fallback_url = sanitize_display_url(f"https://duckduckgo.com/?q={quote(clean_q)}")
        if abstract_text:
            results.append(BrainCitation(url=fallback_url, title=clean_q, snippet=abstract_text[:300], source_tool="duckduckgo_instant"))
    return results[:max_results]


def search_google_news_rss(
    query: str,
    endpoint_url: str = "https://news.google.com/rss/search",
    language: str = "de",
    max_results: int = 6,
    timeout_seconds: float = 5.5,
) -> list[BrainCitation]:
    """Kostenfreie Echtzeit-Nachrichtensuche über Google News RSS (XXE-geschützt)."""
    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    timed_q = clean_q if "when:" in clean_q else f"{clean_q} when:180d"
    hl, gl, ceid = ("en", "US", "US:en") if language.lower() == "en" else ("de", "DE", "DE:de")
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://news.google.com/rss/search",
        params={"q": timed_q, "hl": hl, "gl": gl, "ceid": ceid},
        headers=BROWSER_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=400_000,
    )
    if status != 200 or not body:
        return []
    root = ET.fromstring(body.decode("utf-8", errors="replace"))
    now = datetime.datetime.now(datetime.UTC)
    results: list[BrainCitation] = []
    for item in root.findall(".//item"):
        if len(results) >= max_results:
            break
        pub_raw = (item.findtext("pubDate") or "").strip()
        if pub_raw:
            try:
                dt = email.utils.parsedate_to_datetime(pub_raw)
                if dt and (now - dt.astimezone(datetime.UTC)).days > 365:
                    continue
            except Exception:  # noqa: S110
                pass
        link = sanitize_display_url((item.findtext("link") or "").strip())
        title = (item.findtext("title") or "").strip()[:180]
        desc = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)[:300]
        if link != "#" and title:
            results.append(BrainCitation(url=link, title=title, snippet=desc, source_tool="google_news_rss", published_at=pub_raw))
    return results
