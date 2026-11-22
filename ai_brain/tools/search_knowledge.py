"""Kostenfreie Wissens-, Community- und Fachdatenbank-Tools (Wikipedia, Wikidata, HN, Reddit, arXiv)."""

from __future__ import annotations

import html
import json
import logging
import re
from urllib.parse import quote

import defusedxml.ElementTree as ET
import requests

from ai_brain.models import BrainCitation, BrainSecrets
from ai_brain.security import sanitize_display_url
from ai_brain.tools.http_utils import API_HEADERS, safe_http_request
from ai_brain.tools.search_web import search_duckduckgo_lite, search_yahoo

logger = logging.getLogger(__name__)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_wikipedia(
    query: str,
    language: str = "de",
    max_results: int = 4,
    timeout_seconds: float = 4.5,
) -> list[BrainCitation]:
    """Sehr schnelle, key-freie Enzyklopädie-Suche über Wikipedia MediaWiki API."""
    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    lang = "en" if language.lower() == "en" else "de"
    status, _, _, body = safe_http_request(
        "GET",
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": clean_q, "srlimit": max_results, "format": "json", "utf8": 1},
        headers=API_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=250_000,
    )
    if status != 200 or not body:
        return []
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for item in payload.get("query", {}).get("search", []):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        snippet = html.unescape(_HTML_TAG_RE.sub("", str(item.get("snippet") or ""))).strip()[:320]
        slug = quote(title.replace(" ", "_"), safe="_:/-")
        safe_url = sanitize_display_url(f"https://{lang}.wikipedia.org/wiki/{slug}")
        if safe_url != "#":
            results.append(BrainCitation(url=safe_url, title=f"Wikipedia: {title}", snippet=snippet, source_tool="wikipedia_api"))
    return results[:max_results]


def search_wikidata(
    query: str,
    language: str = "de",
    max_results: int = 4,
    timeout_seconds: float = 4.5,
) -> list[BrainCitation]:
    """Strukturierte Entitäts- und Faktensuche via Wikidata API (key-frei)."""
    clean_q = " ".join(query.split())[:160]
    if not clean_q:
        return []
    lang = "en" if language.lower() == "en" else "de"
    status, _, _, body = safe_http_request(
        "GET",
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbsearchentities", "search": clean_q, "language": lang, "limit": max_results, "format": "json"},
        headers=API_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=200_000,
    )
    if status != 200 or not body:
        return []
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for item in payload.get("search", []):
        qid = str(item.get("id") or "").strip()
        label = str(item.get("label") or qid).strip()
        desc = str(item.get("description") or "").strip()
        safe = sanitize_display_url(str(item.get("concepturi") or f"https://www.wikidata.org/wiki/{qid}"))
        if safe != "#" and label:
            results.append(BrainCitation(url=safe, title=f"Wikidata: {label}", snippet=desc[:300], source_tool="wikidata_api"))
    return results[:max_results]


def search_hackernews(
    query: str,
    endpoint_url: str = "https://hn.algolia.com/api/v1/search",
    max_results: int = 5,
    timeout_seconds: float = 4.5,
) -> list[BrainCitation]:
    """Tech-, Software- & Startup-Diskussionen via Hacker News Algolia API."""
    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://hn.algolia.com/api/v1/search",
        params={"query": clean_q, "tags": "story", "hitsPerPage": min(max(1, max_results), 15)},
        headers=API_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=300_000,
    )
    if status != 200 or not body:
        return []
    payload = json.loads(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    seen: set[str] = set()
    for hit in payload.get("hits", []):
        title = str(hit.get("title") or "").strip()
        raw_url = str(hit.get("url") or "").strip()
        target = raw_url if raw_url.startswith("http") else f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        safe = sanitize_display_url(target)
        if not title or safe == "#" or safe in seen:
            continue
        seen.add(safe)
        snippet = f"[HN: {hit.get('points') or 0} Punkte, {hit.get('num_comments') or 0} Kommentare | {hit.get('author') or ''}]"
        results.append(BrainCitation(url=safe, title=f"Hacker News: {title[:160]}", snippet=snippet, source_tool="hackernews_algolia"))
        if len(results) >= max_results:
            break
    return results


def _search_reddit_pullpush(query: str, max_results: int, timeout_seconds: float) -> list[BrainCitation]:
    try:
        st, _, _, body = safe_http_request(
            "GET",
            "https://api.pullpush.io/reddit/search/submission/",
            params={"q": query, "size": min(max(1, max_results), 10), "sort": "desc", "sort_type": "score"},
            headers=API_HEADERS,
            timeout_seconds=timeout_seconds,
            max_bytes=250_000,
        )
        if st != 200 or not body:
            return []
        rows = json.loads(body.decode("utf-8", errors="replace")).get("data", [])
        results: list[BrainCitation] = []
        for post in rows:
            if not isinstance(post, dict):
                continue
            raw_link = str(post.get("full_link") or f"https://www.reddit.com{post.get('permalink', '')}")
            safe = sanitize_display_url(raw_link)
            title = str(post.get("title") or "").strip()
            if safe == "#" or not title or "reddit.com" not in safe:
                continue
            sub = post.get("subreddit_name_prefixed") or f"r/{post.get('subreddit', 'reddit')}"
            snip = f"[{post.get('score') or 0} Upvotes] {str(post.get('selftext') or '')[:220]}"
            results.append(BrainCitation(url=safe, title=f"Reddit ({sub}): {title[:150]}", snippet=snip, source_tool="reddit_search"))
            if len(results) >= max_results:
                break
        return results
    except Exception:  # noqa: S110
        return []


def _search_reddit_public_feeds(query: str, max_results: int, timeout_seconds: float) -> list[BrainCitation]:
    try:
        st, _, _, body = safe_http_request(
            "GET",
            "https://www.reddit.com/search.rss",
            params={"q": query, "sort": "relevance"},
            headers=API_HEADERS,
            timeout_seconds=timeout_seconds,
            max_bytes=250_000,
        )
        if st == 200 and body:
            root = ET.fromstring(body.decode("utf-8", errors="replace"))
            hits: list[BrainCitation] = []
            for entry in root.findall("atom:entry", _ATOM_NS)[:max_results]:
                tnode, lnode = entry.find("atom:title", _ATOM_NS), entry.find("atom:link", _ATOM_NS)
                href = str(lnode.attrib.get("href") or "") if lnode is not None else ""
                safe = sanitize_display_url(href)
                title = (tnode.text or "").strip()[:160] if tnode is not None else ""
                if safe != "#" and title:
                    hits.append(BrainCitation(url=safe, title=f"Reddit: {title}", snippet="Reddit Search Feed", source_tool="reddit_search"))
            if hits:
                return hits
    except Exception:  # noqa: S110
        pass
    try:
        st_j, _, _, jbody = safe_http_request(
            "GET",
            "https://www.reddit.com/search.json",
            params={"q": query, "limit": min(max(1, max_results), 10), "sort": "relevance"},
            headers=API_HEADERS,
            timeout_seconds=timeout_seconds,
            max_bytes=250_000,
        )
        if st_j == 200 and jbody:
            children = json.loads(jbody.decode("utf-8", errors="replace")).get("data", {}).get("children", [])
            hits_j: list[BrainCitation] = []
            for child in children[:max_results]:
                post = child.get("data", {})
                safe = sanitize_display_url(f"https://www.reddit.com{post.get('permalink', '')}")
                title = str(post.get("title") or "").strip()[:150]
                if safe != "#" and title:
                    sub = post.get("subreddit_name_prefixed") or "r/reddit"
                    hits_j.append(BrainCitation(url=safe, title=f"Reddit ({sub}): {title}", snippet=str(post.get("selftext") or "")[:220], source_tool="reddit_search"))
            if hits_j:
                return hits_j
    except Exception:  # noqa: S110
        pass
    return []


def search_reddit(
    query: str,
    secrets: BrainSecrets,
    max_results: int = 5,
    timeout_seconds: float = 5.0,
) -> list[BrainCitation]:
    """Key-freie Reddit-Community-Suche (PullPush, Reddit-RSS/JSON, Yahoo/DDG Site-Fallback; OAuth optional)."""
    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    cid = secrets.resolve_key("REDDIT_CLIENT_ID")
    csec = secrets.resolve_key("REDDIT_CLIENT_SECRET")
    if cid and csec:
        oauth_hits = _search_reddit_oauth(clean_q, cid, csec, max_results, timeout_seconds)
        if oauth_hits:
            return oauth_hits

    for fetcher in (_search_reddit_pullpush, _search_reddit_public_feeds):
        hits = fetcher(clean_q, max_results, timeout_seconds)
        if hits:
            return hits

    site_q = f"site:reddit.com {clean_q}"
    for web_fn in (search_yahoo, search_duckduckgo_lite):
        try:
            raw_hits = web_fn(site_q, max_results=max_results * 2, timeout_seconds=timeout_seconds)
            filtered = [
                c.model_copy(update={"title": f"Reddit: {c.title}", "source_tool": "reddit_search"})
                for c in raw_hits
                if "reddit.com" in c.url
            ][:max_results]
            if filtered:
                return filtered
        except Exception as exc:  # noqa: BLE001
            logger.debug("Reddit Site-Fallback fehlgeschlagen: %s", exc)
    return []


def _search_reddit_oauth(
    query: str,
    client_id: str,
    client_secret: str,
    max_results: int,
    timeout_seconds: float,
) -> list[BrainCitation]:
    st, _, _, tok_body = safe_http_request(
        "POST",
        "https://www.reddit.com/api/v1/access_token",
        auth=requests.auth.HTTPBasicAuth(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers=API_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=50_000,
    )
    if st != 200:
        return []
    token = str(json.loads(tok_body.decode("utf-8", errors="replace")).get("access_token") or "")
    if not token:
        return []
    st2, _, _, sbody = safe_http_request(
        "GET",
        "https://oauth.reddit.com/search",
        params={"q": query, "limit": min(max_results, 10), "sort": "relevance"},
        headers={"Authorization": f"Bearer {token}", **API_HEADERS},
        timeout_seconds=timeout_seconds,
        max_bytes=250_000,
    )
    if st2 != 200:
        return []
    data = json.loads(sbody.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        url = sanitize_display_url(f"https://www.reddit.com{post.get('permalink', '')}")
        title = str(post.get("title") or "").strip()
        if url != "#" and title:
            sub = post.get("subreddit_name_prefixed") or "r/reddit"
            snip = f"[{post.get('score') or 0} Upvotes] {str(post.get('selftext') or '')[:240]}"
            results.append(BrainCitation(url=url, title=f"Reddit ({sub}): {title[:150]}", snippet=snip, source_tool="reddit_search"))
    return results[:max_results]


def search_arxiv(
    query: str,
    endpoint_url: str = "https://export.arxiv.org/api/query",
    max_results: int = 4,
    timeout_seconds: float = 5.5,
) -> list[BrainCitation]:
    """Wissenschaftliche Preprints via arXiv Atom API (key-frei, defusedxml)."""
    clean_q = " ".join(query.split())[:180]
    if not clean_q:
        return []
    status, _, _, body = safe_http_request(
        "GET",
        endpoint_url or "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{clean_q}", "start": 0, "max_results": min(max(1, max_results), 10)},
        headers=API_HEADERS,
        timeout_seconds=timeout_seconds,
        max_bytes=400_000,
    )
    if status != 200 or not body:
        return []
    root = ET.fromstring(body.decode("utf-8", errors="replace"))
    results: list[BrainCitation] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        tnode, inode = entry.find("atom:title", _ATOM_NS), entry.find("atom:id", _ATOM_NS)
        if tnode is None or inode is None:
            continue
        url = sanitize_display_url((inode.text or "").strip())
        if url == "#":
            continue
        title = " ".join((tnode.text or "").split())[:160]
        authors = [a.find("atom:name", _ATOM_NS).text.strip() for a in entry.findall("atom:author", _ATOM_NS)[:3] if a.find("atom:name", _ATOM_NS) is not None and a.find("atom:name", _ATOM_NS).text]
        snode = entry.find("atom:summary", _ATOM_NS)
        summary = " ".join((snode.text or "").split())[:280] if snode is not None else ""
        results.append(BrainCitation(url=url, title=f"arXiv: {title}", snippet=f"[{', '.join(authors) or 'arXiv'}] {summary}", source_tool="arxiv_api"))
        if len(results) >= max_results:
            break
    return results
