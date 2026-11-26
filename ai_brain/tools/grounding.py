"""Ausführung & Snippet-Anreicherung für Gemini AI-Grounding (Google Maps & Google Search)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import quote_plus

from ai_brain.models import AIModelSpec, BrainCitation, BrainSecrets, ToolSpec

AI_SEARCH_EXECUTORS = frozenset({"gemini_google_search", "gemini_google_maps"})

_MAX_EXCERPT_CHARS = 440
_MAX_MERGED_SNIPPET_CHARS = 600
_GENERIC_TITLES = frozenset({"", "google maps place", "google maps"})
_STOPWORDS = frozenset(
    {
        "der", "die", "das", "und", "oder", "fur", "von", "vom", "zur", "zum", "bei", "mit",
        "the", "and", "for", "with", "from", "gmbh", "inc", "llc", "ltd",
    }
)
_METADATA_PREFIXES = (
    "bewertung", "sterne", "rating", "adresse", "address", "standort", "lage",
    "offnungszeiten", "opening hours", "hours", "preis", "price", "telefon",
    "phone", "website", "kategorie", "highlights", "besonderheiten", "hinweis",
)
_HEADER_START_RE = re.compile(r"^\s*(?:#{1,4}\s+|\d+[.)]\s+|[-*•]\s+\*\*|\*\*[^*]{2,120}\*\*\s*:?$)")
_PLAIN_BULLET_RE = re.compile(r"^[-*•]\s+")
_LINE_PREFIX_RE = re.compile(r"^\s*(?:#{1,4}\s+|\d+[.)]\s+|[-*•]\s+)")
_MD_SYNTAX_RE = re.compile(r"[*_`~]+")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _fold_text(value: str) -> str:
    nfkd = unicodedata.normalize("NFKD", value or "")
    ascii_ish = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", _MD_SYNTAX_RE.sub(" ", ascii_ish.lower())).strip()


def _truncate_clean(value: str, max_chars: int) -> str:
    text = _WS_RE.sub(" ", (value or "").strip())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" ·,;:-")
    return f"{cut or text[: max_chars - 1]}…"


def _clean_excerpt(raw: str, max_chars: int = _MAX_EXCERPT_CHARS) -> str:
    parts = [
        _WS_RE.sub(" ", _MD_SYNTAX_RE.sub("", _LINE_PREFIX_RE.sub("", line))).strip()
        for line in (raw or "").splitlines()
    ]
    return _truncate_clean(" · ".join(p for p in parts if p), max_chars)


def _is_metadata_bullet(raw_line: str) -> bool:
    stripped = raw_line.lstrip()
    if not _PLAIN_BULLET_RE.match(stripped):
        return False
    folded = _fold_text(_PLAIN_BULLET_RE.sub("", stripped))
    return any(folded.startswith((f"{p}:", f"{p} ")) for p in _METADATA_PREFIXES)


def _split_text_sections(text: str) -> list[str]:
    """Zerlegt Gemini-Ausgaben in logische Einzel-Abschnitte (pro Ort bzw. Treffer)."""
    sections: list[str] = []
    buf: list[str] = []
    buf_plain_bullet = False

    def _flush() -> None:
        nonlocal buf_plain_bullet
        if buf:
            sections.append("\n".join(buf).strip())
            buf.clear()
        buf_plain_bullet = False

    for line in (text or "").splitlines():
        if not line.strip():
            if len(buf) == 1 and buf[0].lstrip().startswith(("#", "**")):
                continue
            _flush()
            continue
        indent = len(line) - len(line.lstrip())
        stripped_left = line.lstrip()
        is_bullet = bool(_PLAIN_BULLET_RE.match(stripped_left))
        is_meta = _is_metadata_bullet(line)
        is_bold_bullet = bool(indent == 0 and is_bullet and not is_meta and re.match(r"^[-*•]\s+\*\*", line))
        is_header = bool((indent <= 2 and not is_bullet and _HEADER_START_RE.match(line)) or is_bold_bullet)
        is_flat_bullet = bool(indent == 0 and is_bullet and not is_meta)
        if is_header or (is_flat_bullet and (not buf or buf_plain_bullet)):
            _flush()
            buf.append(line)
            buf_plain_bullet = is_flat_bullet and not is_header
        else:
            buf.append(line)
    _flush()
    return [sec for sec in sections if sec]


def _section_heading(section: str) -> str:
    first = _MD_SYNTAX_RE.sub("", _LINE_PREFIX_RE.sub("", section.splitlines()[0])).strip()
    for sep in (" – ", " - ", ": ", " ("):
        if sep in first:
            first = first.split(sep, 1)[0].strip()
    return _fold_text(first)


def _score_section_match(folded_title: str, tokens: list[str], section: str) -> int:
    folded_sec = _fold_text(section)
    if not folded_sec:
        return 0
    pos = folded_sec.find(folded_title)
    if pos != -1:
        return 120 + len(folded_title) + (50 if pos < 45 else 0)
    heading = _section_heading(section)
    if len(heading) >= 4 and heading in folded_title:
        return 100 + len(heading)
    hits = [tok for tok in tokens if tok in folded_sec]
    if len(hits) >= 2 and (len(hits) / len(tokens)) >= 0.6:
        return 60 + int((len(hits) / len(tokens)) * 30) + len(hits) * 5
    if len(tokens) == 1 and len(hits) == 1 and len(hits[0]) >= 5 and hits[0] in heading:
        return 70
    return 0


def _match_section(title: str, sections: list[str]) -> str:
    folded = _fold_text(title)
    if folded in _GENERIC_TITLES or len(folded) < 3:
        return ""
    tokens = [tok for tok in _WORD_RE.findall(folded) if tok not in _STOPWORDS]
    best = max((( _score_section_match(folded, tokens, sec), sec) for sec in sections), key=lambda x: x[0], default=(0, ""))
    return best[1] if best[0] > 0 else ""


def _merge_snippet(existing: str, excerpt: str, max_chars: int = _MAX_MERGED_SNIPPET_CHARS) -> str:
    clean_ex = _WS_RE.sub(" ", (existing or "").strip())
    clean_new = _WS_RE.sub(" ", (excerpt or "").strip())
    if not clean_new:
        return clean_ex
    if not clean_ex or _fold_text(clean_ex) in _fold_text(clean_new):
        return _truncate_clean(clean_new, max_chars)
    if _fold_text(clean_new) in _fold_text(clean_ex):
        return _truncate_clean(clean_ex, max_chars)
    return _truncate_clean(f"{clean_new} · {clean_ex}", max_chars)


def enrich_grounding_citations(
    citations: list[BrainCitation],
    text: str,
    *,
    executor_type: str,
    query: str,
    max_res: int,
) -> list[BrainCitation]:
    """Hängt generierten Gemini-Maps-/Search-Detailtext gezielt als Snippet an Citations an."""
    base = list(citations[:max_res])
    clean_full = _clean_excerpt(text, _MAX_MERGED_SNIPPET_CHARS)
    if not clean_full:
        return base
    if not base:
        clean_q = _WS_RE.sub(" ", (query or "").strip())
        if executor_type == "gemini_google_maps" and clean_q:
            return [
                BrainCitation(
                    url=f"https://www.google.com/maps/search/?api=1&query={quote_plus(clean_q)}",
                    title=f"Google Maps: {clean_q[:140]}",
                    snippet=clean_full,
                    source_tool="gemini_google_maps",
                )
            ]
        return []

    sections = _split_text_sections(text)
    enriched: list[BrainCitation] = []
    matched_any = False
    for cite in base:
        matched = _match_section(cite.title, sections)
        if matched:
            matched_any = True
            snippet = _merge_snippet(cite.snippet, _clean_excerpt(matched, _MAX_EXCERPT_CHARS))
            enriched.append(cite.model_copy(update={"snippet": snippet}))
        else:
            enriched.append(cite)
    if not matched_any:
        enriched[0] = enriched[0].model_copy(update={"snippet": _merge_snippet(enriched[0].snippet, clean_full)})
    return enriched


def execute_ai_grounding_search(
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
    """Führt Gemini Google Search oder Google Maps Grounding aus und übernimmt Detailtexte in Snippets."""
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
        prompt = (
            f"Finde passende Orte auf Google Maps zu: {query}. "
            "Nenne pro Ort kurz Name, Bewertung (Sterne), Adresse, Öffnungszeiten und Preisniveau."
        )
    else:
        tool_spec = {"type": "google_search"}
        prompt = f"Suche aktuell im Web nach: {query}"

    text, citations = active_runner.run_text(
        model,
        prompt,
        timeout_seconds=min(20.0, max(5.0, tool.timeout_seconds)),
        tools=[tool_spec],
        allow_paid=allow_paid,
    )
    return enrich_grounding_citations(citations, text, executor_type=etype, query=query, max_res=max_res)
