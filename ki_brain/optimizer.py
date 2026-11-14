"""Stufen-Optimierer: Berechnet die beste Tool- & Modell-Sequenz für Fast, Standard & Deep."""

from __future__ import annotations

import datetime
from collections.abc import Iterable

from ki_brain.enums import BrainTier, ProbeStatus, ToolCategory
from ki_brain.models import AIModelSpec, TierSequenceConfig, ToolSpec
from ki_brain.tools.registry import is_extractor_capable, is_search_capable

DEFAULT_TEST_QUERY = "Python AI"
DEFAULT_TEST_URL = "https://de.wikipedia.org/wiki/Python_(Programmiersprache)"


def compute_speed_score(latency_ms: int, max_ms: int = 6000) -> float:
    """Normiert Antwortzeit auf 0..100 Punkte (schneller = höher)."""
    if latency_ms <= 0:
        return 75.0
    clamped = min(max_ms, max(100, latency_ms))
    return round(100.0 * (1.0 - (clamped - 100) / max(1, max_ms - 100)), 1)


def compute_tool_quality(status: ProbeStatus, latency_ms: int, result_count: int, text_chars: int = 0) -> float:
    """Berechnet einen Gesamt-Qualitätsscore (0..100) aus Speed, Verlässlichkeit & Ertrag."""
    if status == ProbeStatus.AUTH_MISSING:
        return 0.0
    if status in {ProbeStatus.BLOCKED, ProbeStatus.ERROR}:
        return 5.0
    if status == ProbeStatus.QUOTA_LIMITED:
        return 12.0
    if status == ProbeStatus.UNTESTED:
        return 50.0
    speed = compute_speed_score(latency_ms)
    output_pts = min(100.0, (result_count * 18.0) + (min(text_chars, 4000) / 40.0))
    return round(0.45 * speed + 0.55 * max(20.0, output_pts), 1)


def compute_model_quality(model: AIModelSpec, status: ProbeStatus, latency_ms: int) -> float:
    """Berechnet den Modell-Qualitätsscore (0..100) nach Health, Speed & Tier-Fähigkeit."""
    if status == ProbeStatus.AUTH_MISSING:
        return 0.0
    if status in {ProbeStatus.BLOCKED, ProbeStatus.ERROR}:
        return 5.0
    if status == ProbeStatus.QUOTA_LIMITED:
        return 12.0
    if status == ProbeStatus.UNTESTED:
        return 55.0
    speed = compute_speed_score(latency_ms, max_ms=8000)
    tier_bonus = 92.0 if model.tier_hint == BrainTier.DEEP else (84.0 if model.tier_hint == BrainTier.STANDARD else 76.0)
    struct_bonus = 6.0 if model.supports_structured_output else 0.0
    return round(min(100.0, 0.45 * speed + 0.55 * tier_bonus + struct_bonus), 1)


def _is_usable_status(status: ProbeStatus) -> bool:
    return status in {ProbeStatus.OK, ProbeStatus.UNTESTED}


def _rank_tools_for_tier(tools: list[ToolSpec], tier: BrainTier) -> list[ToolSpec]:
    usable = [t for t in tools if t.enabled and _is_usable_status(t.last_status)]
    if not usable:
        usable = [t for t in tools if t.enabled and t.last_status != ProbeStatus.AUTH_MISSING]

    def _score(t: ToolSpec) -> tuple[int, float]:
        status_rank = 2 if t.last_status == ProbeStatus.OK else 1
        speed = compute_speed_score(t.last_latency_ms or 1200)
        keyfree_bonus = 10.0 if not t.requires_api_key else 0.0
        if tier == BrainTier.FAST:
            val = 0.60 * speed + 0.25 * (t.reliability_score or 50.0) + 0.15 * t.quality_score + keyfree_bonus
        elif tier == BrainTier.STANDARD:
            val = 0.35 * speed + 0.35 * (t.quality_score or 50.0) + 0.30 * (t.reliability_score or 50.0)
        else:
            rich_cat = 10.0 if t.category in {ToolCategory.ACADEMIC_SEARCH, ToolCategory.COMMUNITY_SEARCH} else 5.0
            val = 0.50 * (t.quality_score or 50.0) + 0.35 * (t.reliability_score or 50.0) + 0.15 * speed + rich_cat
        return status_rank, val

    sorted_tools = sorted(usable, key=_score, reverse=True)
    if tier == BrainTier.FAST:
        return sorted_tools

    # Für Standard/Deep zuerst besten Vertreter jeder Kategorie wählen (Quellen-Diversität)
    picked: list[ToolSpec] = []
    seen_ids: set[str] = set()
    for cat in (
        ToolCategory.WEB_SEARCH,
        ToolCategory.NEWS_SEARCH,
        ToolCategory.ACADEMIC_SEARCH,
        ToolCategory.COMMUNITY_SEARCH,
    ):
        for t in sorted_tools:
            if t.category == cat and t.id not in seen_ids:
                picked.append(t)
                seen_ids.add(t.id)
                break
    for t in sorted_tools:
        if t.id not in seen_ids:
            picked.append(t)
    return picked


def _rank_models_for_tier(models: list[AIModelSpec], tier: BrainTier) -> list[AIModelSpec]:
    usable = [m for m in models if m.enabled and _is_usable_status(m.last_status)]
    if not usable:
        usable = [m for m in models if m.enabled and m.last_status != ProbeStatus.AUTH_MISSING]

    def _score(m: AIModelSpec) -> tuple[int, float]:
        status_rank = 2 if m.last_status == ProbeStatus.OK else 1
        speed = compute_speed_score(m.last_latency_ms or 1500, max_ms=8000)
        qual = m.quality_score or 55.0
        if tier == BrainTier.FAST:
            if m.is_deep_agent:
                return 0, -10.0
            hint = 15.0 if m.tier_hint == BrainTier.FAST else (5.0 if m.tier_hint == BrainTier.STANDARD else 0.0)
            return status_rank, 0.65 * speed + 0.25 * qual + hint
        if tier == BrainTier.STANDARD:
            if m.is_deep_agent:
                return 0, 0.0
            hint = 12.0 if m.tier_hint == BrainTier.STANDARD else (8.0 if m.tier_hint == BrainTier.FAST else 5.0)
            return status_rank, 0.45 * qual + 0.40 * speed + hint
        hint = 18.0 if (m.is_deep_agent or m.tier_hint == BrainTier.DEEP) else (10.0 if m.tier_hint == BrainTier.STANDARD else 2.0)
        return status_rank, 0.70 * qual + 0.15 * speed + hint

    ranked = sorted(usable, key=_score, reverse=True)
    if tier != BrainTier.DEEP:
        ranked = [m for m in ranked if not m.is_deep_agent] or ranked
    return ranked


def _format_rationale(tier: BrainTier, search_ids: list[str], model_ids: list[str], tools: Iterable[ToolSpec]) -> str:
    skipped = [
        f"{t.id} ({t.last_status.value})"
        for t in tools
        if t.enabled and t.last_status in {ProbeStatus.QUOTA_LIMITED, ProbeStatus.AUTH_MISSING, ProbeStatus.BLOCKED, ProbeStatus.ERROR}
    ]
    top_s = " → ".join(search_ids[:4]) or "–"
    top_m = " → ".join(model_ids[:3]) or "Heuristik"
    skip_note = f" | Übersprungen: {', '.join(skipped[:4])}" if skipped else ""
    return f"[{tier.value.upper()}] Suche: {top_s} | Modelle: {top_m}{skip_note}"[:1150]


def optimize_tier_sequences(
    models: list[AIModelSpec],
    tools: list[ToolSpec],
    existing_sequences: dict[BrainTier, TierSequenceConfig],
    *,
    force_overwrite: bool = False,
) -> dict[BrainTier, TierSequenceConfig]:
    """Berechnet optimierte Ausführungs-Sequenzen für Fast, Standard & Deep."""
    search_tools = [t for t in tools if is_search_capable(t)]
    extractors = [t for t in tools if is_extractor_capable(t)]
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    result: dict[BrainTier, TierSequenceConfig] = {}

    for tier in (BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP):
        prev = existing_sequences.get(tier)
        if prev is not None and prev.manual_override and not force_overwrite:
            result[tier] = prev
            continue
        ranked_search = _rank_tools_for_tier(search_tools, tier)
        ranked_ext = _rank_tools_for_tier(extractors, tier)
        ranked_models = _rank_models_for_tier(models, tier)
        s_ids = [t.id for t in ranked_search]
        e_ids = [t.id for t in ranked_ext]
        m_ids = [m.id for m in ranked_models]
        has_deep_agent = any(m.is_deep_agent and _is_usable_status(m.last_status) for m in ranked_models)

        if tier == BrainTier.FAST:
            cfg = TierSequenceConfig(
                tier=tier,
                search_tool_sequence=s_ids,
                extractor_tool_sequence=e_ids,
                model_sequence=m_ids,
                max_search_tools=2,
                enable_page_scraping=False,
                max_pages_to_scrape=0,
                max_followup_rounds=0,
                single_pass=True,
                use_deep_agent=False,
                optimization_rationale=_format_rationale(tier, s_ids, m_ids, tools),
                updated_at=now,
            )
        elif tier == BrainTier.STANDARD:
            cfg = TierSequenceConfig(
                tier=tier,
                search_tool_sequence=s_ids,
                extractor_tool_sequence=e_ids,
                model_sequence=m_ids,
                max_search_tools=4,
                enable_page_scraping=bool(e_ids),
                max_pages_to_scrape=2 if e_ids else 0,
                max_followup_rounds=1,
                single_pass=False,
                use_deep_agent=False,
                optimization_rationale=_format_rationale(tier, s_ids, m_ids, tools),
                updated_at=now,
            )
        else:
            cfg = TierSequenceConfig(
                tier=tier,
                search_tool_sequence=s_ids,
                extractor_tool_sequence=e_ids,
                model_sequence=m_ids,
                max_search_tools=max(6, len(s_ids)),
                enable_page_scraping=bool(e_ids),
                max_pages_to_scrape=5 if e_ids else 0,
                max_followup_rounds=2,
                single_pass=False,
                use_deep_agent=has_deep_agent,
                optimization_rationale=_format_rationale(tier, s_ids, m_ids, tools),
                updated_at=now,
            )
        result[tier] = cfg
    return result
