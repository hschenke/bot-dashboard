"""Stufen-Optimierer: Berechnet die beste Tool- & Modell-Sequenz für Fast, Standard & Deep."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from typing import Any

from ai_brain.capability import (
    estimate_model_capability,
    is_fast_class_model,
    is_small_capacity_model,
)
from ai_brain.enums import BrainTier, ProbeStatus, ToolCategory
from ai_brain.models import AIModelSpec, TierSequenceConfig, ToolSpec
from ai_brain.pricing import compute_value_score
from ai_brain.quotas import evaluate_model_quota, evaluate_tool_quota
from ai_brain.tools.registry import is_extractor_capable, is_search_capable

DEFAULT_TEST_QUERY = "Python AI"
DEFAULT_TEST_URL = "https://de.wikipedia.org/wiki/Python_(Programmiersprache)"

_MODEL_SPEED_WEIGHTS: dict[BrainTier, float] = {
    BrainTier.FAST: 0.06,
    BrainTier.STANDARD: 0.04,
    BrainTier.DEEP: 0.02,
}
_PAID_VALUE_SENSITIVITY: dict[BrainTier, float] = {
    BrainTier.FAST: 0.50,
    BrainTier.STANDARD: 0.45,
    BrainTier.DEEP: 0.35,
}


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
    """Berechnet den Modell-Qualitätsscore (0..100) primär aus Modellstärke & Status (+ leichtem Speed-Anteil)."""
    if status == ProbeStatus.AUTH_MISSING:
        return 0.0
    if status in {ProbeStatus.BLOCKED, ProbeStatus.ERROR}:
        return 5.0
    if status == ProbeStatus.QUOTA_LIMITED:
        return 12.0
    cap = estimate_model_capability(model)
    if status == ProbeStatus.UNTESTED:
        return round(max(15.0, min(95.0, cap * 0.75)), 1)
    speed = compute_speed_score(latency_ms, max_ms=8000)
    struct_bonus = 2.0 if model.supports_structured_output else -2.0
    return round(max(15.0, min(100.0, 0.92 * cap + 0.08 * speed + struct_bonus)), 1)


def _is_usable_status(status: ProbeStatus) -> bool:
    return status in {ProbeStatus.OK, ProbeStatus.UNTESTED}


def _rank_tools_for_tier(
    tools: list[ToolSpec],
    tier: BrainTier,
    *,
    usage_state: dict[str, Any] | None = None,
    allow_paid: bool = False,
) -> list[ToolSpec]:
    enabled = [t for t in tools if t.enabled]
    if not allow_paid:
        enabled = [t for t in enabled if evaluate_tool_quota(t, usage_state).is_free_available]
    usable = [t for t in enabled if _is_usable_status(t.last_status)]
    if not usable:
        usable = [t for t in enabled if t.last_status != ProbeStatus.AUTH_MISSING]

    def _score(t: ToolSpec) -> tuple[int, float]:
        snap = evaluate_tool_quota(t, usage_state)
        budget_score = 100.0 if (snap.is_free and snap.limit == 0) else (20.0 + 80.0 * snap.budget_ratio if snap.is_free_available else 0.0)
        free_pref = 15.0 if snap.is_free_available else 0.0
        status_rank = 2 if t.last_status == ProbeStatus.OK else 1
        speed = compute_speed_score(t.last_latency_ms or 1200)
        keyfree_bonus = 10.0 if not t.requires_api_key else 0.0
        if tier == BrainTier.FAST:
            val = 0.48 * speed + 0.22 * (t.reliability_score or 50.0) + 0.15 * t.quality_score + 0.15 * budget_score + keyfree_bonus + free_pref
        elif tier == BrainTier.STANDARD:
            val = 0.28 * speed + 0.30 * (t.quality_score or 50.0) + 0.27 * (t.reliability_score or 50.0) + 0.15 * budget_score + free_pref
        else:
            rich_cat = 10.0 if t.category in {ToolCategory.ACADEMIC_SEARCH, ToolCategory.COMMUNITY_SEARCH} else 5.0
            val = 0.45 * (t.quality_score or 50.0) + 0.30 * (t.reliability_score or 50.0) + 0.15 * speed + 0.10 * budget_score + rich_cat + (0.0 if allow_paid else free_pref)
        return status_rank, val

    sorted_tools = sorted(usable, key=_score, reverse=True)
    if tier == BrainTier.FAST:
        return sorted_tools

    # Für Standard/Deep zuerst besten Vertreter jeder Kategorie wählen (inkl. AI Grounding)
    picked: list[ToolSpec] = []
    seen_ids: set[str] = set()
    for cat in (
        ToolCategory.WEB_SEARCH,
        ToolCategory.NEWS_SEARCH,
        ToolCategory.ACADEMIC_SEARCH,
        ToolCategory.COMMUNITY_SEARCH,
        ToolCategory.AI_GROUNDING,
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


def _score_model_for_tier(
    model: AIModelSpec,
    tier: BrainTier,
    *,
    allow_paid: bool,
) -> tuple[int, int, float, float, float]:
    """Berechnet Ranking-Schlüssel: Free = Best→Worst (Fast bevorzugt leichte Modelle), Paid = Bestes Preis-Leistungs-Verhältnis."""
    if model.is_deep_agent and tier != BrainTier.DEEP:
        return 0, -1, -10.0, 0.0, 0.0
    status_rank = 2 if model.last_status == ProbeStatus.OK else (1 if model.last_status == ProbeStatus.UNTESTED else 0)
    cap = estimate_model_capability(model)
    speed = compute_speed_score(model.last_latency_ms or 1500, max_ms=8000)
    small = is_small_capacity_model(model)
    if tier == BrainTier.FAST:
        tier_bucket = 2 if (is_fast_class_model(model) and not small) else (1 if not small else 0)
    else:
        tier_bucket = 1 if not small else 0

    speed_w = _MODEL_SPEED_WEIGHTS[tier]
    base_metric = (
        compute_value_score(model, price_sensitivity=_PAID_VALUE_SENSITIVITY[tier])
        if allow_paid
        else cap
    )
    primary = (1.0 - speed_w) * base_metric + speed_w * speed
    return status_rank, tier_bucket, primary, cap, speed


def _rank_models_for_tier(
    models: list[AIModelSpec],
    tier: BrainTier,
    *,
    usage_state: dict[str, Any] | None = None,
    allow_paid: bool = False,
) -> list[AIModelSpec]:
    enabled = [m for m in models if m.enabled]
    if not allow_paid:
        enabled = [m for m in enabled if evaluate_model_quota(m, usage_state).is_free_available]
    usable = [m for m in enabled if _is_usable_status(m.last_status)]
    if not usable:
        usable = [m for m in enabled if m.last_status != ProbeStatus.AUTH_MISSING]

    ranked = sorted(
        usable,
        key=lambda m: _score_model_for_tier(m, tier, allow_paid=allow_paid),
        reverse=True,
    )
    if tier != BrainTier.DEEP:
        ranked = [m for m in ranked if not m.is_deep_agent] or ranked
    return ranked


def _format_rationale(
    tier: BrainTier,
    search_ids: list[str],
    model_ids: list[str],
    tools: Iterable[ToolSpec],
    models: Iterable[AIModelSpec],
    usage_state: dict[str, Any] | None = None,
) -> str:
    skipped = [
        f"{t.id} ({t.last_status.value})"
        for t in tools
        if t.enabled and t.last_status in {ProbeStatus.QUOTA_LIMITED, ProbeStatus.AUTH_MISSING, ProbeStatus.BLOCKED, ProbeStatus.ERROR}
    ]
    paid_only = [m.id for m in models if m.enabled and not evaluate_model_quota(m, usage_state).is_free_available]
    exhausted_t = [t.id for t in tools if t.enabled and not evaluate_tool_quota(t, usage_state).is_free_available]
    top_s = " → ".join(search_ids[:4]) or "–"
    top_m = " → ".join(model_ids[:3]) or "Heuristik"
    skip_note = f" | Übersprungen: {', '.join(skipped[:3])}" if skipped else ""
    paid_count = len(paid_only) + len(exhausted_t)
    paid_note = f" | Paid-Opt-In (Preis-Leistung): {paid_count}" if paid_count else ""
    return f"[{tier.value.upper()} · 0% Kosten · Best→Worst] Suche: {top_s} | Modelle: {top_m}{skip_note}{paid_note}"[:1150]


def optimize_tier_sequences(
    models: list[AIModelSpec],
    tools: list[ToolSpec],
    existing_sequences: dict[BrainTier, TierSequenceConfig],
    *,
    force_overwrite: bool = False,
    usage_state: dict[str, Any] | None = None,
) -> dict[BrainTier, TierSequenceConfig]:
    """Berechnet 0%-kostenfreie Sequenzen (und Paid-Opt-In-Ketten) für Fast, Standard & Deep."""
    search_tools = [t for t in tools if is_search_capable(t)]
    extractors = [t for t in tools if is_extractor_capable(t)]
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    result: dict[BrainTier, TierSequenceConfig] = {}

    for tier in (BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP):
        prev = existing_sequences.get(tier)
        if prev is not None and prev.manual_override and not force_overwrite:
            result[tier] = prev
            continue
        free_s = [t.id for t in _rank_tools_for_tier(search_tools, tier, usage_state=usage_state, allow_paid=False)]
        free_e = [t.id for t in _rank_tools_for_tier(extractors, tier, usage_state=usage_state, allow_paid=False)]
        free_m_objs = _rank_models_for_tier(models, tier, usage_state=usage_state, allow_paid=False)
        free_m = [m.id for m in free_m_objs]

        paid_s = [t.id for t in _rank_tools_for_tier(search_tools, tier, usage_state=usage_state, allow_paid=True)]
        paid_e = [t.id for t in _rank_tools_for_tier(extractors, tier, usage_state=usage_state, allow_paid=True)]
        paid_m_objs = _rank_models_for_tier(models, tier, usage_state=usage_state, allow_paid=True)
        paid_m = [m.id for m in paid_m_objs]

        has_free_deep = any(m.is_deep_agent and _is_usable_status(m.last_status) for m in free_m_objs)
        has_paid_deep = any(m.is_deep_agent and _is_usable_status(m.last_status) for m in paid_m_objs)
        rationale = _format_rationale(tier, free_s, free_m, tools, models, usage_state)

        if tier == BrainTier.FAST:
            cfg = TierSequenceConfig(
                tier=tier, search_tool_sequence=free_s, extractor_tool_sequence=free_e, model_sequence=free_m,
                paid_search_tool_sequence=paid_s, paid_extractor_tool_sequence=paid_e, paid_model_sequence=paid_m,
                max_search_tools=2, enable_page_scraping=False, max_pages_to_scrape=0, max_followup_rounds=0,
                single_pass=True, use_deep_agent=False, paid_use_deep_agent=False,
                optimization_rationale=rationale, updated_at=now,
            )
        elif tier == BrainTier.STANDARD:
            has_ext = bool(free_e or paid_e)
            cfg = TierSequenceConfig(
                tier=tier, search_tool_sequence=free_s, extractor_tool_sequence=free_e, model_sequence=free_m,
                paid_search_tool_sequence=paid_s, paid_extractor_tool_sequence=paid_e, paid_model_sequence=paid_m,
                max_search_tools=4, enable_page_scraping=has_ext, max_pages_to_scrape=2 if has_ext else 0,
                max_followup_rounds=1, single_pass=False, use_deep_agent=False, paid_use_deep_agent=False,
                optimization_rationale=rationale, updated_at=now,
            )
        else:
            has_ext = bool(free_e or paid_e)
            cfg = TierSequenceConfig(
                tier=tier, search_tool_sequence=free_s, extractor_tool_sequence=free_e, model_sequence=free_m,
                paid_search_tool_sequence=paid_s, paid_extractor_tool_sequence=paid_e, paid_model_sequence=paid_m,
                max_search_tools=max(6, len(free_s), len(paid_s)), enable_page_scraping=has_ext,
                max_pages_to_scrape=5 if has_ext else 0, max_followup_rounds=2, single_pass=False,
                use_deep_agent=has_free_deep, paid_use_deep_agent=has_paid_deep,
                optimization_rationale=rationale, updated_at=now,
            )
        result[tier] = cfg
    return result
