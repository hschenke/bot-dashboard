"""Provider-neutrale Modellstärke-Schätzung (Capability) & Tier-Erkennung für das Sequencing."""

from __future__ import annotations

import re

from ai_brain.enums import BrainTier
from ai_brain.models import AIModelSpec

_GEMINI_VER_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")
_PRO_WORD_RE = re.compile(r"(?:^|[-_/: ])pro(?:$|[-_/: ])")
_MOE_PARAMS_RE = re.compile(r"(?:^|[-_/: ])(\d+)x(\d+(?:\.\d+)?)b(?:$|[-_/: ])")
_PARAM_B_RE = re.compile(r"(?:^|[-_/: ])(\d+(?:\.\d+)?)b(?:$|[-_/: ])")
_SMALL_MODEL_RE = re.compile(r"(?:^|[-_/: ])(?:0\.\d+b|[1-9](?:\.\d+)?b|nano|micro)(?:$|[-_/: ])")
_REASONING_RE = re.compile(r"(?:^|[-_/: ])(?:qwq|r1|thinking|reasoner|o1|o3|o4)(?:$|[-_/: ])")
_DEEP_FLAGSHIP_TOKENS = ("deep-research", "argon", "carbon", "ultra", "opus")
_FAST_CLASS_RE = re.compile(r"(?:^|[-_/: ])(?:lite|mini|instant|haiku)(?:$|[-_/: ])")
_MINI_OR_SMALL_RE = re.compile(r"(?:^|[-_/: ])(?:mini|small)(?:$|[-_/: ])")
_LARGE_WORD_RE = re.compile(r"(?:^|[-_/: ])large(?:$|[-_/: ])")
_LITE_WORD_RE = re.compile(r"(?:^|[-_/: ])lite(?:$|[-_/: ])")


def _model_slug(model: AIModelSpec) -> str:
    return f"{model.id} {model.model_id}".lower()


def _extract_effective_params_b(slug: str) -> float | None:
    moe = _MOE_PARAMS_RE.search(slug)
    if moe:
        experts = int(moe.group(1))
        per_expert = float(moe.group(2))
        return round(per_expert * (1.0 + 0.45 * max(0, experts - 1)), 1)
    matches = [float(m.group(1)) for m in _PARAM_B_RE.finditer(slug)]
    return max(matches) if matches else None


def is_small_capacity_model(model: AIModelSpec) -> bool:
    """Erkennt sehr kleine Modelle (<=9B, Nano/Micro), damit sie stärkere Modelle nicht verdrängen."""
    return bool(_SMALL_MODEL_RE.search(_model_slug(model)))


def _score_gemini_capability(slug: str, major: int, minor: int) -> float:
    gen = major + min(minor, 15) * 0.1
    if any(tok in slug for tok in ("argon", "carbon", "ultra")) or _PRO_WORD_RE.search(slug):
        return min(96.5, 72.0 + gen * 5.8)
    if "hydrogen" in slug:
        return min(95.0, 68.0 + gen * 5.8)
    if _LITE_WORD_RE.search(slug):
        return min(80.0, 49.0 + gen * 6.0)
    return min(90.0, 60.0 + gen * 6.0)


def _score_named_family_capability(slug: str) -> float | None:
    if "llama-4-maverick" in slug:
        return 83.0
    if "llama-4-scout" in slug:
        return 77.0
    if "compound" in slug and "groq" in slug:
        return 78.0
    if "claude" in slug:
        if "opus" in slug:
            return 93.0
        if "haiku" in slug:
            return 72.0
        return 88.0
    if "gpt-" in slug and "gpt-oss" not in slug:
        if _SMALL_MODEL_RE.search(slug):
            return 60.0
        if _MINI_OR_SMALL_RE.search(slug):
            return 72.0
        if "gpt-5" in slug:
            return 94.0
        return 86.0
    if "deepseek-r1" in slug and "distill" not in slug and _extract_effective_params_b(slug) is None:
        return 88.0
    if "deepseek-v3" in slug:
        return 86.0
    return None


def _base_param_capability(params_b: float) -> float:
    if params_b >= 400.0:
        return 89.0
    if params_b >= 200.0:
        return 86.5
    if params_b >= 100.0:
        return 84.0
    if params_b >= 65.0:
        return 80.0
    if params_b >= 27.0:
        return min(76.0, 72.0 + (params_b - 27.0) * 0.25)
    if params_b >= 18.0:
        return min(69.0, 64.0 + (params_b - 18.0) * 0.6)
    if params_b >= 10.0:
        return min(63.0, 58.0 + (params_b - 10.0) * 0.8)
    if params_b >= 6.0:
        return 46.0 + (params_b - 6.0) * 2.0
    return max(32.0, 36.0 + params_b * 1.5)


def _score_param_sized_capability(slug: str, params_b: float) -> float:
    score = _base_param_capability(params_b)
    if "gemma-4" in slug:
        score += 2.5
    elif "gemma-3" in slug:
        score += 0.5
    elif "llama-4" in slug or "qwen3" in slug or "qwen-3" in slug:
        score += 2.0
    elif any(tok in slug for tok in ("llama-3.3", "qwen-2.5", "qwen2.5", "gpt-oss")):
        score += 1.0
    if params_b >= 14.0 and _REASONING_RE.search(slug):
        score += 2.0
    return score


def estimate_model_capability(model: AIModelSpec) -> float:
    """Schätzt die reine Modellstärke (20..98, höher = stärker) strikt provider-neutral."""
    slug = _model_slug(model)
    if model.is_deep_agent or "deep-research" in slug:
        return 98.0

    gem = _GEMINI_VER_RE.search(slug)
    if gem:
        score = _score_gemini_capability(slug, int(gem.group(1)), int(gem.group(2) or 0))
    elif (named := _score_named_family_capability(slug)) is not None:
        score = named
    elif (params_b := _extract_effective_params_b(slug)) is not None:
        score = _score_param_sized_capability(slug, params_b)
    elif _SMALL_MODEL_RE.search(slug):
        score = 44.0
    elif _MINI_OR_SMALL_RE.search(slug):
        score = 66.0
    elif _LARGE_WORD_RE.search(slug):
        score = 78.0
    elif model.tier_hint == BrainTier.DEEP:
        score = 82.0
    elif model.tier_hint == BrainTier.STANDARD:
        score = 70.0
    else:
        score = 60.0

    return round(max(20.0, min(98.0, score)), 1)


def infer_model_tier(model_slug: str, *, is_deep_agent: bool = False) -> BrainTier:
    """Ordnet ein Modell konsistent in Fast, Standard oder Deep ein (provider-neutral)."""
    low = (model_slug or "").lower()
    if is_deep_agent or any(tok in low for tok in _DEEP_FLAGSHIP_TOKENS) or _PRO_WORD_RE.search(low):
        return BrainTier.DEEP
    if "llama-4-maverick" in low or "llama-4-scout" in low:
        return BrainTier.STANDARD
    if _SMALL_MODEL_RE.search(low) or _FAST_CLASS_RE.search(low):
        return BrainTier.FAST
    params_b = _extract_effective_params_b(low)
    if params_b is not None and params_b <= 20.0:
        return BrainTier.FAST
    return BrainTier.STANDARD


def resolve_effective_tier(model: AIModelSpec) -> BrainTier:
    """Ermittelt die wirksame Modell-Stufe (respektiert manuelle `custom`-Einträge)."""
    if model.is_deep_agent:
        return BrainTier.DEEP
    if model.source == "custom":
        return model.tier_hint
    return infer_model_tier(_model_slug(model), is_deep_agent=model.is_deep_agent)


def is_fast_class_model(model: AIModelSpec) -> bool:
    """Prüft, ob ein Modell zur leichten/schnellen Fast-Klasse gehört."""
    return resolve_effective_tier(model) == BrainTier.FAST


def compute_capability_adjustment(model: AIModelSpec) -> float:
    """Kompatibler, zentrierter Kapazitäts-Offset gegenüber einem 70-Punkte-Standardmodell."""
    return round(estimate_model_capability(model) - 70.0, 1)
