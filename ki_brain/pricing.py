"""Modell-Kosten-Schätzung (USD je 1M Tokens) & Kapazitäts-Signale für kostenbewusstes Sequencing."""

from __future__ import annotations

import re

from ki_brain.enums import BrainTier, ModelProvider
from ki_brain.models import AIModelSpec

# Verifizierte Token-Preise (Input, Output in USD je 1M Tokens, Stand Nov 2026).
# Spezifischere Keys stehen vor allgemeineren Präfixen.
_KNOWN_TOKEN_RATES_PER_1M: tuple[tuple[str, tuple[float, float]], ...] = (
    ("gemini-2.0-flash-lite", (0.075, 0.30)),
    ("gemini-2.0-flash", (0.10, 0.40)),
    ("gemini-2.5-flash-lite", (0.10, 0.40)),
    ("gemini-2.5-flash", (0.30, 2.50)),
    ("gemini-2.5-pro", (1.25, 10.00)),
    ("gemini-3.1-flash-lite", (0.25, 1.50)),
    ("gemini-3.5-flash-lite", (0.30, 2.50)),
    ("gemini-3.0-flash", (0.50, 3.00)),
    ("gemini-3-flash", (0.50, 3.00)),
    ("gemini-3.6-flash", (0.75, 3.75)),
    ("gemini-3.7-flash", (0.75, 3.75)),
    ("gemini-3.8-flash", (0.75, 3.75)),
    ("gemini-3.9-flash", (0.75, 3.75)),
    ("gemini-3.1-pro", (2.00, 12.00)),
    ("gemini-3-pro", (2.00, 12.00)),
    ("gemini-4-hydrogen", (1.50, 7.50)),
    ("gemini-4-argon", (2.00, 10.00)),
    ("gemini-4-carbon", (2.00, 10.00)),
    ("deep-research", (2.00, 12.00)),
    ("llama-3.1-8b-instant", (0.05, 0.08)),
    ("gpt-oss-20b", (0.075, 0.30)),
    ("gpt-oss-120b", (0.15, 0.60)),
    ("llama-3.3-70b", (0.59, 0.79)),
    ("llama-4-scout", (0.11, 0.34)),
    ("llama-4-maverick", (0.20, 0.60)),
    ("qwen-qwq-32b", (0.29, 0.59)),
    ("qwen3-32b", (0.29, 0.59)),
    ("deepseek-r1-distill", (0.75, 0.99)),
)

_SMALL_MODEL_RE = re.compile(r"(?:^|[-_/: ])(?:0\.5b|[1-9]b|nano|micro)(?:$|[-_/: ])")
_MODERN_CAPABLE_TOKENS = (
    "gemini-3.5",
    "gemini-3.6",
    "gemini-3.7",
    "gemini-3.8",
    "gemini-3.9",
    "gemini-4",
    "70b",
    "120b",
)


def is_small_capacity_model(model: AIModelSpec) -> bool:
    """Erkennt sehr kleine Modelle (<=9B, Nano/Micro), damit sie starke Modelle in Standard/Deep nicht verdrängen."""
    haystack = f"{model.id} {model.model_id}".lower()
    return bool(_SMALL_MODEL_RE.search(haystack))


def compute_capability_adjustment(model: AIModelSpec) -> float:
    """Moderater Qualitäts-Ausgleich nach Modellkapazität (-6 für <=9B, +1/+2 für moderne Flash/70B/G4)."""
    if is_small_capacity_model(model):
        return -6.0
    haystack = f"{model.id} {model.model_id}".lower()
    if any(tok in haystack for tok in _MODERN_CAPABLE_TOKENS):
        return 2.0
    if "gemini-2.5-flash" in haystack and "lite" not in haystack:
        return 1.0
    return 0.0


def resolve_model_token_costs(model: AIModelSpec) -> tuple[float, float]:
    """Ermittelt (Input-USD/1M, Output-USD/1M) aus explizitem Wert, Preistabelle oder Tier-Heuristik."""
    if model.input_cost_per_1m is not None or model.output_cost_per_1m is not None:
        return float(model.input_cost_per_1m or 0.0), float(model.output_cost_per_1m or 0.0)

    slug = f"{model.id} {model.model_id}".lower()
    if ":free" in slug or (model.id.lower().startswith("or/") and model.id.lower().endswith("-free")):
        return 0.0, 0.0
    if "gemma-" in slug:
        return 0.0, 0.0

    for key, rates in _KNOWN_TOKEN_RATES_PER_1M:
        if key in slug:
            return rates

    if "flash-lite" in slug:
        return 0.25, 1.25
    if "flash" in slug:
        return 0.60, 3.00
    if any(tok in slug for tok in ("-pro", "argon", "carbon", "ultra")):
        return 2.00, 10.00
    if "hydrogen" in slug:
        return 1.50, 7.50
    if model.provider == ModelProvider.OPENAI_COMPATIBLE:
        if is_small_capacity_model(model) or "mini" in slug:
            return 0.06, 0.12
        if "70b" in slug:
            return 0.59, 0.79

    if model.tier_hint == BrainTier.FAST:
        return 0.25, 1.25
    if model.tier_hint == BrainTier.DEEP:
        return 2.00, 10.00
    return 0.75, 3.50


_PAID_ONLY_PRO_RE = re.compile(r"(?:^|[-_/: ])pro(?:$|[-_/: ])")
_PAID_ONLY_MARKERS = ("deep-research", "argon", "carbon", "ultra")


def infer_model_is_free(
    model_id: str,
    name: str = "",
    *,
    provider: ModelProvider = ModelProvider.GOOGLE_GEMINI,
    input_cost_per_1m: float | None = None,
    output_cost_per_1m: float | None = None,
) -> bool:
    """Ermittelt, ob ein Modell im Free-Tier / gratis nutzbar ist (vs. strikt Paid-Only)."""
    slug = f"{model_id} {name}".lower()
    if ":free" in slug or "gemma-" in slug or (model_id.lower().startswith("or/") and model_id.lower().endswith("-free")):
        return True
    if any(tok in slug for tok in _PAID_ONLY_MARKERS) or _PAID_ONLY_PRO_RE.search(slug):
        return False
    return not (
        provider == ModelProvider.OPENAI_COMPATIBLE
        and (float(input_cost_per_1m or 0.0) > 0.0 or float(output_cost_per_1m or 0.0) > 0.0)
    )


def estimate_blended_cost_per_1m(model: AIModelSpec) -> float:
    """Berechnet geschätzte Mischkosten je 1M Tokens (75 % Input + 25 % Output in USD)."""
    in_cost, out_cost = resolve_model_token_costs(model)
    return round(0.75 * in_cost + 0.25 * out_cost, 2)


def compute_cost_score(model: AIModelSpec) -> float:
    """Normiert Modellkosten auf 0..100 Punkte (günstiger = höher)."""
    blended = estimate_blended_cost_per_1m(model)
    return round(100.0 / (1.0 + 0.50 * max(0.0, blended)), 1)


def format_model_budget_label(model: AIModelSpec) -> str:
    """Kurze Budget-Einstufung für die KI-Modelle-Tabelle."""
    if not model.is_free:
        return "💳 Paid-Only"
    if estimate_blended_cost_per_1m(model) == 0.0:
        return "🆓 $0 Forever"
    return "🆓 Free Tier"

