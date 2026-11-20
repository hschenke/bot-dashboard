"""Free-Kontingent-Tracking, Grounding-Modell-Auswahl & Geo-Smart-Erkennung für 0%-Kosten-Betrieb."""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ki_brain.enums import ModelProvider, ProbeStatus
from ki_brain.models import AIModelSpec, BrainSecrets, ToolSpec

if TYPE_CHECKING:
    from ki_brain.storage import BrainRepository

USAGE_QUOTAS_META_KEY = "usage_quotas_v1"

_GEMINI_GEN3_RE = re.compile(r"gemini-(?:[3-9]|[1-9]\d)(?:\.[0-9]+)?\b", re.IGNORECASE)
_GEO_PHRASES = (
    "in der nähe",
    "in meiner nähe",
    "near me",
    "nearby",
    "google maps",
    "route nach",
    "wegbeschreibung",
    "öffnungszeiten",
    "entfernung",
    "anfahrt",
    "wo ist",
    "wo finde ich",
    "sehenswürdigkeiten",
    "stadtplan",
    "bahnhof",
    "flughafen",
    "restaurant",
    "café",
    "cafe",
    "bäckerei",
    "apotheke",
    "tankstelle",
    "parkplatz",
    "supermarkt",
    "hotel",
    "krankenhaus",
    "standort",
    "adresse von",
    "plz",
    "postleitzahl",
    "directions to",
    "hours for",
)
_GEO_WORD_RE = re.compile(
    r"\b(?:berlin|münchen|munich|hamburg|köln|cologne|frankfurt|stuttgart|düsseldorf|leipzig|dresden|"
    r"wien|vienna|zürich|zurich|london|paris|new york|maps|adresse|straße|strasse|platz|allee)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuotaSnapshot:
    """Aktueller Verbrauchs- & Kosten-Status eines Tools oder Modells in der laufenden Periode."""

    item_id: str
    is_free: bool
    limit: int
    period: str
    unit: str
    used: int
    remaining: int
    period_key: str
    is_free_available: bool
    exhausted: bool
    budget_ratio: float


def current_period_key(period: str, now: datetime.datetime | None = None) -> str:
    """Liefert den periodenspezifischen Reset-Schlüssel (`YYYY-MM`, `YYYY-MM-DD`, `lifetime`)."""
    ts = now or datetime.datetime.now(datetime.UTC)
    clean = (period or "unlimited").strip().lower()
    if clean == "monthly":
        return ts.strftime("%Y-%m")
    if clean == "daily":
        return ts.strftime("%Y-%m-%d")
    if clean == "lifetime":
        return "lifetime"
    return clean or "unlimited"


def parse_usage_quotas(raw_json: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Parst gespeicherte Kontingent-Zähler defensiv aus `brain_meta`."""
    try:
        data = json.loads(raw_json or "{}")
    except json.JSONDecodeError:
        return {"tools": {}, "models": {}}
    if not isinstance(data, dict):
        return {"tools": {}, "models": {}}
    tools = data.get("tools") if isinstance(data.get("tools"), dict) else {}
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    return {"tools": dict(tools), "models": dict(models)}


def _evaluate_quota(
    item_id: str,
    *,
    is_free: bool,
    limit: int,
    period: str,
    unit: str,
    bucket: dict[str, Any],
    now: datetime.datetime | None = None,
) -> QuotaSnapshot:
    clean_period = (period or ("unlimited" if is_free else "none")).strip().lower()
    p_key = current_period_key(clean_period, now=now)
    entry = bucket.get(item_id) if isinstance(bucket.get(item_id), dict) else {}
    stored_period = str(entry.get("period_key") or "")
    used = int(entry.get("used") or 0) if (stored_period == p_key and int(entry.get("used") or 0) > 0) else 0

    if not is_free or clean_period == "none":
        return QuotaSnapshot(
            item_id=item_id, is_free=False, limit=0, period="none", unit=unit,
            used=used, remaining=0, period_key=p_key, is_free_available=False, exhausted=True, budget_ratio=0.0,
        )
    if limit <= 0 or clean_period == "unlimited":
        return QuotaSnapshot(
            item_id=item_id, is_free=True, limit=0, period="unlimited", unit=unit,
            used=used, remaining=0, period_key="unlimited", is_free_available=True, exhausted=False, budget_ratio=1.0,
        )
    remaining = max(0, limit - used)
    exhausted = remaining == 0
    return QuotaSnapshot(
        item_id=item_id, is_free=True, limit=limit, period=clean_period, unit=unit,
        used=used, remaining=remaining, period_key=p_key,
        is_free_available=not exhausted, exhausted=exhausted, budget_ratio=round(remaining / limit, 4),
    )


def evaluate_tool_quota(
    tool: ToolSpec, usage_state: dict[str, Any] | None = None, *, now: datetime.datetime | None = None
) -> QuotaSnapshot:
    """Berechnet Restkontingent & 0%-Kosten-Verfügbarkeit eines Tools."""
    bucket = (usage_state or {}).get("tools") or {}
    return _evaluate_quota(
        tool.id, is_free=tool.is_free, limit=tool.free_quota_limit, period=tool.free_quota_period,
        unit=tool.free_quota_unit, bucket=bucket if isinstance(bucket, dict) else {}, now=now,
    )


def evaluate_model_quota(
    model: AIModelSpec, usage_state: dict[str, Any] | None = None, *, now: datetime.datetime | None = None
) -> QuotaSnapshot:
    """Berechnet Restkontingent & 0%-Kosten-Verfügbarkeit eines KI-Modells."""
    bucket = (usage_state or {}).get("models") or {}
    return _evaluate_quota(
        model.id, is_free=model.is_free, limit=model.free_quota_limit, period=model.free_quota_period,
        unit=model.free_quota_unit, bucket=bucket if isinstance(bucket, dict) else {}, now=now,
    )


def is_quota_limited_tool(tool: ToolSpec) -> bool:
    """Erkennt Tools mit begrenztem Free-Kontingent (Monat/Tag/Lifetime statt Unlimited)."""
    period = (tool.free_quota_period or "unlimited").strip().lower()
    return tool.free_quota_limit > 0 and period not in {"unlimited", "none"}


def is_gemini_3_or_newer(model: AIModelSpec) -> bool:
    """Prüft auf Gemini 3+ (`gemini-3*`, `gemini-4*`), wo 5.000/Mo Grounding-Freikontingent gilt."""
    return bool(_GEMINI_GEN3_RE.search(f"{model.id} {model.model_id}"))


def select_grounding_model(
    models: list[AIModelSpec],
    secrets: BrainSecrets,
    *,
    executor_type: str = "gemini_google_search",
    allow_paid: bool = False,
) -> AIModelSpec | None:
    """Wählt das passende Gemini-Modell für Grounding (Maps erzwingt bei `allow_paid=False` Gemini 3+)."""
    need_gen3 = executor_type == "gemini_google_maps" and not allow_paid
    candidates = [
        m for m in models
        if m.enabled
        and m.provider == ModelProvider.GOOGLE_GEMINI
        and not m.is_deep_agent
        and secrets.has_model_key(m, allow_paid=allow_paid)
        and (allow_paid or m.is_free)
        and (not need_gen3 or is_gemini_3_or_newer(m))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda m: (
            1 if m.last_status == ProbeStatus.OK else (0 if m.last_status == ProbeStatus.UNTESTED else -1),
            1 if is_gemini_3_or_newer(m) else 0,
            1 if "lite" in m.id.lower() else 0,
            -(m.last_latency_ms or 1500),
        ),
        reverse=True,
    )
    return candidates[0]


def is_geo_or_maps_query(query: str) -> bool:
    """Erkennt Orts-, Adress-, Routen- oder Places-Bezug, um das 5.000/Mo-Maps-Kontingent zu schonen."""
    low = f" {(query or '').lower()} "
    if any(phrase in low for phrase in _GEO_PHRASES):
        return True
    return bool(_GEO_WORD_RE.search(low))


def record_quota_usage(
    repo: BrainRepository,
    *,
    tool_counts: dict[str, int] | None = None,
    model_counts: dict[str, int] | None = None,
    now: datetime.datetime | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Inkrementiert verbrauchte Free-Kontingent-Zähler atomar im `brain_meta`-Store."""
    state = parse_usage_quotas(repo.get_meta(USAGE_QUOTAS_META_KEY, "{}"))
    ts = now or datetime.datetime.now(datetime.UTC)
    if tool_counts:
        tools_by_id = {t.id: t for t in repo.list_tools()}
        for tid, delta in tool_counts.items():
            if delta <= 0 or tid not in tools_by_id:
                continue
            spec = tools_by_id[tid]
            p_key = current_period_key(spec.free_quota_period, now=ts)
            prev = state["tools"].get(tid) if isinstance(state["tools"].get(tid), dict) else {}
            base = int(prev.get("used") or 0) if prev.get("period_key") == p_key else 0
            state["tools"][tid] = {"used": base + int(delta), "period_key": p_key, "updated_at": ts.isoformat(timespec="seconds")}
    if model_counts:
        models_by_id = {m.id: m for m in repo.list_models()}
        for mid, delta in model_counts.items():
            if delta <= 0 or mid not in models_by_id:
                continue
            mspec = models_by_id[mid]
            p_key = current_period_key(mspec.free_quota_period, now=ts)
            prev = state["models"].get(mid) if isinstance(state["models"].get(mid), dict) else {}
            base = int(prev.get("used") or 0) if prev.get("period_key") == p_key else 0
            state["models"][mid] = {"used": base + int(delta), "period_key": p_key, "updated_at": ts.isoformat(timespec="seconds")}
    repo.set_meta(USAGE_QUOTAS_META_KEY, json.dumps(state))
    return state


def format_quota_label(snap: QuotaSnapshot, *, paid_cost_per_1k: float = 0.0) -> str:
    """Kompakte UI-Anzeige für Free-Kontingente und Paid-Schwelle."""
    if not snap.is_free:
        return f"💳 Paid-Only (${paid_cost_per_1k:g}/1k)" if paid_cost_per_1k > 0 else "💳 Paid-Only"
    if snap.limit <= 0 or snap.period == "unlimited":
        return "♾️ $0 (Unlimited)"
    suffix = "/Mo" if snap.period == "monthly" else ("/Tag" if snap.period == "daily" else " Total")
    if snap.exhausted:
        rate = f" · ${paid_cost_per_1k:g}/1k" if paid_cost_per_1k > 0 else ""
        return f"🔴 0/{snap.limit}{suffix} (Paid{rate})"
    ratio = snap.remaining / snap.limit
    icon = "🎟️" if ratio > 0.2 else "🟠"
    return f"{icon} {snap.remaining}/{snap.limit}{suffix}"
