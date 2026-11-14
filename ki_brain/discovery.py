"""Automatische Modell- & Tool-Erkennung (Gemini API, OpenRouter/Groq, Builtins & Manifest)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai

from ki_brain.catalog import build_default_models, build_default_tools
from ki_brain.enums import BrainTier, ModelProvider
from ki_brain.models import AIModelSpec, BrainSecrets, ToolSpec
from ki_brain.security import validate_identifier
from ki_brain.storage import BrainRepository
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request

logger = logging.getLogger(__name__)

_GEMINI_SKIP_PARTS = ("embedding", "imagen", "veo", "aqa", "tts", "audio", "robotics", "gemini-1.")
_GEMINI_INCLUDE_PREFIXES = ("gemini-2.", "gemini-3.", "gemini-4.", "gemma-", "deep-research", "learnlm")


@dataclass(frozen=True)
class DiscoverySummary:
    """Ergebnis eines automatischen Katalog-Syncs."""

    added_models: list[str] = field(default_factory=list)
    added_tools: list[str] = field(default_factory=list)
    checked_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _guess_gemini_tier(model_id: str) -> tuple[BrainTier, bool]:
    low = model_id.lower()
    if "deep-research" in low:
        return BrainTier.DEEP, True
    if "lite" in low:
        return BrainTier.FAST, False
    if "argon" in low or "pro" in low:
        return BrainTier.DEEP, False
    return BrainTier.STANDARD, False


def _sync_new_builtins(repo: BrainRepository) -> tuple[list[str], list[str]]:
    raw_meta = repo.get_meta("known_builtin_ids", "{}")
    try:
        known = json.loads(raw_meta)
    except json.JSONDecodeError:
        known = {}
    known_models = set(known.get("models", []))
    known_tools = set(known.get("tools", []))
    existing_models = {m.id for m in repo.list_models()}
    existing_tools = {t.id for t in repo.list_tools()}

    added_m: list[str] = []
    for m in build_default_models():
        if m.id not in known_models and m.id not in existing_models:
            repo.upsert_model(m)
            added_m.append(m.id)
        known_models.add(m.id)

    added_t: list[str] = []
    for t in build_default_tools():
        if t.id not in known_tools and t.id not in existing_tools:
            repo.upsert_tool(t)
            added_t.append(t.id)
        known_tools.add(t.id)

    repo.set_meta(
        "known_builtin_ids",
        json.dumps({"models": sorted(known_models), "tools": sorted(known_tools)}),
    )
    return added_m, added_t


def _discover_gemini_models(client: genai.Client) -> list[AIModelSpec]:
    discovered: list[AIModelSpec] = []
    for raw in client.models.list():
        raw_name = str(getattr(raw, "name", "") or "").strip()
        model_id = raw_name.removeprefix("models/").strip()
        low = model_id.lower()
        if not low or any(skip in low for skip in _GEMINI_SKIP_PARTS):
            continue
        if not low.startswith(_GEMINI_INCLUDE_PREFIXES):
            continue
        actions = [str(a) for a in (getattr(raw, "supported_actions", None) or [])]
        if actions and not any(act in actions for act in ("generateContent", "streamGenerateContent")) and "deep-research" not in low:
            continue
        tier, is_deep = _guess_gemini_tier(model_id)
        display = str(getattr(raw, "display_name", "") or model_id).strip()[:120]
        discovered.append(
            AIModelSpec(
                id=validate_identifier(model_id, "Modell-ID"),
                name=display,
                provider=ModelProvider.GEMINI_DEEP_AGENT if is_deep else ModelProvider.GOOGLE_GEMINI,
                model_id=model_id,
                tier_hint=tier,
                supports_structured_output=not is_deep,
                supports_google_search=not is_deep,
                supports_url_context=not is_deep,
                is_deep_agent=is_deep,
                max_rpm=4 if is_deep else 15,
                source="auto_discovered",
                origin_apps=["gemini-api-discovery"],
                notes="Automatisch über Gemini API (`models.list`) erkannt.",
            )
        )
    return discovered


def _discover_groq_models(api_key: str) -> list[AIModelSpec]:
    status, _, _, body = safe_http_request(
        "GET",
        "https://api.groq.com/openai/v1/models",
        headers={**API_HEADERS, "Authorization": f"Bearer {api_key}"},
        timeout_seconds=5.0,
        max_bytes=200_000,
    )
    if status != 200 or not body:
        return []
    items = json.loads(body.decode("utf-8", errors="replace")).get("data", [])
    specs: list[AIModelSpec] = []
    for entry in items:
        mid = str(entry.get("id") or "").strip()
        if not mid or any(bad in mid.lower() for bad in ("whisper", "guard", "tts")):
            continue
        tier = BrainTier.FAST if ("8b" in mid.lower() or "instant" in mid.lower()) else BrainTier.STANDARD
        specs.append(
            AIModelSpec(
                id=validate_identifier(f"groq/{mid}"[:94], "Groq-Modell-ID"),
                name=f"Groq: {mid}"[:120],
                provider=ModelProvider.OPENAI_COMPATIBLE,
                model_id=mid,
                tier_hint=tier,
                api_key_env="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                supports_google_search=False,
                supports_url_context=False,
                max_rpm=25,
                source="auto_discovered",
                origin_apps=["groq-discovery"],
                notes="Kostenfreier Groq-OpenAI-Compat-Endpunkt.",
            )
        )
    return specs[:6]


def _discover_openrouter_free_models(api_key: str) -> list[AIModelSpec]:
    status, _, _, body = safe_http_request(
        "GET",
        "https://openrouter.ai/api/v1/models",
        headers={**API_HEADERS, "Authorization": f"Bearer {api_key}"},
        timeout_seconds=6.0,
        max_bytes=500_000,
    )
    if status != 200 or not body:
        return []
    specs: list[AIModelSpec] = []
    for entry in json.loads(body.decode("utf-8", errors="replace")).get("data", []):
        mid = str(entry.get("id") or "").strip()
        pricing = entry.get("pricing") or {}
        is_zero = str(pricing.get("prompt", "")) == "0" and str(pricing.get("completion", "")) == "0"
        if not (mid.endswith(":free") or is_zero):
            continue
        safe_id = validate_identifier(f"or/{mid.replace(':', '-')}"[:94], "OpenRouter-ID")
        specs.append(
            AIModelSpec(
                id=safe_id,
                name=f"OpenRouter Free: {str(entry.get('name') or mid)[:95]}",
                provider=ModelProvider.OPENAI_COMPATIBLE,
                model_id=mid,
                tier_hint=BrainTier.STANDARD,
                api_key_env="OPENROUTER_API_KEY",
                base_url="https://openrouter.ai/api/v1",
                supports_google_search=False,
                supports_url_context=False,
                max_rpm=15,
                source="auto_discovered",
                origin_apps=["openrouter-discovery"],
                notes="Automatisch erkanntes `:free`-Modell über OpenRouter.",
            )
        )
        if len(specs) >= 6:
            break
    return specs


def _sync_remote_manifest(repo: BrainRepository, manifest_url: str) -> tuple[list[str], list[str]]:
    status, _, _, body = safe_http_request("GET", manifest_url, headers=API_HEADERS, timeout_seconds=6.0, max_bytes=300_000)
    if status != 200 or not body:
        return [], []
    data: dict[str, Any] = json.loads(body.decode("utf-8", errors="replace"))
    existing_m = {m.id for m in repo.list_models()}
    existing_t = {t.id for t in repo.list_tools()}
    added_m, added_t = [], []
    for raw_m in data.get("models", []):
        spec = AIModelSpec.model_validate(raw_m).model_copy(update={"source": "auto_discovered"})
        if spec.id not in existing_m:
            repo.upsert_model(spec)
            added_m.append(spec.id)
    for raw_t in data.get("tools", []):
        spec_t = ToolSpec.model_validate(raw_t).model_copy(update={"source": "auto_discovered"})
        if spec_t.id not in existing_t:
            repo.upsert_tool(spec_t)
            added_t.append(spec_t.id)
    return added_m, added_t


def auto_sync_catalog(
    repo: BrainRepository,
    secrets: BrainSecrets,
    *,
    genai_client: genai.Client | None = None,
    manifest_url: str = "",
) -> DiscoverySummary:
    """Erkennt neue Builtins, Gemini-, Groq-, OpenRouter-Free-Modelle und optionale Manifeste."""
    added_models, added_tools = _sync_new_builtins(repo)
    sources = ["builtins"]
    warnings: list[str] = []
    existing_ids = {m.id for m in repo.list_models()}

    def _register_new(items: list[AIModelSpec]) -> None:
        for spec in items:
            if spec.id not in existing_ids:
                repo.upsert_model(spec)
                existing_ids.add(spec.id)
                added_models.append(spec.id)

    gem_key = secrets.resolve_key("GEMINI_API_KEY")
    if genai_client is not None or gem_key:
        sources.append("gemini-models")
        try:
            client = genai_client or genai.Client(api_key=gem_key)
            _register_new(_discover_gemini_models(client))
        except Exception as exc:
            warnings.append(f"Gemini Auto-Discovery: {exc}")

    groq_key = secrets.resolve_key("GROQ_API_KEY")
    if groq_key:
        sources.append("groq-models")
        try:
            _register_new(_discover_groq_models(groq_key))
        except Exception as exc:
            warnings.append(f"Groq Auto-Discovery: {exc}")

    or_key = secrets.resolve_key("OPENROUTER_API_KEY")
    if or_key:
        sources.append("openrouter-free")
        try:
            _register_new(_discover_openrouter_free_models(or_key))
        except Exception as exc:
            warnings.append(f"OpenRouter Auto-Discovery: {exc}")

    if manifest_url.strip():
        sources.append("remote-manifest")
        try:
            rm, rt = _sync_remote_manifest(repo, manifest_url.strip())
            added_models.extend(rm)
            added_tools.extend(rt)
        except Exception as exc:
            warnings.append(f"Manifest-Sync: {exc}")

    return DiscoverySummary(
        added_models=added_models,
        added_tools=added_tools,
        checked_sources=sources,
        warnings=warnings,
    )
