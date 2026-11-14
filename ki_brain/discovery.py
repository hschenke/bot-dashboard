"""Automatische Modell- & Tool-Erkennung (Gemini API, OpenRouter/Groq, Builtins & Manifest)."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from google import genai

from ki_brain.benchmark import BrainBenchmarkRunner
from ki_brain.catalog import build_default_models, build_default_tools
from ki_brain.enums import BrainTier, ModelProvider, ProbeStatus
from ki_brain.llm_runner import UnifiedLLMRunner
from ki_brain.models import AIModelSpec, BrainSecrets, ToolSpec
from ki_brain.security import validate_identifier
from ki_brain.storage import BrainRepository
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request

logger = logging.getLogger(__name__)

_GEMINI_SKIP_PARTS = (
    "embedding",
    "imagen",
    "veo",
    "aqa",
    "tts",
    "audio",
    "robotics",
    "gemini-1.",
    "-live",
    "image",
    "computer-use",
    "customtools",
    "thinking-exp",
)
_GEMINI_INCLUDE_PREFIXES = ("gemini-2.", "gemini-3.", "gemini-4.", "gemma-", "deep-research", "learnlm")


@dataclass(frozen=True)
class DiscoverySummary:
    """Ergebnis eines automatischen Katalog-Syncs."""

    added_models: list[str] = field(default_factory=list)
    added_tools: list[str] = field(default_factory=list)
    checked_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_excluded: list[str] = field(default_factory=list)
    skipped_unhealthy: list[str] = field(default_factory=list)


def _guess_gemini_tier(model_id: str) -> tuple[BrainTier, bool]:
    low = model_id.lower()
    if "deep-research" in low:
        return BrainTier.DEEP, True
    if "lite" in low:
        return BrainTier.FAST, False
    if "argon" in low or "pro" in low:
        return BrainTier.DEEP, False
    return BrainTier.STANDARD, False


def _sync_new_builtins(
    repo: BrainRepository,
    excluded_models: set[str],
    excluded_tools: set[str],
) -> tuple[list[str], list[str], list[str]]:
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
    skipped: list[str] = []
    for m in build_default_models():
        if m.id in excluded_models:
            skipped.append(m.id)
        elif m.id not in known_models and m.id not in existing_models:
            repo.upsert_model(m)
            added_m.append(m.id)
        known_models.add(m.id)

    added_t: list[str] = []
    for t in build_default_tools():
        if t.id in excluded_tools:
            skipped.append(t.id)
        elif t.id not in known_tools and t.id not in existing_tools:
            repo.upsert_tool(t)
            added_t.append(t.id)
        known_tools.add(t.id)

    repo.set_meta(
        "known_builtin_ids",
        json.dumps({"models": sorted(known_models), "tools": sorted(known_tools)}),
    )
    return added_m, added_t, skipped


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
    return discovered[:24]


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
    return specs[:24]


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
        if len(specs) >= 24:
            break
    return specs


def _sync_remote_manifest(
    repo: BrainRepository,
    manifest_url: str,
    excluded_tools: set[str],
    skipped_excluded: list[str],
) -> tuple[list[AIModelSpec], list[str]]:
    status, _, _, body = safe_http_request("GET", manifest_url, headers=API_HEADERS, timeout_seconds=6.0, max_bytes=300_000)
    if status != 200 or not body:
        return [], []
    data: dict[str, Any] = json.loads(body.decode("utf-8", errors="replace"))
    existing_t = {t.id for t in repo.list_tools()}
    candidate_models = [
        AIModelSpec.model_validate(raw_m).model_copy(update={"source": "auto_discovered"})
        for raw_m in data.get("models", [])
    ]
    added_t: list[str] = []
    for raw_t in data.get("tools", []):
        spec_t = ToolSpec.model_validate(raw_t).model_copy(update={"source": "auto_discovered"})
        if spec_t.id in excluded_tools:
            skipped_excluded.append(spec_t.id)
        elif spec_t.id not in existing_t:
            repo.upsert_tool(spec_t)
            existing_t.add(spec_t.id)
            added_t.append(spec_t.id)
    return candidate_models, added_t


def _probe_candidate_models(
    candidates: list[AIModelSpec],
    bench: BrainBenchmarkRunner,
) -> list[tuple[AIModelSpec, ProbeStatus]]:
    if not candidates:
        return []
    openai_items = [m for m in candidates if m.provider == ModelProvider.OPENAI_COMPATIBLE]
    gemini_items = [m for m in candidates if m.provider != ModelProvider.OPENAI_COMPATIBLE]
    results: list[tuple[AIModelSpec, ProbeStatus]] = []
    if openai_items:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for probed, entry in pool.map(bench.probe_model, openai_items):
                results.append((probed, entry.status))
    for spec in gemini_items:
        probed, entry = bench.probe_model(spec)
        results.append((probed, entry.status))
    return results


def auto_sync_catalog(
    repo: BrainRepository,
    secrets: BrainSecrets,
    *,
    genai_client: genai.Client | None = None,
    manifest_url: str = "",
    runner: UnifiedLLMRunner | None = None,
    verify_discovered: bool = True,
) -> DiscoverySummary:
    """Erkennt neue Builtins, Gemini-, Groq-, OpenRouter-Free-Modelle und schließt defekte/entfernte IDs aus."""
    excluded_models, excluded_tools = repo.list_excluded_ids()
    added_models, added_tools, skipped_excluded = _sync_new_builtins(repo, excluded_models, excluded_tools)
    sources = ["builtins"]
    warnings: list[str] = []
    skipped_unhealthy: list[str] = []
    existing_ids = {m.id for m in repo.list_models()}
    candidates: list[AIModelSpec] = []
    seen_candidates: set[str] = set()

    def _collect(items: list[AIModelSpec], limit: int = 8) -> None:
        collected_here = 0
        for spec in items:
            if spec.id in existing_ids or spec.id in seen_candidates:
                continue
            seen_candidates.add(spec.id)
            if spec.id in excluded_models:
                skipped_excluded.append(spec.id)
                continue
            candidates.append(spec)
            collected_here += 1
            if limit > 0 and collected_here >= limit:
                break

    gem_key = secrets.resolve_key("GEMINI_API_KEY")
    if genai_client is not None or gem_key:
        sources.append("gemini-models")
        try:
            client = genai_client or genai.Client(api_key=gem_key)
            _collect(_discover_gemini_models(client), limit=8)
        except Exception as exc:
            warnings.append(f"Gemini Auto-Discovery: {exc}")

    groq_key = secrets.resolve_key("GROQ_API_KEY")
    if groq_key:
        sources.append("groq-models")
        try:
            _collect(_discover_groq_models(groq_key), limit=6)
        except Exception as exc:
            warnings.append(f"Groq Auto-Discovery: {exc}")

    or_key = secrets.resolve_key("OPENROUTER_API_KEY")
    if or_key:
        sources.append("openrouter-free")
        try:
            _collect(_discover_openrouter_free_models(or_key), limit=6)
        except Exception as exc:
            warnings.append(f"OpenRouter Auto-Discovery: {exc}")

    if manifest_url.strip():
        sources.append("remote-manifest")
        try:
            rm_candidates, rt = _sync_remote_manifest(
                repo, manifest_url.strip(), excluded_tools, skipped_excluded
            )
            _collect(rm_candidates, limit=0)
            added_tools.extend(rt)
        except Exception as exc:
            warnings.append(f"Manifest-Sync: {exc}")

    if verify_discovered and candidates:
        probe_runner = runner or UnifiedLLMRunner(secrets, genai_client=genai_client)
        bench = BrainBenchmarkRunner(repo, secrets, probe_runner)
        for probed, status in _probe_candidate_models(candidates, bench):
            if status in {ProbeStatus.ERROR, ProbeStatus.QUOTA_LIMITED}:
                skipped_unhealthy.append(probed.id)
                continue
            repo.upsert_model(probed)
            existing_ids.add(probed.id)
            added_models.append(probed.id)
        if skipped_unhealthy:
            repo.add_excluded_ids(model_ids=skipped_unhealthy)
    else:
        for spec in candidates:
            repo.upsert_model(spec)
            existing_ids.add(spec.id)
            added_models.append(spec.id)

    return DiscoverySummary(
        added_models=added_models,
        added_tools=added_tools,
        checked_sources=sources,
        warnings=warnings,
        skipped_excluded=skipped_excluded,
        skipped_unhealthy=skipped_unhealthy,
    )
