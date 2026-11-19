"""Automatische Modell- & Tool-Erkennung (Gemini API, OpenRouter/Groq, Builtins & Manifest)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from google import genai

from ki_brain.benchmark import BrainBenchmarkRunner
from ki_brain.catalog import build_default_models, build_default_tools
from ki_brain.enums import BrainTier, ModelProvider, ProbeStatus, ToolCategory
from ki_brain.llm_runner import UnifiedLLMRunner
from ki_brain.models import AIModelSpec, BrainSecrets, ToolSpec
from ki_brain.optimizer import optimize_tier_sequences
from ki_brain.pricing import infer_model_is_free
from ki_brain.quotas import select_grounding_model
from ki_brain.security import validate_identifier
from ki_brain.storage import BrainRepository
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

_GEMINI_SKIP_PARTS = (
    "embedding",
    "imagen",
    "veo",
    "aqa",
    "tts",
    "audio",
    "robotics",
    "gemini-1.",
    "gemini-2.0",
    "gemini-pro",
    "-live",
    "image",
    "computer-use",
    "customtools",
    "thinking-exp",
)
_GEMINI_TEXT_SERIES_RE = re.compile(
    r"^(?:gemini-(?:2\.5|3(?:\.[0-9]+)?|4(?:\.[0-9]+)?|5(?:\.[0-9]+)?|[3-9]-)|gemma-(?:3|4)|deep-research|learnlm)",
    re.IGNORECASE,
)
_CANONICAL_DEEP_RESEARCH_ID = "deep-research-preview-04-2026"
_PREVIEW_MARKERS = (
    "preview",
    "-exp",
    "_exp",
    "(exp)",
    "experimental",
    "-alpha",
    "-beta",
    "(beta)",
    "[rerouted]",
    "rerouted",
    "deprecated",
)
_VERSION_RE = re.compile(r"(?:gemini|gemma)-(\d+)(?:\.(\d+))?")
_DATED_CLONE_RE = re.compile(r"-(?:00[1-9]|202\d|\d{2}-\d{2,4})$")


def is_preview_or_experimental_model(model_id: str, display_name: str = "") -> bool:
    """Erkennt Preview-/Experimental-Modelle (ausgenommen kanonischer Deep Research Agent)."""
    clean_id = (model_id or "").strip().lower()
    name_low = (display_name or "").strip().lower()
    if clean_id == _CANONICAL_DEEP_RESEARCH_ID and not any(
        bad in name_low for bad in ("rerouted", "deprecated")
    ):
        return False
    low = f"{clean_id} {name_low}"
    return any(tok in low for tok in _PREVIEW_MARKERS) or clean_id.endswith("-exp")


def _rank_gemini_candidate(spec: AIModelSpec) -> tuple[int, int, int, int, int, str]:
    low = spec.model_id.lower()
    m = _VERSION_RE.search(low)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if (m and m.group(2)) else (5 if "hydrogen" in low else 0)
    is_ga = 0 if is_preview_or_experimental_model(spec.model_id, spec.name) else 1
    is_alias = 0 if _DATED_CLONE_RE.search(low) else 1
    is_gemini_family = 1 if low.startswith("gemini-") else 0
    return (is_ga, is_gemini_family, major, minor, is_alias, spec.id)


@dataclass(frozen=True)
class DiscoverySummary:
    """Ergebnis eines automatischen Katalog-Syncs."""

    added_models: list[str] = field(default_factory=list)
    added_tools: list[str] = field(default_factory=list)
    recovered_models: list[str] = field(default_factory=list)
    rechecked_models: int = 0
    checked_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_excluded: list[str] = field(default_factory=list)
    skipped_unhealthy: list[str] = field(default_factory=list)
    skipped_previews: list[str] = field(default_factory=list)
    existing_previews_count: int = 0


def _guess_gemini_tier(model_id: str) -> tuple[BrainTier, bool]:
    low = model_id.lower()
    if "deep-research" in low:
        return BrainTier.DEEP, True
    if "lite" in low:
        return BrainTier.FAST, False
    if any(tok in low for tok in ("hydrogen", "argon", "carbon", "pro", "ultra")):
        return BrainTier.STANDARD if "hydrogen" in low else BrainTier.DEEP, False
    return BrainTier.STANDARD, False


def _sync_new_builtins(
    repo: BrainRepository,
    secrets: BrainSecrets,
    excluded_models: set[str],
    excluded_tools: set[str],
    *,
    retry_model_ids: set[str] | None = None,
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
    retry_set = retry_model_ids or set()

    added_m: list[str] = []
    skipped: list[str] = []
    for m in build_default_models():
        if m.id in excluded_models:
            if m.id not in retry_set:
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
            to_save = t
            if t.requires_api_key and t.api_key_env and not secrets.has_tool_key(t, allow_paid=False):
                to_save = t.model_copy(
                    update={"last_status": ProbeStatus.AUTH_MISSING, "last_error": f"Key {t.api_key_env} fehlt."}
                )
            repo.upsert_tool(to_save)
            added_t.append(t.id)
        known_tools.add(t.id)

    repo.set_meta(
        "known_builtin_ids",
        json.dumps({"models": sorted(known_models), "tools": sorted(known_tools)}),
    )
    return added_m, added_t, skipped


def _discover_gemini_models(client: genai.Client, *, exclude_previews: bool = True) -> tuple[list[AIModelSpec], list[str]]:
    discovered: list[AIModelSpec] = []
    skipped_previews: list[str] = []
    for raw in client.models.list():
        raw_name = str(getattr(raw, "name", "") or "").strip()
        model_id = raw_name.removeprefix("models/").strip()
        low = model_id.lower()
        if not low or any(skip in low for skip in _GEMINI_SKIP_PARTS):
            continue
        if not _GEMINI_TEXT_SERIES_RE.match(low):
            continue
        actions = [str(a) for a in (getattr(raw, "supported_actions", None) or [])]
        if actions and not any(act in actions for act in ("generateContent", "streamGenerateContent")) and "deep-research" not in low:
            continue
        display = str(getattr(raw, "display_name", "") or model_id).strip()[:120]
        if exclude_previews and is_preview_or_experimental_model(model_id, display):
            skipped_previews.append(model_id)
            continue
        tier, is_deep = _guess_gemini_tier(model_id)
        prov = ModelProvider.GEMINI_DEEP_AGENT if is_deep else ModelProvider.GOOGLE_GEMINI
        is_free_flag = infer_model_is_free(model_id, display, provider=prov)
        discovered.append(
            AIModelSpec(
                id=validate_identifier(model_id, "Modell-ID"),
                name=display,
                provider=prov,
                model_id=model_id,
                tier_hint=tier,
                is_free=is_free_flag,
                free_quota_period="unlimited" if is_free_flag else "none",
                paid_api_key_env="GEMINI_PAID_API_KEY",
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
    discovered.sort(key=_rank_gemini_candidate, reverse=True)
    return discovered[:32], skipped_previews


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
                paid_api_key_env="GROQ_PAID_API_KEY",
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


def _parse_openrouter_per_1m(raw_val: object) -> float:
    try:
        return round(max(0.0, float(raw_val or 0.0) * 1_000_000.0), 4)
    except (TypeError, ValueError):
        return 0.0


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
                paid_api_key_env="OPENROUTER_PAID_API_KEY",
                base_url="https://openrouter.ai/api/v1",
                supports_google_search=False,
                supports_url_context=False,
                max_rpm=15,
                source="auto_discovered",
                origin_apps=["openrouter-discovery"],
                notes="Automatisch erkanntes `:free`-Modell über OpenRouter.",
                input_cost_per_1m=_parse_openrouter_per_1m(pricing.get("prompt")),
                output_cost_per_1m=_parse_openrouter_per_1m(pricing.get("completion")),
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
    on_progress: ProgressCallback | None = None,
) -> list[tuple[AIModelSpec, ProbeStatus]]:
    if not candidates:
        return []
    openai_items = [m for m in candidates if m.provider == ModelProvider.OPENAI_COMPATIBLE]
    gemini_items = [m for m in candidates if m.provider != ModelProvider.OPENAI_COMPATIBLE]
    results: list[tuple[AIModelSpec, ProbeStatus]] = []
    total = len(candidates)
    done = 0
    if openai_items:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for probed, entry in pool.map(bench.probe_model, openai_items):
                done += 1
                if on_progress:
                    on_progress(0.55 + 0.40 * (done / total), f"Prüfe neues Modell ({done}/{total}): `{probed.id}`...")
                results.append((probed, entry.status))
    for spec in gemini_items:
        done += 1
        if on_progress:
            on_progress(0.55 + 0.40 * (done / total), f"Prüfe neues Gemini-Modell ({done}/{total}): `{spec.id}`...")
        probed, entry = bench.probe_model(spec)
        results.append((probed, entry.status))
    return results


def _recheck_unhealthy_existing(
    repo: BrainRepository,
    secrets: BrainSecrets,
    bench: BrainBenchmarkRunner,
    *,
    exclude_previews: bool,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[str], int, bool]:
    targets = [
        m
        for m in repo.list_models(enabled_only=True)
        if m.last_status in {ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR}
        and secrets.has_model_key(m, allow_paid=not m.is_free)
        and not (exclude_previews and is_preview_or_experimental_model(m.model_id, m.name))
    ]
    recovered: list[str] = []
    quota_recovered = False
    for idx, model in enumerate(targets, start=1):
        if on_progress:
            on_progress(
                0.08 + 0.12 * (idx / len(targets)),
                f"Prüfe bisher limitiertes Modell ({idx}/{len(targets)}): `{model.id}`...",
            )
        was_quota = model.last_status == ProbeStatus.QUOTA_LIMITED
        probed, entry = bench.probe_model(model)
        repo.upsert_model(probed)
        if entry.status == ProbeStatus.OK:
            recovered.append(probed.id)
            if was_quota:
                quota_recovered = True

    ai_tools = [
        t
        for t in repo.list_tools(enabled_only=True)
        if t.category == ToolCategory.AI_GROUNDING
        and t.last_status in {ProbeStatus.QUOTA_LIMITED, ProbeStatus.ERROR}
        and secrets.has_tool_key(t, allow_paid=False)
    ]
    if ai_tools:
        healthy_models = repo.list_models(enabled_only=True)
        for tool in ai_tools:
            grounding_model = select_grounding_model(
                healthy_models,
                secrets,
                executor_type=tool.executor_type,
                allow_paid=False,
            )
            if grounding_model is None:
                continue
            was_quota = tool.last_status == ProbeStatus.QUOTA_LIMITED
            probed_t, entry_t = bench.probe_tool(tool, grounding_model=grounding_model)
            repo.upsert_tool(probed_t)
            if was_quota and entry_t.status == ProbeStatus.OK:
                quota_recovered = True
    return recovered, len(targets), quota_recovered


def _resolve_retry_model_ids(
    excluded_models: set[str],
    excluded_reasons: dict[str, str],
    *,
    retry_excluded: bool,
    quota_recovered: bool,
) -> set[str]:
    if retry_excluded:
        return set(excluded_models)
    if quota_recovered:
        return {
            mid
            for mid in excluded_models
            if excluded_reasons.get(mid, "legacy") in {"quota_limited", "legacy"}
        }
    return set()


def _persist_discovered_candidates(
    repo: BrainRepository,
    candidates: list[AIModelSpec],
    bench: BrainBenchmarkRunner,
    *,
    verify_discovered: bool,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[str], list[str]]:
    if not candidates:
        return [], []
    added: list[str] = []
    unhealthy: list[str] = []
    unhealthy_reasons: dict[str, str] = {}
    if not verify_discovered:
        for spec in candidates:
            repo.upsert_model(spec)
            repo.remove_excluded_id(model_id=spec.id)
            added.append(spec.id)
        return added, unhealthy

    for probed, status in _probe_candidate_models(candidates, bench, on_progress=on_progress):
        if status in {ProbeStatus.ERROR, ProbeStatus.QUOTA_LIMITED}:
            unhealthy.append(probed.id)
            unhealthy_reasons[probed.id] = (
                "quota_limited" if status == ProbeStatus.QUOTA_LIMITED else "error"
            )
            continue
        repo.upsert_model(probed)
        repo.remove_excluded_id(model_id=probed.id)
        added.append(probed.id)
    if unhealthy:
        repo.add_excluded_ids(model_ids=unhealthy, model_reasons=unhealthy_reasons)
    return added, unhealthy


def auto_sync_catalog(
    repo: BrainRepository,
    secrets: BrainSecrets,
    *,
    genai_client: genai.Client | None = None,
    manifest_url: str = "",
    runner: UnifiedLLMRunner | None = None,
    verify_discovered: bool = True,
    exclude_previews: bool = True,
    retry_excluded: bool = False,
    on_progress: ProgressCallback | None = None,
) -> DiscoverySummary:
    """Erkennt neue/reaktivierte Modelle & Tools, prüft Quota-Recovery und schließt defekte/Preview-IDs aus."""
    from ki_brain.tools.searxng import build_searxng_tool, discover_verified_searxng_pool

    probe_runner = runner or UnifiedLLMRunner(secrets, genai_client=genai_client)
    bench = BrainBenchmarkRunner(repo, secrets, probe_runner)

    recovered_models: list[str] = []
    rechecked_models = 0
    quota_recovered = False
    if verify_discovered:
        recovered_models, rechecked_models, quota_recovered = _recheck_unhealthy_existing(
            repo, secrets, bench, exclude_previews=exclude_previews, on_progress=on_progress
        )

    if on_progress:
        on_progress(0.22, "Gleiche Standardkatalog-Erweiterungen (Builtins) ab...")
    excluded_models, excluded_tools = repo.list_excluded_ids()
    excluded_m_reasons, _ = repo.list_excluded_reasons()
    retry_ids = _resolve_retry_model_ids(
        excluded_models,
        excluded_m_reasons,
        retry_excluded=retry_excluded,
        quota_recovered=quota_recovered,
    )
    added_models, added_tools, skipped_excluded = _sync_new_builtins(
        repo, secrets, excluded_models, excluded_tools, retry_model_ids=retry_ids
    )
    sources = ["builtins"]
    warnings: list[str] = []
    skipped_previews: list[str] = []

    if "searxng_public" not in excluded_tools:
        sources.append("searx.space")
        existing_searx = repo.get_tool("searxng_public")
        if existing_searx is None or (existing_searx.last_status != ProbeStatus.OK and existing_searx.source != "custom"):
            if on_progress:
                on_progress(0.28, "Prüfe öffentliche SearXNG-Instanzen (searx.space + Fallback-Pool)...")
            try:
                verified_pool = discover_verified_searxng_pool(max_verify=10, max_keep=4, http_request_fn=safe_http_request)
                if verified_pool:
                    searx_spec = build_searxng_tool(verified_pool).model_copy(
                        update={"last_status": ProbeStatus.OK, "quality_score": 82.0, "reliability_score": 100.0}
                    )
                    repo.upsert_tool(searx_spec)
                    if existing_searx is None:
                        added_tools.append("searxng_public")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"searx.space Discovery: {exc}")

    existing_ids = {m.id for m in repo.list_models()}
    candidates: list[AIModelSpec] = []
    seen_candidates: set[str] = set()

    def _collect(items: list[AIModelSpec], limit: int = 12) -> None:
        collected_here = 0
        for spec in items:
            if spec.id in existing_ids or spec.id in seen_candidates:
                continue
            seen_candidates.add(spec.id)
            if exclude_previews and is_preview_or_experimental_model(spec.model_id, spec.name):
                skipped_previews.append(spec.id)
                continue
            if spec.id in excluded_models and spec.id not in retry_ids:
                skipped_excluded.append(spec.id)
                continue
            candidates.append(spec)
            collected_here += 1
            if limit > 0 and collected_here >= limit:
                break

    gem_key = secrets.resolve_any_gemini_key()
    if genai_client is not None or gem_key:
        sources.append("gemini-models")
        if on_progress:
            on_progress(0.36, "Frage aktuelle Gemini-Modelle über `models.list()` ab...")
        try:
            client = genai_client or genai.Client(api_key=gem_key)
            gem_found, gem_previews = _discover_gemini_models(client, exclude_previews=exclude_previews)
            for pid in gem_previews:
                if pid not in existing_ids and pid not in skipped_previews:
                    skipped_previews.append(pid)
            _collect(gem_found, limit=20)
        except Exception as exc:
            warnings.append(f"Gemini Auto-Discovery: {exc}")

    if retry_ids:
        retry_builtins = [
            m
            for m in build_default_models()
            if m.id in retry_ids
            and m.id not in existing_ids
            and (
                not verify_discovered
                or genai_client is not None
                or secrets.has_model_key(m, allow_paid=not m.is_free)
            )
        ]
        _collect(retry_builtins, limit=0)

    groq_key = secrets.resolve_key("GROQ_API_KEY") or secrets.resolve_key("GROQ_PAID_API_KEY")
    if groq_key:
        sources.append("groq-models")
        if on_progress:
            on_progress(0.44, "Frage kostenfreie Groq-Modelle ab...")
        try:
            _collect(_discover_groq_models(groq_key), limit=8)
        except Exception as exc:
            warnings.append(f"Groq Auto-Discovery: {exc}")

    or_key = secrets.resolve_key("OPENROUTER_API_KEY") or secrets.resolve_key("OPENROUTER_PAID_API_KEY")
    if or_key:
        sources.append("openrouter-free")
        if on_progress:
            on_progress(0.50, "Frage OpenRouter `:free`-Modelle ab...")
        try:
            _collect(_discover_openrouter_free_models(or_key), limit=8)
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

    cand_added, skipped_unhealthy = _persist_discovered_candidates(
        repo, candidates, bench, verify_discovered=verify_discovered, on_progress=on_progress
    )
    added_models.extend(cand_added)

    if added_models or recovered_models:
        seqs = optimize_tier_sequences(
            repo.list_models(),
            repo.list_tools(),
            repo.list_sequences(),
            force_overwrite=False,
            usage_state=repo.load_usage_quotas(),
        )
        for seq in seqs.values():
            repo.upsert_sequence(seq)

    existing_previews_count = sum(
        1 for m in repo.list_models() if is_preview_or_experimental_model(m.model_id, m.name)
    )
    if on_progress:
        on_progress(1.0, f"Discovery fertig: +{len(added_models)} Modelle, +{len(added_tools)} Tools.")

    return DiscoverySummary(
        added_models=added_models,
        added_tools=added_tools,
        recovered_models=recovered_models,
        rechecked_models=rechecked_models,
        checked_sources=sources,
        warnings=warnings,
        skipped_excluded=skipped_excluded,
        skipped_unhealthy=skipped_unhealthy,
        skipped_previews=skipped_previews,
        existing_previews_count=existing_previews_count,
    )
