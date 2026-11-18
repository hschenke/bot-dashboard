"""Einheitlicher KI-Ausführungs-Runner für Gemini Interactions API, Deep Agent & OpenAI-Compat."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from google import genai
from pydantic import BaseModel

from ki_brain.enums import ModelProvider
from ki_brain.exceptions import BrainProviderError
from ki_brain.models import AIModelSpec, BrainCitation, BrainSecrets
from ki_brain.rate_limiter import LlmRateLimiter, get_global_rate_limiter
from ki_brain.security import sanitize_display_url
from ki_brain.tools.http_utils import API_HEADERS, safe_http_request

ModelT = TypeVar("ModelT", bound=BaseModel)
_GROUNDING_LOCAL = threading.local()
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_QUOTA_MARKERS = ("429", "quota", "resource_exhausted", "too_many_requests", "ratelimit")
_PERMANENT_ERROR_MARKERS = (
    "400",
    "401",
    "403",
    "404",
    "invalid_argument",
    "permission_denied",
    "not_found",
    "api_key",
    "unauthenticated",
    "billing",
)
_TRANSIENT_ERROR_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "overloaded",
    "internal",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "eof",
    "reset by peer",
    "leere antwort",
    "leere json-antwort",
)
_OPENAI_BASE_PRESETS = {
    "GROQ_API_KEY": "https://api.groq.com/openai/v1",
    "OPENROUTER_API_KEY": "https://openrouter.ai/api/v1",
    "CEREBRAS_API_KEY": "https://api.cerebras.ai/v1",
    "MISTRAL_API_KEY": "https://api.mistral.ai/v1",
}


def is_quota_error(exc: Exception) -> bool:
    """Erkennt 429- oder Free-Tier-Kontingentfehler."""
    msg = str(exc).lower()
    return any(tok in msg for tok in _QUOTA_MARKERS)


def is_transient_llm_error(exc: Exception) -> bool:
    """Erkennt vorübergehende 5xx-/503-/Netzwerk-Spikes (ohne 429/400/404 zu wiederholen)."""
    if is_quota_error(exc):
        return False
    msg = str(exc).lower()
    if any(tok in msg for tok in _PERMANENT_ERROR_MARKERS):
        return False
    return any(tok in msg for tok in _TRANSIENT_ERROR_MARKERS)


def _describe_failed_interaction(interaction: Any) -> str:
    status = getattr(interaction, "status", None)
    errors = getattr(interaction, "errors", None)
    if isinstance(status, str) and status.lower() in {"failed", "cancelled", "incomplete"}:
        err_txt = str(errors)[:160] if isinstance(errors, (list, tuple, str)) and errors else ""
        return f"Interaction-Status '{status}'" + (f": {err_txt}" if err_txt else "")
    return ""


def strip_json_fences(raw_text: str) -> str:
    return _JSON_FENCE_RE.sub("", (raw_text or "").strip()).strip()


def extract_interaction_text(interaction: Any) -> str:
    """Liest `output_text` oder verkettet Text-Blöcke aus Gemini `model_output`-Steps."""
    direct = str(getattr(interaction, "output_text", "") or "").strip()
    if direct:
        return direct
    parts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for block in getattr(step, "content", None) or []:
            if getattr(block, "type", None) == "text":
                txt = str(getattr(block, "text", "") or "").strip()
                if txt:
                    parts.append(txt)
    return "\n\n".join(parts).strip()


def pop_last_grounding_counts() -> dict[str, int]:
    """Liest und leert die im aktuellen Thread zuletzt gemeldeten Grounding-Query-Zähler."""
    counts = getattr(_GROUNDING_LOCAL, "counts", None)
    _GROUNDING_LOCAL.counts = {}
    return dict(counts) if isinstance(counts, dict) else {}


def extract_grounding_tool_counts(interaction: Any) -> dict[str, int]:
    """Liest abgerechnete Search-/Maps-Grounding-Anfragen aus `interaction.usage.grounding_tool_count`."""
    usage = getattr(interaction, "usage", None)
    raw_counts = getattr(usage, "grounding_tool_count", None) if usage is not None else None
    if not isinstance(raw_counts, (list, tuple)):
        return {}
    totals: dict[str, int] = {}
    for entry in raw_counts:
        g_type = getattr(entry, "type", None)
        g_cnt = getattr(entry, "count", None)
        if isinstance(g_type, str) and isinstance(g_cnt, int) and not isinstance(g_cnt, bool) and g_cnt > 0:
            clean = g_type.strip().lower()
            totals[clean] = totals.get(clean, 0) + g_cnt
    return totals


def _build_place_citation(obj: Any) -> BrainCitation | None:
    raw_url = str(getattr(obj, "url", "") or "").strip()
    raw_pid = str(getattr(obj, "place_id", "") or "").strip()
    if not raw_url and raw_pid:
        clean_pid = raw_pid.removeprefix("places/").strip()
        if clean_pid:
            raw_url = f"https://www.google.com/maps/place/?q=place_id:{clean_pid}"
    safe = sanitize_display_url(raw_url)
    if safe == "#":
        return None
    title = str(getattr(obj, "name", "") or getattr(obj, "title", "") or "").strip()[:180]
    snippets_raw = getattr(obj, "review_snippets", None)
    snippets: list[str] = []
    if isinstance(snippets_raw, (list, tuple)):
        for rs in snippets_raw[:3]:
            r_title = str(getattr(rs, "title", "") or "").strip()
            if r_title:
                snippets.append(r_title)
    return BrainCitation(
        url=safe,
        title=title or "Google Maps Place",
        snippet=" · ".join(snippets)[:320],
        source_tool="gemini_google_maps",
    )


def extract_citations_from_interaction(interaction: Any) -> list[BrainCitation]:
    """Extrahiert Web- (`url_citation`) und Maps-Zitate (`place_citation`, `google_maps_result`)."""
    citations: list[BrainCitation] = []
    seen: set[str] = set()
    steps = getattr(interaction, "steps", None)
    if not isinstance(steps, (list, tuple)):
        return citations

    def _add(cite: BrainCitation | None) -> None:
        if cite is not None and cite.url not in seen:
            seen.add(cite.url)
            citations.append(cite)

    for step in steps:
        stype = getattr(step, "type", None)
        if stype == "google_maps_result":
            results = getattr(step, "result", None)
            if isinstance(results, (list, tuple)):
                for res in results:
                    places = getattr(res, "places", None)
                    if isinstance(places, (list, tuple)):
                        for pl in places:
                            _add(_build_place_citation(pl))
            continue
        if stype != "model_output":
            continue
        content = getattr(step, "content", None)
        if not isinstance(content, (list, tuple)):
            continue
        for block in content:
            if getattr(block, "type", None) != "text":
                continue
            anns = getattr(block, "annotations", None)
            if not isinstance(anns, (list, tuple)):
                continue
            for ann in anns:
                atype = getattr(ann, "type", None)
                if atype == "url_citation":
                    safe = sanitize_display_url(str(getattr(ann, "url", "") or ""))
                    if safe != "#":
                        _add(BrainCitation(url=safe, title=str(getattr(ann, "title", "") or "").strip()[:180], source_tool="gemini_google_search"))
                elif atype == "place_citation":
                    _add(_build_place_citation(ann))
    return citations


class UnifiedLLMRunner:
    """Führt Text-, Struktur- und Agenten-Anfragen über beliebige Brain-Modelle aus."""

    def __init__(
        self,
        secrets: BrainSecrets,
        *,
        genai_client: genai.Client | None = None,
        rate_limiter: LlmRateLimiter | None = None,
    ) -> None:
        self._secrets = secrets
        self._genai_client = genai_client
        self._rate_limiter = rate_limiter or get_global_rate_limiter()

    def _get_genai_client(self, timeout_seconds: float = 60.0) -> genai.Client:
        if self._genai_client is not None:
            return self._genai_client
        key = self._secrets.resolve_key("GEMINI_API_KEY")
        if not key:
            raise BrainProviderError("Kein gültiger GEMINI_API_KEY konfiguriert.")
        http_opts = {"timeout": max(60_000, int(timeout_seconds * 1000)), "retry_options": {"attempts": 2}}
        try:
            self._genai_client = genai.Client(api_key=key, http_options=http_opts)
        except TypeError:
            self._genai_client = genai.Client(api_key=key)
        return self._genai_client

    def _run_with_transient_retry(
        self,
        fn: Callable[[], Any],
        *,
        cancel_check: Callable[[], None] | None = None,
        max_attempts: int = 3,
    ) -> Any:
        in_pytest = "PYTEST_CURRENT_TEST" in os.environ
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            if attempt > 0:
                delay = 0.0 if in_pytest else min(2.4, 0.8 * attempt)
                if delay > 0:
                    time.sleep(delay)
                self._rate_limiter.acquire(cancel_check)
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if is_quota_error(exc):
                    self._rate_limiter.notify_rate_limit(5.0)
                    raise
                if not is_transient_llm_error(exc) or attempt + 1 >= max_attempts:
                    raise
        if last_exc is not None:
            raise last_exc
        raise BrainProviderError("Unbekannter LLM-Fehler.")

    def run_text(
        self,
        model: AIModelSpec,
        prompt: str,
        *,
        system_instruction: str = "",
        timeout_seconds: float = 40.0,
        tools: list[dict[str, Any]] | None = None,
        on_poll_tick: Callable[[], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> tuple[str, list[BrainCitation]]:
        """Führt eine Text- oder Agenten-Anfrage gegen genau ein Modell aus."""
        _GROUNDING_LOCAL.counts = {}
        if model.provider == ModelProvider.OPENAI_COMPATIBLE:
            if cancel_check is not None:
                cancel_check()
            text = self._call_openai_compat(model, prompt, system_instruction, timeout_seconds=timeout_seconds)
            return text, []
        self._rate_limiter.acquire(cancel_check)
        if model.provider == ModelProvider.GEMINI_DEEP_AGENT or model.is_deep_agent:
            return self._call_gemini_deep_agent(
                model, prompt, timeout_seconds=max(timeout_seconds, 120.0), on_poll_tick=on_poll_tick, cancel_check=cancel_check
            )
        client = self._get_genai_client(timeout_seconds=timeout_seconds)
        kwargs: dict[str, Any] = {"model": model.model_id, "input": prompt, "timeout": timeout_seconds}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if tools:
            kwargs["tools"] = tools

        def _invoke() -> tuple[str, list[BrainCitation]]:
            interaction = client.interactions.create(**kwargs)
            fail_reason = _describe_failed_interaction(interaction)
            if fail_reason:
                raise BrainProviderError(f"Modell '{model.model_id}': {fail_reason}")
            text = extract_interaction_text(interaction)
            if not text:
                raise BrainProviderError(f"Leere Antwort von Modell '{model.model_id}'.")
            _GROUNDING_LOCAL.counts = extract_grounding_tool_counts(interaction)
            return text, extract_citations_from_interaction(interaction)

        return self._run_with_transient_retry(_invoke, cancel_check=cancel_check)

    def run_structured(
        self,
        model: AIModelSpec,
        prompt: str,
        schema_cls: type[ModelT],
        *,
        system_instruction: str = "",
        timeout_seconds: float = 40.0,
    ) -> ModelT:
        """Erzwingt gültiges Pydantic-JSON vom gewählten Modell."""
        if model.provider == ModelProvider.OPENAI_COMPATIBLE:
            schema_hint = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
            sys_txt = f"{system_instruction}\nGib ausschließlich valides JSON gemäß Schema zurück:\n{schema_hint}".strip()
            raw = self._call_openai_compat(model, prompt, sys_txt, timeout_seconds=timeout_seconds, json_mode=True)
            return schema_cls.model_validate_json(strip_json_fences(raw))
        self._rate_limiter.acquire()

        client = self._get_genai_client(timeout_seconds=timeout_seconds)

        def _invoke() -> ModelT:
            interaction = client.interactions.create(
                model=model.model_id,
                input=prompt,
                system_instruction=system_instruction or "Liefere präzises JSON entsprechend dem Schema.",
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema_cls.model_json_schema(),
                },
                timeout=timeout_seconds,
            )
            fail_reason = _describe_failed_interaction(interaction)
            if fail_reason:
                raise BrainProviderError(f"Modell '{model.model_id}': {fail_reason}")
            raw_json = strip_json_fences(extract_interaction_text(interaction))
            if not raw_json:
                raise BrainProviderError(f"Leere JSON-Antwort von '{model.model_id}'.")
            return schema_cls.model_validate_json(raw_json)

        return self._run_with_transient_retry(_invoke)

    def _call_gemini_deep_agent(
        self,
        model: AIModelSpec,
        prompt: str,
        *,
        timeout_seconds: float = 600.0,
        on_poll_tick: Callable[[], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> tuple[str, list[BrainCitation]]:
        client = self._get_genai_client(timeout_seconds=40.0)
        started = client.interactions.create(agent=model.model_id, input=prompt, background=True, timeout=30.0)
        run_id = str(getattr(started, "id", "") or "")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if on_poll_tick:
                on_poll_tick()
            if cancel_check:
                cancel_check()
            current = client.interactions.get(run_id, timeout=20.0)
            status = str(getattr(current, "status", "") or "").lower()
            if status == "completed":
                text = extract_interaction_text(current)
                if not text:
                    raise BrainProviderError(f"Deep Agent '{model.model_id}' lieferte keinen Text.")
                return text, extract_citations_from_interaction(current)
            if status in {"failed", "cancelled", "incomplete"}:
                raise BrainProviderError(f"Deep Agent '{model.model_id}' Status: {status}")
            time.sleep(3.0)
        raise BrainProviderError(f"Deep Agent '{model.model_id}' Zeitlimit überschritten.")

    def _call_openai_compat(
        self,
        model: AIModelSpec,
        prompt: str,
        system_instruction: str,
        *,
        timeout_seconds: float = 35.0,
        json_mode: bool = False,
    ) -> str:
        base = (model.base_url or _OPENAI_BASE_PRESETS.get(model.api_key_env.upper(), "")).rstrip("/")
        if not base:
            raise BrainProviderError(f"Keine base_url für OpenAI-Compat-Modell '{model.id}' definiert.")
        headers = {**API_HEADERS, "Content-Type": "application/json"}
        if model.requires_api_key:
            key = self._secrets.resolve_key(model.api_key_env)
            if not key:
                raise BrainProviderError(f"API-Key '{model.api_key_env}' fehlt.")
            headers["Authorization"] = f"Bearer {key}"
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": model.model_id, "messages": messages, "temperature": 0.2}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        status, _, _, raw = safe_http_request(
            "POST", f"{base}/chat/completions", json_body=body, headers=headers, timeout_seconds=timeout_seconds, max_bytes=1_000_000
        )
        if status == 429:
            raise BrainProviderError("HTTP 429 Rate-Limit bei OpenAI-Compat-Endpunkt.")
        if status != 200:
            raise BrainProviderError(f"OpenAI-Compat HTTP {status} für '{model.model_id}'.")
        payload = json.loads(raw.decode("utf-8", errors="replace"))
        choices = payload.get("choices") or []
        content = str(choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        if not content:
            raise BrainProviderError(f"Leere Antwort von '{model.model_id}'.")
        return content
