"""Einheitlicher KI-Ausführungs-Runner für Gemini Interactions API, Deep Agent & OpenAI-Compat."""

from __future__ import annotations

import json
import re
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
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_QUOTA_MARKERS = ("429", "quota", "resource_exhausted", "too_many_requests", "ratelimit")
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


def extract_citations_from_interaction(interaction: Any) -> list[BrainCitation]:
    """Extrahiert `url_citation`-Annotationen aus einer Gemini Interaction."""
    citations: list[BrainCitation] = []
    seen: set[str] = set()
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for block in getattr(step, "content", None) or []:
            if getattr(block, "type", None) != "text":
                continue
            for ann in getattr(block, "annotations", None) or []:
                if getattr(ann, "type", None) != "url_citation":
                    continue
                safe = sanitize_display_url(str(getattr(ann, "url", "") or ""))
                if safe == "#" or safe in seen:
                    continue
                seen.add(safe)
                citations.append(
                    BrainCitation(
                        url=safe,
                        title=str(getattr(ann, "title", "") or "").strip()[:180],
                        source_tool="gemini_google_search",
                    )
                )
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

    def _get_genai_client(self, timeout_seconds: float = 40.0) -> genai.Client:
        if self._genai_client is not None:
            return self._genai_client
        key = self._secrets.resolve_key("GEMINI_API_KEY")
        if not key:
            raise BrainProviderError("Kein gültiger GEMINI_API_KEY konfiguriert.")
        http_opts = {"timeout": int(timeout_seconds * 1000), "retry_options": {"attempts": 1}}
        try:
            self._genai_client = genai.Client(api_key=key, http_options=http_opts)
        except TypeError:
            self._genai_client = genai.Client(api_key=key)
        return self._genai_client

    def run_text(
        self,
        model: AIModelSpec,
        prompt: str,
        *,
        system_instruction: str = "",
        timeout_seconds: float = 40.0,
        tools: list[dict[str, str]] | None = None,
        on_poll_tick: Callable[[], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> tuple[str, list[BrainCitation]]:
        """Führt eine Text- oder Agenten-Anfrage gegen genau ein Modell aus."""
        self._rate_limiter.acquire(cancel_check)
        if model.provider == ModelProvider.OPENAI_COMPATIBLE:
            text = self._call_openai_compat(model, prompt, system_instruction, timeout_seconds=timeout_seconds)
            return text, []
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
        interaction = client.interactions.create(**kwargs)
        text = extract_interaction_text(interaction)
        if not text:
            raise BrainProviderError(f"Leere Antwort von Modell '{model.model_id}'.")
        return text, extract_citations_from_interaction(interaction)

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
        self._rate_limiter.acquire()
        if model.provider == ModelProvider.OPENAI_COMPATIBLE:
            schema_hint = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
            sys_txt = f"{system_instruction}\nGib ausschließlich valides JSON gemäß Schema zurück:\n{schema_hint}".strip()
            raw = self._call_openai_compat(model, prompt, sys_txt, timeout_seconds=timeout_seconds, json_mode=True)
            return schema_cls.model_validate_json(strip_json_fences(raw))

        client = self._get_genai_client(timeout_seconds=timeout_seconds)
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
        raw_json = strip_json_fences(extract_interaction_text(interaction))
        if not raw_json:
            raise BrainProviderError(f"Leere JSON-Antwort von '{model.model_id}'.")
        return schema_cls.model_validate_json(raw_json)

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
