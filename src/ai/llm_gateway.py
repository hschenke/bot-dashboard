"""Google Gemini Interactions API Client (`google-genai` >= 2.30.1) inkl. Heuristik-Fallback."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from typing import Any

from google import genai

from src.config.constants import (
    CANDIDATE_MODELS,
    LLM_MAX_ATTEMPTS,
    LLM_TIMEOUT_SECONDS,
    RETRY_BACKOFF_BASE_SECONDS,
)
from src.config.settings import AppSettings
from src.domain.enums import HealthStatus, WorkflowRunState
from src.domain.exceptions import LLMProviderError
from src.domain.models import AppFleetSnapshot
from src.domain.schemas import AppDiagnosticInsight, FleetAIAnalysis
from src.security.sanitization import validate_user_question

logger = logging.getLogger(__name__)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_interaction_text(interaction: Any) -> str:
    """Liest `output_text` oder verkettet Text-Blöcke aus `model_output`-Steps."""
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


def format_snapshots_telemetry_context(snapshots: Sequence[AppFleetSnapshot]) -> str:
    """Formatiert die aktuellen Prüfdaten aller 3 Bots als kompakten KI-Kontext."""
    lines: list[str] = ["Aktuelle Live-Telemetrie der 3 überwachten Streamlit-Bots:"]
    for snap in snapshots:
        tables_str = ", ".join(
            f"{m.table_name}={m.row_count}" for m in snap.db_check.table_metrics
        )
        recent_str = "; ".join(
            f"[{item.category}] {item.title} ({item.status})"
            for item in snap.db_check.recent_items[:3]
        )
        lines.append(
            f"\n--- {snap.spec.icon} {snap.spec.title} ({snap.spec.app_id.value}) ---\n"
            f"- Gesamtstatus: {snap.overall_status.value}\n"
            f"- Streamlit-URL ({snap.spec.streamlit_url}): {snap.url_check.status.value} | "
            f"{snap.url_check.message} | Health-Endpoint OK: {snap.url_check.health_endpoint_ok}\n"
            f"- GitHub Actions ({snap.workflow_check.workflow_file}): "
            f"{snap.workflow_check.state.value} ({snap.workflow_check.status_label}) | "
            f"Letzter Lauf: {snap.workflow_check.updated_at or 'unbekannt'}\n"
            f"- Datenbank ({snap.db_check.backend_label}): {snap.db_check.status.value} | "
            f"Gesamtzeilen: {snap.db_check.total_rows} ({tables_str or 'keine Tabellen'}) | "
            f"Aktive Elemente: {snap.db_check.active_count} | Fehlerzähler: {snap.db_check.error_count}\n"
            f"- Jüngste DB-Einträge: {recent_str or 'keine'}"
        )
    return "\n".join(lines)


def build_heuristic_fleet_analysis(snapshots: Sequence[AppFleetSnapshot]) -> FleetAIAnalysis:
    """Erstellt eine deterministische Systemdiagnose, falls kein Gemini API-Key aktiv ist."""
    if not snapshots:
        return FleetAIAnalysis(
            fleet_health_score=0,
            headline="Noch keine Telemetriedaten erfasst",
            executive_summary="Bitte starte zunächst einen System-Check, um Daten für alle Apps zu erheben.",
            model_used="heuristic-fallback",
        )

    score = 100
    insights: list[AppDiagnosticInsight] = []
    warnings: list[str] = []
    next_steps: list[str] = []

    for snap in snapshots:
        rating = "optimal"
        action = "Keine Aktion erforderlich – System läuft regulär."

        if snap.url_check.status == HealthStatus.OFFLINE:
            score -= 20
            rating = "kritisch"
            warnings.append(f"{snap.spec.title}: Web-App-URL ist aktuell nicht erreichbar.")
            action = f"Streamlit Cloud Deployment unter {snap.spec.streamlit_url} prüfen."
        elif snap.url_check.status == HealthStatus.ASLEEP:
            score -= 10
            rating = "hinweis"
            warnings.append(f"{snap.spec.title}: Streamlit-Container befindet sich im Ruhezustand.")
            action = f"App unter {snap.spec.streamlit_url} einmalig im Browser aufwecken."

        if snap.workflow_check.state == WorkflowRunState.FAILURE:
            score -= 15
            rating = "kritisch" if rating != "kritisch" else rating
            warnings.append(
                f"{snap.spec.title}: Letzter GitHub-Workflow ({snap.workflow_check.workflow_file}) fehlgeschlagen."
            )
            action = "GitHub Actions Protokoll prüfen und Lauf ggf. neu anstoßen."

        if snap.db_check.status == HealthStatus.OFFLINE:
            score -= 15
            rating = "hinweis" if rating == "optimal" else rating
            warnings.append(f"{snap.spec.title}: Datenbank nicht verbunden ({snap.db_check.message}).")
            action = f"Turso-Key ({snap.spec.app_id.value.upper()}_TURSO_KEY) in den Secrets hinterlegen."
        elif snap.db_check.error_count > 0:
            score -= 8
            rating = "hinweis" if rating == "optimal" else rating
            warnings.append(
                f"{snap.spec.title}: {snap.db_check.error_count} fehlerhafte Jobs/Regeln in der Datenbank."
            )

        obs = (
            f"URL: {snap.url_check.status.value} ({snap.url_check.latency_ms} ms), "
            f"Worker: {snap.workflow_check.state.value}, "
            f"DB: {snap.db_check.total_rows} Datensätze ({snap.db_check.backend_label})."
        )
        insights.append(
            AppDiagnosticInsight(
                app_id=snap.spec.app_id.value,
                app_title=snap.spec.title,
                status_rating=rating,
                key_observation=obs,
                recommended_action=action,
            )
        )
        if action != "Keine Aktion erforderlich – System läuft regulär.":
            next_steps.append(f"{snap.spec.title}: {action}")

    clamped_score = max(0, min(100, score))
    online_count = sum(1 for s in snapshots if s.overall_status == HealthStatus.ONLINE)
    headline = (
        f"System-Status: {online_count} von {len(snapshots)} Bots voll einsatzbereit "
        f"(Health-Score {clamped_score}/100)"
    )
    summary = (
        f"Automatische Auswertung über {len(snapshots)} Apps abgeschlossen. "
        f"{len(warnings)} Hinweise oder Warnungen erkannt."
    )
    if not next_steps:
        next_steps.append("Alle 3 Bots arbeiten unauffällig – reguläres Monitoring fortsetzen.")

    return FleetAIAnalysis(
        fleet_health_score=clamped_score,
        headline=headline,
        executive_summary=summary,
        app_insights=insights,
        anomalies_or_warnings=warnings,
        next_steps=next_steps,
        model_used="heuristic-engine",
    )


class DashboardAIGateway:
    """Führt KI-gestützte Flotten-Diagnosen und Q&A über die Gemini Interactions API aus."""

    def __init__(
        self,
        settings: AppSettings,
        client: genai.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._models = self._build_model_chain(settings)

    @staticmethod
    def _build_model_chain(settings: AppSettings) -> list[str]:
        candidates = [settings.gemini_fast_model, settings.gemini_deep_model, *settings.allowed_models]
        chain: list[str] = []
        for model in candidates:
            clean = model.strip()
            if clean and clean not in chain:
                chain.append(clean)
        return chain or list(CANDIDATE_MODELS)

    def _get_or_create_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self._settings.has_gemini_configured:
            raise LLMProviderError("Kein gültiger GEMINI_API_KEY konfiguriert.")
        http_opts = {"timeout": int(LLM_TIMEOUT_SECONDS * 1000), "retry_options": {"attempts": 1}}
        try:
            self._client = genai.Client(api_key=self._settings.gemini_api_key, http_options=http_opts)
        except TypeError:
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def analyze_fleet(
        self,
        snapshots: Sequence[AppFleetSnapshot],
        prefer_heuristic_on_error: bool = True,
    ) -> FleetAIAnalysis:
        """Erzeugt eine strukturierte KI-Gesamtdiagnose für alle 3 überwachten Bots."""
        if not self._settings.has_gemini_configured:
            return build_heuristic_fleet_analysis(snapshots)

        telemetry = format_snapshots_telemetry_context(snapshots)
        sys_prompt = (
            self._settings.fleet_diagnosis_prompt
            or "Analysiere die Telemetrie der 3 Bots präzise und faktenbasiert."
        )
        client = self._get_or_create_client()
        last_exc: Exception | None = None

        for attempt in range(LLM_MAX_ATTEMPTS):
            model_name = self._models[attempt % len(self._models)]
            try:
                interaction = client.interactions.create(
                    model=model_name,
                    input=telemetry,
                    system_instruction=sys_prompt,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": FleetAIAnalysis.model_json_schema(),
                    },
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                raw_json = _JSON_FENCE_RE.sub("", _extract_interaction_text(interaction)).strip()
                if not raw_json:
                    raise ValueError("Leere JSON-Antwort von Gemini erhalten.")
                parsed = FleetAIAnalysis.model_validate_json(raw_json)
                return parsed.model_copy(update={"model_used": model_name})
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini System-Analyse (%s) fehlgeschlagen: %s", model_name, exc)
                if attempt + 1 < LLM_MAX_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_BASE_SECONDS * (attempt + 1))

        if prefer_heuristic_on_error:
            return build_heuristic_fleet_analysis(snapshots)
        raise LLMProviderError(f"KI-Systemanalyse fehlgeschlagen: {last_exc}")

    def answer_fleet_question(
        self,
        snapshots: Sequence[AppFleetSnapshot],
        question: str,
    ) -> str:
        """Beantwortet eine Freitext-Frage des Nutzers auf Basis der Live-Telemetrie."""
        clean_q = validate_user_question(question)
        telemetry = format_snapshots_telemetry_context(snapshots)

        if not self._settings.has_gemini_configured:
            heuristic = build_heuristic_fleet_analysis(snapshots)
            return (
                f"**Hinweis (Lokaler Diagnose-Modus ohne Gemini API-Key):**\n\n"
                f"{heuristic.headline}\n\n{heuristic.executive_summary}\n\n"
                + "\n".join(f"- **{i.app_title}**: {i.key_observation}" for i in heuristic.app_insights)
            )

        client = self._get_or_create_client()
        prompt = f"{telemetry}\n\nBenutzerfrage:\n{clean_q}"
        sys_prompt = (
            self._settings.fleet_qa_prompt
            or "Beantworte die Frage präzise auf Basis der Live-Telemetrie der 3 Bots."
        )

        last_exc: Exception | None = None
        for attempt in range(LLM_MAX_ATTEMPTS):
            model_name = self._models[attempt % len(self._models)]
            try:
                interaction = client.interactions.create(
                    model=model_name,
                    input=prompt,
                    system_instruction=sys_prompt,
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                answer = _extract_interaction_text(interaction)
                if answer:
                    return answer
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini Q&A (%s) fehlgeschlagen: %s", model_name, exc)
                if attempt + 1 < LLM_MAX_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_BASE_SECONDS * (attempt + 1))

        raise LLMProviderError(f"KI-Antwort konnte nicht erstellt werden: {last_exc}")
