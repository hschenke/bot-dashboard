"""KI-Brain: UI-State-Cache, Formatierung & Task-Overlay-Integration."""

from __future__ import annotations

import time

import streamlit as st
from ki_brain import (
    BrainDashboardState,
    BrainTier,
    DiscoverySummary,
    KIWebsearchBrain,
    PipelineSummary,
    ProbeStatus,
    SecretStatusEntry,
    UnblockSummary,
)

from src.ui.task_overlay import (
    INSTANT_OVERLAY_HTML,
    render_instant_overlay_bridge,
    run_with_task_overlay,
)

__all__ = [
    "INSTANT_OVERLAY_HTML",
    "format_cleanup_message",
    "format_discover_feedback",
    "format_pipeline_feedback",
    "format_unblock_feedback",
    "invalidate_ki_ui_cache",
    "load_cached_ui_state",
    "render_instant_overlay_bridge",
    "run_with_task_overlay",
]

_KI_UI_CACHE_KEY = "_ki_ui_state_cache"
_KI_UI_CACHE_TTL_SECONDS = 20.0


def invalidate_ki_ui_cache() -> None:
    """Verwirft den kurzlebigen UI-State-Cache nach Schreibaktionen oder Seitenwechsel."""
    st.session_state.pop(_KI_UI_CACHE_KEY, None)


def load_cached_ui_state(
    brain: KIWebsearchBrain,
) -> tuple[BrainDashboardState, list[SecretStatusEntry]]:
    """Lädt den gebündelten KI-Brain-State kurzzeitig aus `session_state`, um Dialog-/Widget-Reruns ohne DB-Latenz zu bedienen."""
    now = time.monotonic()
    backend = brain.backend_name
    cached = st.session_state.get(_KI_UI_CACHE_KEY)
    if (
        isinstance(cached, dict)
        and cached.get("backend") == backend
        and (now - float(cached.get("ts", 0.0))) < _KI_UI_CACHE_TTL_SECONDS
        and cached.get("ui_state") is not None
        and cached.get("secret_statuses") is not None
    ):
        return cached["ui_state"], cached["secret_statuses"]
    ui_state, secret_statuses = brain.load_ui_state()
    st.session_state[_KI_UI_CACHE_KEY] = {
        "backend": backend,
        "ts": now,
        "ui_state": ui_state,
        "secret_statuses": secret_statuses,
    }
    return ui_state, secret_statuses


def format_unblock_feedback(summary: UnblockSummary) -> tuple[str, str]:
    """Formatiert das Ergebnis von `Unblock Tools` inkl. konkreter Hilfe bei DuckDuckGo HTTP 202."""
    if not summary.checked_blocked and not summary.activated_alternatives:
        return "info", "Keine Blocked-Tools gefunden — alle Such-Tools sind bereits einsatzbereit oder per API-Key aktivierbar."
    parts = [f"Unblock Tools: {len(summary.unblocked_ids)}/{len(summary.checked_blocked)} entsperrt"]
    if summary.unblocked_ids:
        parts.append(f"OK: {', '.join(summary.unblocked_ids)}")
    if summary.activated_alternatives:
        parts.append(f"Pool aktiviert: {', '.join(summary.activated_alternatives)}")
    if summary.rules_applied:
        parts.append(f"{len(summary.rules_applied)} Regel(n) angewendet")
    ddg_blocked = False
    if summary.still_blocked:
        formatted = []
        for tid, raw in list(summary.still_blocked.items())[:3]:
            reason = raw if len(raw) <= 115 else f"{raw[:112].rstrip()}…"
            formatted.append(f"{tid} ({reason})")
            if "duckduckgo" in tid or "202" in raw or "anti-bot" in raw.lower():
                ddg_blocked = True
        parts.append(f"Weiterhin Blocked: {', '.join(formatted)}")
    text = " · ".join(parts) + "."
    if ddg_blocked:
        text += (
            "\n\n💡 **Was du bei DuckDuckGo `HTTP 202` tun kannst:** Cloud-Datacenter-IPs erhalten von DuckDuckGo oft eine "
            "serverseitige Anti-Bot-Leerantwort (nicht per Code umgehbar). Hinterlege unter **🔐 Vault** einen "
            "kostenfreien Key (`TAVILY_API_KEY` oder `SERPER_API_KEY`) oder nutze key-frei "
            "`yahoo_search` + `searxng_public`."
        )
    level = "success" if (summary.unblocked_ids or summary.activated_alternatives) and not summary.still_blocked else "warning"
    return level, text


def format_cleanup_message(brain: KIWebsearchBrain) -> tuple[str, str]:
    """Führt `Prune Failing` aus und dokumentiert exakt, ob `Error`, `Quota 429` oder `HTTP 202` vorlag."""
    rem_m, rem_t = brain.cleanup_unhealthy_items_with_status()
    if not rem_m and not rem_t:
        return "info", "Prune Failing: Keine Error-, Quota-429- oder HTTP-202-Elemente zum Entfernen gefunden (sonstige Blocked-Tools bleiben erhalten)."

    def _lbl(status: ProbeStatus) -> str:
        if status == ProbeStatus.QUOTA_LIMITED:
            return "Quota 429"
        if status == ProbeStatus.BLOCKED:
            return "HTTP 202"
        return "Error"

    m_txt = ", ".join(f"{mid} [{_lbl(sv)}]" for mid, sv in rem_m) or "–"
    t_txt = ", ".join(f"{tid} [{_lbl(sv)}]" for tid, sv in rem_t) or "–"
    return (
        "success",
        f"Prune Failing: {len(rem_m)} Modelle ({m_txt}) & {len(rem_t)} Tools ({t_txt}) mit Status Error/429/202 entfernt (übrige Blocked erhalten).",
    )


def format_discover_feedback(summary: DiscoverySummary, *, prefix: str = "Discover fertig") -> tuple[str, str]:
    """Formatiert eine verständliche Zusammenfassung für `Discover`."""
    parts = [f"+{len(summary.added_models)} Modelle", f"+{len(summary.added_tools)} Tools"]
    if summary.recovered_models:
        parts.append(f"{len(summary.recovered_models)} von Quota/Error auf OK")
    skipped = len(summary.skipped_excluded) + len(summary.skipped_unhealthy)
    notes = [
        n
        for n in (
            f"{skipped} defekte/ausgeschlossene" if skipped else "",
            f"{len(summary.skipped_previews)} Preview-Modelle" if summary.skipped_previews else "",
        )
        if n
    ]
    note_txt = f" ({', '.join(notes)} übersprungen)" if notes else ""
    warn_txt = f" · Hinweise: {'; '.join(summary.warnings[:2])}" if summary.warnings else ""
    level = "warning" if summary.warnings else "success"
    return level, f"{prefix} ({', '.join(summary.checked_sources)}): {', '.join(parts)}{note_txt}.{warn_txt}"


def format_pipeline_feedback(summary: PipelineSummary) -> tuple[str, str]:
    """Formatiert das Gesamtergebnis des kombinierten KI-Brain-Ablaufs kompakt."""
    if not summary.executed_steps:
        return "warning", "Kein Ablauf-Schritt ausgewählt — bitte mindestens eine Option unter 'Ablauf-Optionen' aktivieren."

    parts: list[str] = []
    if summary.discovery is not None:
        d = summary.discovery
        rec = f", {len(d.recovered_models)} recovered" if d.recovered_models else ""
        parts.append(f"Discover: +{len(d.added_models)}M / +{len(d.added_tools)}T{rec}")
    if summary.benchmark is not None:
        b = summary.benchmark
        fast_seq = b.sequences.get(BrainTier.FAST.value)
        fast_top = " → ".join(fast_seq.search_tool_sequence[:3]) if fast_seq else "–"
        parts.append(f"Benchmark `{b.run_id}`: {b.healthy_tools_count}T / {b.healthy_models_count}M OK (Fast: `{fast_top or '–'}`)")
    if summary.unblock is not None:
        u = summary.unblock
        if u.checked_blocked or u.activated_alternatives:
            parts.append(f"Unblock: {len(u.unblocked_ids)}/{len(u.checked_blocked)} entsperrt")
    if "prune" in summary.executed_steps:
        parts.append(f"Prune: -{len(summary.pruned_models)}M / -{len(summary.pruned_tools)}T")

    text = "Sync & Optimize abgeschlossen — " + " · ".join(parts) + "."
    if summary.warnings:
        text += f" · Hinweise: {'; '.join(summary.warnings[:2])}"
    has_warn = bool(summary.warnings) or bool(summary.unblock and summary.unblock.still_blocked)
    return ("warning" if has_warn else "success"), text
