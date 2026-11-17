"""KI-View: Gray/Blocked Task-Overlay im Hintergrund & Live-Fortschritt im Vordergrund."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import TypeVar

import streamlit as st
from ki_brain import (
    BrainDashboardState,
    KIWebsearchBrain,
    ProbeStatus,
    SecretStatusEntry,
    UnblockSummary,
)

_T = TypeVar("_T")
_KI_UI_CACHE_KEY = "_ki_ui_state_cache"
_KI_UI_CACHE_TTL_SECONDS = 20.0


def invalidate_ki_ui_cache() -> None:
    """Verwirft den kurzlebigen UI-State-Cache nach Schreibaktionen oder Seitenwechsel."""
    st.session_state.pop(_KI_UI_CACHE_KEY, None)


def load_cached_ui_state(
    brain: KIWebsearchBrain,
) -> tuple[BrainDashboardState, list[SecretStatusEntry]]:
    """Lädt den gebündelten KI-View-State kurzzeitig aus `session_state`, um Dialog-/Widget-Reruns ohne DB-Latenz zu bedienen."""
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


INSTANT_OVERLAY_HTML = """
<div id="ki-brain-instant-backdrop">
  <div class="ki-instant-card">
    <strong>⚙️ Aufgabe wird ausgeführt…</strong>
    <div id="ki-instant-sub" style="font-size:0.9rem;color:#94a3b8;margin-top:0.35rem;">Bitte kurz warten — Hintergrund wird gesperrt.</div>
  </div>
</div>
<script>
(function () {
  if (window.__kiBrainOverlayBound) return;
  window.__kiBrainOverlayBound = true;
  document.addEventListener("click", function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('[class*="st-key-ki_top_btn_"] button') : null;
    if (!btn || btn.disabled) return;
    var bd = document.getElementById("ki-brain-instant-backdrop");
    var sub = document.getElementById("ki-instant-sub");
    if (sub) sub.textContent = (btn.innerText || "Task").trim() + " läuft — Live-Fortschritt folgt…";
    if (bd) bd.classList.add("active");
    if (window.__kiBrainWatch) clearInterval(window.__kiBrainWatch);
    var idle = 0;
    window.__kiBrainWatch = setInterval(function () {
      var cur = document.getElementById("ki-brain-instant-backdrop");
      if (!cur) { clearInterval(window.__kiBrainWatch); return; }
      var busy = document.querySelector('[data-testid="stStatusWidgetRunningIcon"], [data-test-script-state="running"], .st-key-ki_task_overlay');
      if (!busy) {
        if (++idle >= 3) { cur.classList.remove("active"); clearInterval(window.__kiBrainWatch); }
      } else { idle = 0; }
    }, 120);
  }, true);
})();
</script>
"""


def render_instant_overlay_bridge() -> None:
    """Rendert den sofortigen JS-Klick-Blocker beim Klick auf die 4 Top-Action-Buttons."""
    with contextlib.suppress(TypeError):
        st.html(INSTANT_OVERLAY_HTML, unsafe_allow_javascript=True)


def run_with_task_overlay(
    title: str,
    initial_text: str,
    work_fn: Callable[[Callable[[float, str], None]], _T],
) -> _T:
    """Sperrt den Hintergrund per Blur-Backdrop und hält Live-Status + Progress im Vordergrund."""
    st.markdown('<div class="ki-task-backdrop" aria-hidden="true"></div>', unsafe_allow_html=True)
    with st.container(key="ki_task_overlay"):
        st.markdown(f"#### {title}")
        status_slot = st.empty()
        progress_bar = st.progress(0.05, text=initial_text)
        status_slot.info(initial_text)

        def _on_progress(fraction: float, message: str) -> None:
            progress_bar.progress(max(0.0, min(1.0, float(fraction))), text=message)
            status_slot.info(message)

        with st.spinner(initial_text):
            return work_fn(_on_progress)


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
            "serverseitige Anti-Bot-Leerantwort (nicht per Code umgehbar). Hinterlege unter **🔧 Websearch & Tools** einen "
            "kostenfreien Key (`TAVILY_API_KEY`, `SERPER_API_KEY` oder `BRAVE_SEARCH_API_KEY`) oder nutze key-frei "
            "`yahoo_search` + `searxng_public` (DDG kannst du dort optional deaktivieren)."
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
