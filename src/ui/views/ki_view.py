"""KI-View: Zentrales KI & Websearch Brain Control Center."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from ki_brain import BrainError, BrainTier, KIWebsearchBrain

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_models_tab import render_models_tab
from src.ui.views.ki_playground_tab import render_playground_tab
from src.ui.views.ki_secret_dialog import render_active_secret_dialog
from src.ui.views.ki_secrets_section import render_secrets_manager_expander
from src.ui.views.ki_sequences_tab import render_sequences_tab
from src.ui.views.ki_task_overlay import (
    format_cleanup_message,
    format_unblock_feedback,
    invalidate_ki_ui_cache,
    load_cached_ui_state,
    render_instant_overlay_bridge,
    run_with_task_overlay,
)
from src.ui.views.ki_tools_tab import render_tools_tab

KI_TAB_SEQ = "🎯 Sequenzen (Fast / Standard / Deep)"
KI_TAB_MODELS = "🧩 KI-Modelle"
KI_TAB_TOOLS = "🔧 Websearch & Tools"
KI_TAB_PLAY = "🧪 Live-Test & Bots-API"
KI_TAB_LABELS: tuple[str, ...] = (KI_TAB_SEQ, KI_TAB_MODELS, KI_TAB_TOOLS, KI_TAB_PLAY)


def _open_ki_tabs() -> tuple[object, ...]:
    """Rendert Tabs mit stabiler ID ohne `on_change="rerun"` (0 ms Client-Wechsel, Tab bleibt erhalten)."""
    try:
        return tuple(st.tabs(list(KI_TAB_LABELS), key="ki_view_tabs"))
    except TypeError:
        return tuple(st.tabs(list(KI_TAB_LABELS)))


def _flash(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_feedback_banner() -> None:
    feedback = st.session_state.pop("ki_brain_feedback", None)
    if not isinstance(feedback, dict):
        return
    level, msg = str(feedback.get("level") or "info"), str(feedback.get("message") or "").strip()
    if msg:
        getattr(st, level if level in {"success", "warning", "error"} else "info")(msg)


def _handle_top_actions(
    brain: KIWebsearchBrain, *, run_sync: bool, run_bench: bool, run_clean: bool, run_unblock: bool, exclude_previews: bool
) -> None:
    if run_bench:
        try:
            report = run_with_task_overlay(
                "🚀 Benchmark & Optimize",
                "Teste aktive Tools & KI-Modelle live und optimiere Fast / Standard / Deep...",
                lambda cb: brain.run_benchmark(on_progress=cb),
            )
            fast_top = " → ".join(report.sequences[BrainTier.FAST.value].search_tool_sequence[:3])
            _flash("success", f"Benchmark & Optimize `{report.run_id}` abgeschlossen: {report.healthy_tools_count} Tools & {report.healthy_models_count} Modelle OK. Fast-Top: `{fast_top or '–'}`")
        except BrainError as exc:
            _flash("error", f"Benchmark & Optimize fehlgeschlagen: {exc}")
        invalidate_ki_ui_cache()
        st.rerun()
    if run_sync:
        try:
            s = run_with_task_overlay(
                "🔎 Discover",
                "Suche neue Gemini-/Groq-/OpenRouter-Modelle & searx.space-Instanzen...",
                lambda cb: brain.auto_sync(verify_discovered=True, exclude_previews=exclude_previews, on_progress=cb),
            )
            skipped = len(s.skipped_excluded) + len(s.skipped_unhealthy)
            notes = [n for n in (f"{skipped} defekte/ausgeschlossene" if skipped else "", f"{len(s.skipped_previews)} Preview-Modelle" if s.skipped_previews else "") if n]
            note_txt = f" ({', '.join(notes)} übersprungen)" if notes else ""
            _flash("success" if not s.warnings else "warning", f"Discover fertig ({', '.join(s.checked_sources)}): +{len(s.added_models)} Modelle, +{len(s.added_tools)} Tools{note_txt}.")
        except BrainError as exc:
            _flash("error", f"Discover fehlgeschlagen: {exc}")
        invalidate_ki_ui_cache()
        st.rerun()
    if run_clean:
        try:
            def _clean_task(cb: Callable[[float, str], None]) -> tuple[str, str]:
                cb(0.35, "Bereinige Einträge mit Status Error, Quota 429 oder HTTP 202 (übrige Blocked bleiben erhalten)...")
                res = format_cleanup_message(brain)
                cb(1.0, "Bereinigung abgeschlossen.")
                return res
            _flash(*run_with_task_overlay("🧹 Prune Failing", "Prüfe Modelle & Tools auf Status Error, Quota 429 oder HTTP 202...", _clean_task))
        except BrainError as exc:
            _flash("error", f"Prune Failing fehlgeschlagen: {exc}")
        invalidate_ki_ui_cache()
        st.rerun()
    if run_unblock:
        try:
            summary = run_with_task_overlay(
                "🔓 Unblock Tools",
                "Wende Rules an, prüfe searx.space-Instanzen & teste Blocked-Tools live...",
                lambda cb: brain.resolve_blocked_tools(on_progress=cb),
            )
            _flash(*format_unblock_feedback(summary))
        except BrainError as exc:
            _flash("error", f"Unblock Tools fehlgeschlagen: {exc}")
        invalidate_ki_ui_cache()
        st.rerun()


def render_ki_view(settings: AppSettings, role: AuthRole = AuthRole.ADMIN) -> None:
    """Rendert das zentrale KI & Websearch Brain zur Modell-, Tool- & Sequenz-Steuerung."""
    st.header("🧠 KI & Websearch Brain")
    st.caption("Zentrale Schaltstelle für alle kostenfreien KI-Modelle & Websearch-Tools mit automatischem Benchmark-Sequencer für Fast, Standard & Deep.")

    is_admin = role == AuthRole.ADMIN
    brain = create_dashboard_brain(settings)
    try:
        ui_state, secret_statuses = load_cached_ui_state(brain)
        models, tools, seqs, latest_bench = ui_state.models, ui_state.tools, ui_state.sequences, ui_state.latest_benchmark
        with st.container(key="ki_kpi_row"):
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("KI-Modelle", f"{sum(1 for m in models if m.enabled)} / {len(models)}")
            with k2:
                st.metric("Websearch & Tools", f"{sum(1 for t in tools if t.enabled)} / {len(tools)}")
            with k3:
                st.metric("Tier-Profile", f"{len(seqs)} (Fast / Std / Deep)")
            with k4:
                bench_txt = f"{latest_bench.healthy_tools_count}T / {latest_bench.healthy_models_count}M OK" if latest_bench else "Noch kein Testlauf"
                st.metric("Letzter Testlauf", bench_txt, help="Letztes Benchmark-Ergebnis (gesunde Tools / Modelle).")

        b1, b2, b3, b4 = st.columns(4, vertical_alignment="center")
        with b1:
            run_sync = st.button("🔎 Discover", key="ki_top_btn_sync", disabled=not is_admin, use_container_width=True, help="Sucht neue kostenfreie Modelle (Gemini, Groq, OpenRouter) & searx.space-Instanzen.")
        with b2:
            run_bench = st.button("🚀 Benchmark & Optimize", key="ki_top_btn_bench", disabled=not is_admin, use_container_width=True, help="Testet alle aktivierten Modelle & Tools live und optimiert Fast, Standard & Deep.")
        with b3:
            run_clean = st.button("🧹 Prune Failing", key="ki_top_btn_clean", disabled=not is_admin, use_container_width=True, help="Entfernt Modelle/Tools mit Status Error, Quota 429 oder HTTP 202 dauerhaft. Sonstige Blocked-Tools bleiben erhalten.")
        with b4:
            run_unblock = st.button("🔓 Unblock Tools", key="ki_top_btn_unblock", disabled=not is_admin, use_container_width=True, help="Wendet Rules an, prüft searx.space-Instanzen & testet Blocked-Tools live.")
        exclude_previews = st.checkbox(
            "Preview- & Experimental-Modelle bei 'Discover' ausschließen",
            value=True, key="ki_exclude_previews", disabled=not is_admin,
            help="Ignoriert ältere `-preview`/`-exp` Gemini-Modelle (Deep Research Agent bleibt erhalten).",
        )
        if is_admin:
            render_instant_overlay_bridge()

        # Overlay, Spinner & Alerts strikt außerhalb schmaler Spalten rendern
        _handle_top_actions(brain, run_sync=run_sync, run_bench=run_bench, run_clean=run_clean, run_unblock=run_unblock, exclude_previews=exclude_previews)
        _render_feedback_banner()
        render_secrets_manager_expander(brain, is_admin=is_admin, statuses=secret_statuses)
        st.caption(f"Speicher-Backend: `{brain.backend_name}`" + ("" if is_admin else " · Modus: Read-Only"))

        tab_seq, tab_mod, tab_tool, tab_play = _open_ki_tabs()
        with tab_seq:
            render_sequences_tab(brain, is_admin=is_admin, models=models, tools=tools, sequences=seqs, latest_benchmark=latest_bench, secret_statuses=secret_statuses, active_tab_label=KI_TAB_SEQ)
        with tab_mod:
            render_models_tab(brain, is_admin=is_admin, models=models, secret_statuses=secret_statuses, active_tab_label=KI_TAB_MODELS)
        with tab_tool:
            render_tools_tab(brain, is_admin=is_admin, tools=tools, secret_statuses=secret_statuses, active_tab_label=KI_TAB_TOOLS)
        with tab_play:
            render_playground_tab(brain, is_admin=is_admin, ui_state=ui_state)
    finally:
        brain.close()

    render_active_secret_dialog(settings, is_admin=is_admin, secret_statuses=secret_statuses)
