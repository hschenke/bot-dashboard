"""KI-View: Zentrales KI & Websearch Brain Control Center."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, BrainTier, KIWebsearchBrain, UnblockSummary

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_models_tab import render_models_tab
from src.ui.views.ki_playground_tab import render_playground_tab
from src.ui.views.ki_secret_dialog import render_active_secret_dialog
from src.ui.views.ki_secrets_section import render_secrets_manager_expander
from src.ui.views.ki_sequences_tab import render_sequences_tab
from src.ui.views.ki_tools_tab import render_tools_tab


def _flash(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_feedback_banner() -> None:
    feedback = st.session_state.pop("ki_brain_feedback", None)
    if not isinstance(feedback, dict):
        return
    level, msg = str(feedback.get("level") or "info"), str(feedback.get("message") or "").strip()
    if msg:
        getattr(st, level if level in {"success", "warning", "error"} else "info")(msg)


def _format_unblock_feedback(summary: UnblockSummary) -> tuple[str, str]:
    if not summary.checked_blocked and not summary.activated_alternatives:
        return "info", "Keine Blocked-Tools gefunden — alle Such-Tools sind bereits einsatzbereit oder per API-Key aktivierbar."
    parts = [f"Blocked lösen: {len(summary.unblocked_ids)}/{len(summary.checked_blocked)} entsperrt"]
    if summary.unblocked_ids:
        parts.append(f"OK: {', '.join(summary.unblocked_ids)}")
    if summary.activated_alternatives:
        parts.append(f"Pool aktiviert: {', '.join(summary.activated_alternatives)}")
    if summary.rules_applied:
        parts.append(f"{len(summary.rules_applied)} Regel(n) angewendet")
    if summary.still_blocked:
        rem = ", ".join(f"{tid} ({reason[:65]})" for tid, reason in list(summary.still_blocked.items())[:3])
        parts.append(f"Weiterhin Blocked: {rem}")
    level = "success" if (summary.unblocked_ids or summary.activated_alternatives) and not summary.still_blocked else "warning"
    return level, " · ".join(parts) + "."


def _handle_top_actions(
    brain: KIWebsearchBrain, *, run_sync: bool, run_bench: bool, run_clean: bool, run_unblock: bool
) -> None:
    if run_bench:
        try:
            with st.spinner("Teste Tools & KI-Modelle live und optimiere Fast / Standard / Deep..."):
                report = brain.run_benchmark()
            fast_top = " → ".join(report.sequences[BrainTier.FAST.value].search_tool_sequence[:3])
            _flash(
                "success",
                f"Testlauf & Optimizing `{report.run_id}` abgeschlossen: {report.healthy_tools_count} Tools & "
                f"{report.healthy_models_count} Modelle OK. Fast-Top: `{fast_top or '–'}`",
            )
        except BrainError as exc:
            _flash("error", f"Testlauf & Optimizing fehlgeschlagen: {exc}")
        st.rerun()
    if run_sync:
        try:
            with st.spinner("Suche nach neuen funktionierenden Modellen, APIs & searx.space-Instanzen..."):
                s = brain.auto_sync(verify_discovered=True)
            skipped = len(s.skipped_excluded) + len(s.skipped_unhealthy)
            skip_note = f" ({skipped} defekte/ausgeschlossene übersprungen)" if skipped else ""
            _flash(
                "success" if not s.warnings else "warning",
                f"Search for New fertig ({', '.join(s.checked_sources)}): +{len(s.added_models)} Modelle, +{len(s.added_tools)} Tools{skip_note}.",
            )
        except BrainError as exc:
            _flash("error", f"Search for New fehlgeschlagen: {exc}")
        st.rerun()
    if run_clean:
        try:
            rem_m, rem_t = brain.cleanup_unhealthy_items()
            total = len(rem_m) + len(rem_t)
            msg = (
                f"Clear Errors: {len(rem_m)} Modelle ({', '.join(rem_m) or '–'}) & {len(rem_t)} Tools ({', '.join(rem_t) or '–'}) entfernt (Blocked erhalten)."
                if total
                else "Keine Error- oder Quota-429-Elemente zum Entfernen gefunden (Blocked-Tools bleiben erhalten)."
            )
            _flash("success" if total else "info", msg)
        except BrainError as exc:
            _flash("error", f"Clear Errors fehlgeschlagen: {exc}")
        st.rerun()
    if run_unblock:
        try:
            with st.spinner("Wende Rules an, prüfe searx.space-Instanzen & teste Blocked-Tools live..."):
                _flash(*_format_unblock_feedback(brain.resolve_blocked_tools()))
        except BrainError as exc:
            _flash("error", f"Blocked lösen fehlgeschlagen: {exc}")
        st.rerun()


def render_ki_view(settings: AppSettings, role: AuthRole = AuthRole.ADMIN) -> None:
    """Rendert das zentrale KI & Websearch Brain zur Modell-, Tool- & Sequenz-Steuerung."""
    st.header("🧠 KI & Websearch Brain")
    st.caption(
        "Zentrale Schaltstelle für alle kostenfreien KI-Modelle & Websearch-Tools "
        "(Research-Bot, Alert-Bot & News-Bot) mit automatischem Benchmark-Sequencer für Fast, Standard & Deep."
    )

    is_admin = role == AuthRole.ADMIN
    brain = create_dashboard_brain(settings)
    try:
        models, tools, latest_bench = brain.list_models(), brain.list_tools(), brain.get_latest_benchmark()
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("KI-Modelle", f"{sum(1 for m in models if m.enabled)} / {len(models)}")
        with k2:
            st.metric("Websearch & Tools", f"{sum(1 for t in tools if t.enabled)} / {len(tools)}")
        with k3:
            st.metric("Tier-Profile", f"{len(brain.list_sequences())} (Fast / Std / Deep)")
        with k4:
            bench_txt = f"{latest_bench.healthy_tools_count} Tools / {latest_bench.healthy_models_count} Modelle OK" if latest_bench else "Noch kein Testlauf"
            st.metric("Letzter Testlauf", bench_txt)

        btn_sync, btn_bench, btn_clean, btn_unblock = st.columns(4, vertical_alignment="center")
        with btn_sync:
            run_sync = st.button("🔎 Search for New", disabled=not is_admin, use_container_width=True, help="Sucht nach neuen kostenfreien KI-Modellen (Gemini, Groq, OpenRouter), Brave/Tavily/Serper & searx.space-Instanzen.")
        with btn_bench:
            run_bench = st.button("🚀 Testlauf & Optimizing", disabled=not is_admin, use_container_width=True, help="Testet alle aktivierten Modelle & Tools live und optimiert automatisch die Sequenzen für Fast, Standard & Deep.")
        with btn_clean:
            run_clean = st.button("🧹 Clear Errors", disabled=not is_admin, use_container_width=True, help="Entfernt Modelle und Tools mit Status Error oder Quota 429 dauerhaft. Blocked-Tools bleiben zur Analyse erhalten.")
        with btn_unblock:
            run_unblock = st.button("🔓 Blocked lösen", disabled=not is_admin, use_container_width=True, help="Wendet automatische Query-Rules an, aktiviert verifizierte searx.space-Instanzen & Alternativ-Endpunkte und testet Blocked-Tools live.")

        # Spinner & Alerts strikt außerhalb schmaler Spalten rendern
        _handle_top_actions(brain, run_sync=run_sync, run_bench=run_bench, run_clean=run_clean, run_unblock=run_unblock)

        _render_feedback_banner()
        render_secrets_manager_expander(brain, is_admin=is_admin)
        st.caption(f"Speicher-Backend: `{brain.backend_name}`" + ("" if is_admin else " · Modus: Read-Only"))

        tab_seq, tab_mod, tab_tool, tab_play = st.tabs(
            [
                "🎯 Sequenzen (Fast / Standard / Deep)",
                "🧩 KI-Modelle",
                "🔧 Websearch & Tools",
                "🧪 Live-Test & Bots-API",
            ]
        )
        with tab_seq:
            render_sequences_tab(brain, is_admin=is_admin)
        with tab_mod:
            render_models_tab(brain, is_admin=is_admin)
        with tab_tool:
            render_tools_tab(brain, is_admin=is_admin)
        with tab_play:
            render_playground_tab(brain, is_admin=is_admin)
    finally:
        brain.close()

    render_active_secret_dialog(settings, is_admin=is_admin)
