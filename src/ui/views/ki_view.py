"""KI-View: Zentrales KI & Websearch Brain Control Center."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, BrainTier

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_models_tab import render_models_tab
from src.ui.views.ki_playground_tab import render_playground_tab
from src.ui.views.ki_secret_dialog import render_active_secret_dialog
from src.ui.views.ki_secrets_section import render_secrets_manager_expander
from src.ui.views.ki_sequences_tab import render_sequences_tab
from src.ui.views.ki_tools_tab import render_tools_tab


def _render_feedback_banner() -> None:
    feedback = st.session_state.pop("ki_brain_feedback", None)
    if not isinstance(feedback, dict):
        return
    level = str(feedback.get("level") or "info")
    msg = str(feedback.get("message") or "").strip()
    if not msg:
        return
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def render_ki_view(
    settings: AppSettings,
    role: AuthRole = AuthRole.ADMIN,
) -> None:
    """Rendert das zentrale KI & Websearch Brain zur Modell-, Tool- & Sequenz-Steuerung."""
    st.header("🧠 KI & Websearch Brain")
    st.caption(
        "Zentrale Schaltstelle für alle kostenfreien KI-Modelle & Websearch-Tools "
        "(Research-Bot, Alert-Bot & News-Bot) mit automatischem Benchmark-Sequencer für Fast, Standard & Deep."
    )

    is_admin = role == AuthRole.ADMIN
    brain = create_dashboard_brain(settings)
    try:
        models = brain.list_models()
        tools = brain.list_tools()
        enabled_models = sum(1 for m in models if m.enabled)
        enabled_tools = sum(1 for t in tools if t.enabled)
        latest_bench = brain.get_latest_benchmark()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("KI-Modelle", f"{enabled_models} / {len(models)}")
        with k2:
            st.metric("Websearch & Tools", f"{enabled_tools} / {len(tools)}")
        with k3:
            seqs = brain.list_sequences()
            st.metric("Tier-Profile", f"{len(seqs)} (Fast / Std / Deep)")
        with k4:
            bench_txt = (
                f"{latest_bench.healthy_tools_count} Tools / {latest_bench.healthy_models_count} Modelle OK"
                if latest_bench
                else "Noch kein Testlauf"
            )
            st.metric("Letzter Testlauf", bench_txt)

        btn_bench, btn_sync, btn_clean = st.columns([1.65, 1.2, 1.3], vertical_alignment="center")
        with btn_bench:
            run_bench = st.button(
                "🚀 Testlauf starten & Sequenzen optimieren",
                disabled=not is_admin,
                use_container_width=True,
            )
        with btn_sync:
            run_sync = st.button(
                "🔄 Auto-Sync (Modelle & Tools)",
                disabled=not is_admin,
                use_container_width=True,
            )
        with btn_clean:
            run_clean = st.button(
                "🧹 Bereinigen (Error & Quota 429)",
                disabled=not is_admin,
                use_container_width=True,
                help="Entfernt Modelle und Tools mit Status Error oder Quota 429 dauerhaft. Blocked-Tools bleiben zur Analyse erhalten.",
            )

        # Spinner & Alerts strikt außerhalb schmaler Spalten rendern
        if run_bench:
            try:
                with st.spinner("Teste Tools & KI-Modelle live und optimiere Fast / Standard / Deep..."):
                    report = brain.run_benchmark()
                fast_top = " → ".join(report.sequences[BrainTier.FAST.value].search_tool_sequence[:3])
                st.session_state["ki_brain_feedback"] = {
                    "level": "success",
                    "message": (
                        f"Testlauf `{report.run_id}` abgeschlossen: {report.healthy_tools_count} Tools & "
                        f"{report.healthy_models_count} Modelle OK. Fast-Top: `{fast_top or '–'}`"
                    ),
                }
            except BrainError as exc:
                st.session_state["ki_brain_feedback"] = {"level": "error", "message": f"Testlauf fehlgeschlagen: {exc}"}
            st.rerun()

        if run_sync:
            try:
                with st.spinner("Erkenne funktionierende kostenfreie Modelle & neue Standard-Tools..."):
                    summary = brain.auto_sync(verify_discovered=True)
                skipped_total = len(summary.skipped_excluded) + len(summary.skipped_unhealthy)
                skip_note = (
                    f" ({skipped_total} defekte/ausgeschlossene übersprungen)"
                    if skipped_total
                    else ""
                )
                st.session_state["ki_brain_feedback"] = {
                    "level": "success" if not summary.warnings else "warning",
                    "message": (
                        f"Auto-Sync fertig ({', '.join(summary.checked_sources)}): "
                        f"+{len(summary.added_models)} Modelle, +{len(summary.added_tools)} Tools{skip_note}."
                    ),
                }
            except BrainError as exc:
                st.session_state["ki_brain_feedback"] = {"level": "error", "message": f"Auto-Sync fehlgeschlagen: {exc}"}
            st.rerun()

        if run_clean:
            try:
                rem_models, rem_tools = brain.cleanup_unhealthy_items()
                total_rem = len(rem_models) + len(rem_tools)
                msg = (
                    f"Bereinigung abgeschlossen: {len(rem_models)} Modelle ({', '.join(rem_models) or '–'}) & "
                    f"{len(rem_tools)} Tools ({', '.join(rem_tools) or '–'}) dauerhaft entfernt (Blocked-Tools erhalten)."
                    if total_rem > 0
                    else "Keine Error- oder Quota-429-Elemente zum Entfernen gefunden (Blocked-Tools bleiben erhalten)."
                )
                st.session_state["ki_brain_feedback"] = {"level": "success" if total_rem else "info", "message": msg}
            except BrainError as exc:
                st.session_state["ki_brain_feedback"] = {"level": "error", "message": f"Bereinigung fehlgeschlagen: {exc}"}
            st.rerun()

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
