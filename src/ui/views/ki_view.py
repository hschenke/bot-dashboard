"""KI-View (Fleet Intelligence): Strukturierte Systemdiagnose & interaktiver Assistent."""

from __future__ import annotations

import streamlit as st

from src.ai.llm_gateway import DashboardAIGateway, build_heuristic_fleet_analysis
from src.config.settings import AppSettings
from src.domain.exceptions import DashboardError
from src.domain.models import AppFleetSnapshot
from src.domain.schemas import FleetAIAnalysis


def _rating_badge(rating: str) -> str:
    clean = (rating or "").strip().lower()
    if clean == "optimal":
        return "🟢 Optimal"
    if clean == "hinweis":
        return "🟡 Hinweis"
    if clean == "kritisch":
        return "🔴 Kritisch"
    return f"🔵 {rating}"


def _render_analysis_report(analysis: FleetAIAnalysis) -> None:
    score_col, model_col = st.columns([3, 1])
    with score_col:
        st.subheader(f"🧠 {analysis.headline}")
        st.write(analysis.executive_summary)
    with model_col:
        st.metric("Fleet Health Score", f"{analysis.fleet_health_score} / 100")
        st.caption(f"Engine: `{analysis.model_used}`")

    if analysis.app_insights:
        st.markdown("#### 🔍 Einzelbewertung der Apps")
        cols = st.columns(len(analysis.app_insights))
        for idx, insight in enumerate(analysis.app_insights):
            with cols[idx], st.container(border=True):
                st.markdown(f"**{insight.app_title}** ({_rating_badge(insight.status_rating)})")
                st.caption(insight.key_observation)
                st.markdown(f"👉 *{insight.recommended_action}*")

    col_warn, col_steps = st.columns(2)
    with col_warn:
        st.markdown("#### ⚠️ Erkannte Anomalien & Hinweise")
        if analysis.anomalies_or_warnings:
            for warning in analysis.anomalies_or_warnings:
                st.warning(warning)
        else:
            st.success("Keine Anomalien oder Störungen erkannt.")

    with col_steps:
        st.markdown("#### ✅ Empfohlene nächste Schritte")
        for step in analysis.next_steps:
            st.info(step)


def render_ki_view(
    settings: AppSettings,
    snapshots: list[AppFleetSnapshot],
) -> None:
    """Rendert den KI-View mit Status-Diagnose und interaktivem Operations-Assistenten."""
    st.header("🤖 KI-View & Dashboard-Intelligenz (Google Gemini)")
    st.caption(
        "Automatische Auswertung aller URL-, GitHub-Worker- und Datenbank-Metriken deiner Apps "
        "mit Google Gemini Structured Output (inkl. sofortigem Heuristik-Fallback)."
    )

    gateway = DashboardAIGateway(settings)

    if "fleet_ai_analysis" not in st.session_state:
        st.session_state["fleet_ai_analysis"] = build_heuristic_fleet_analysis(snapshots)

    btn_col, info_col = st.columns([1.4, 2.6])
    with btn_col:
        if st.button("✨ KI-Tiefendiagnose starten", type="primary", use_container_width=True):
            with st.spinner("Analysiere Live-Telemetrie aller Apps mit Google Gemini..."):
                st.session_state["fleet_ai_analysis"] = gateway.analyze_fleet(snapshots)
            st.rerun()
    with info_col:
        if settings.has_gemini_configured:
            st.caption(
                f"Gemini API aktiv (`{settings.gemini_fast_model}` / Fallback `{settings.gemini_deep_model}`)."
            )
        else:
            st.caption(
                "Hinweis: Kein `GEMINI_API_KEY` gesetzt – das Dashboard nutzt die deterministische Diagnose-Engine."
            )

    _render_analysis_report(st.session_state["fleet_ai_analysis"])
    st.divider()

    st.subheader("💬 KI-Operations-Assistent (Frage an deine Bots)")
    with st.form("ki_fleet_question_form"):
        question = st.text_input(
            "Frage zu Status, Datenbank-Inhalten, Fehlern oder Workflows stellen:",
            placeholder="z. B. Welche App braucht aktuell Aufmerksamkeit und wie viele Datensätze sind gespeichert?",
        )
        ask_submitted = st.form_submit_button("🤖 Frage auswerten", use_container_width=True)

    if ask_submitted:
        try:
            with st.spinner("Werte Telemetriedaten aus..."):
                answer = gateway.answer_fleet_question(snapshots, question)
            st.markdown("##### Antwort")
            st.markdown(answer)
        except DashboardError as exc:
            st.error(str(exc))
