"""AI-Brain: Zentrales AI & Websearch Brain Control Center."""

from __future__ import annotations

import streamlit as st
from ai_brain import AIWebsearchBrain, BrainError

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ai_logs_tab import render_logs_tab
from src.ui.views.ai_models_tab import render_models_tab
from src.ui.views.ai_playground_tab import render_playground_tab
from src.ui.views.ai_secret_dialog import render_active_secret_dialog
from src.ui.views.ai_sequences_tab import render_sequences_tab
from src.ui.views.ai_task_overlay import (
    format_pipeline_feedback,
    invalidate_ai_ui_cache,
    load_cached_ui_state,
    render_instant_overlay_bridge,
    run_with_task_overlay,
)
from src.ui.views.ai_tools_tab import render_tools_tab
from src.ui.views.ai_vault_tab import render_vault_tab

AI_TAB_SEQ = "🎯 Sequencing"
AI_TAB_MODELS = "🧩 AI"
AI_TAB_TOOLS = "🔧 Tools"
AI_TAB_VAULT = "🔐 Vault"
AI_TAB_CHECK = "🧪 Check"
AI_TAB_LOGS = "📜 Logs"
AI_TAB_LABELS: tuple[str, ...] = (
    AI_TAB_SEQ,
    AI_TAB_MODELS,
    AI_TAB_TOOLS,
    AI_TAB_VAULT,
    AI_TAB_CHECK,
    AI_TAB_LOGS,
)


def _open_ai_tabs() -> tuple[object, ...]:
    """Rendert die 6 AI-Brain-Tabs mit stabiler Client-ID ohne Server-Reload beim Wechsel."""
    try:
        return tuple(st.tabs(list(AI_TAB_LABELS), key="ai_brain_tabs"))
    except TypeError:
        return tuple(st.tabs(list(AI_TAB_LABELS)))


def _flash(level: str, message: str) -> None:
    st.session_state["ai_brain_feedback"] = {"level": level, "message": message}


def _render_feedback_banner() -> None:
    feedback = st.session_state.pop("ai_brain_feedback", None)
    if not isinstance(feedback, dict):
        return
    level, msg = str(feedback.get("level") or "info"), str(feedback.get("message") or "").strip()
    if msg:
        getattr(st, level if level in {"success", "warning", "error"} else "info")(msg)


def _run_pipeline_action(
    brain: AIWebsearchBrain,
    *,
    run_discover: bool,
    run_benchmark: bool,
    run_unblock: bool,
    run_prune: bool,
    include_paid: bool,
    include_quota_tools: bool,
) -> None:
    if not any((run_discover, run_benchmark, run_unblock, run_prune)):
        _flash("warning", "Kein Ablauf-Schritt ausgewählt — bitte unter 'Ablauf-Optionen' mindestens einen Schritt aktivieren.")
        st.rerun()
    try:
        summary = run_with_task_overlay(
            "🚀 Sync & Optimize",
            "Starte kombinierten AI-Brain-Ablauf (Discover → Benchmark → Unblock → Prune)...",
            lambda cb: brain.run_full_pipeline(
                run_discover=run_discover,
                run_benchmark=run_benchmark,
                run_unblock=run_unblock,
                run_prune=run_prune,
                include_paid=include_paid,
                include_quota_tools=include_quota_tools,
                on_progress=cb,
            ),
        )
        _flash(*format_pipeline_feedback(summary))
    except (BrainError, Exception) as exc:  # noqa: BLE001
        _flash("error", f"Sync & Optimize fehlgeschlagen: {exc}")
    invalidate_ai_ui_cache()
    st.rerun()


def render_ai_view(settings: AppSettings, role: AuthRole = AuthRole.ADMIN) -> None:
    """Rendert das zentrale AI-Brain zur Modell-, Tool-, Vault- & Sequenz-Steuerung."""
    st.header("🧠 AI-Brain")
    st.caption(
        "Zentrale Schaltstelle für kostenfreie & optionale Paid-AI-Modelle, Websearch-Tools, "
        "API-Key-Vault und automatischen Benchmark-Sequencer (Fast, Standard & Deep)."
    )

    is_admin = role == AuthRole.ADMIN
    brain = create_dashboard_brain(settings)
    try:
        ui_state, secret_statuses = load_cached_ui_state(brain)
        models, tools, seqs, latest_bench = (
            ui_state.models,
            ui_state.tools,
            ui_state.sequences,
            ui_state.latest_benchmark,
        )
        with st.container(key="ai_kpi_row"):
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("AI-Modelle", f"{sum(1 for m in models if m.enabled)} / {len(models)}")
            with k2:
                st.metric("Tools", f"{sum(1 for t in tools if t.enabled)} / {len(tools)}")
            with k3:
                bench_txt = (
                    f"{latest_bench.healthy_tools_count}T / {latest_bench.healthy_models_count}M OK"
                    if latest_bench
                    else "Kein Testlauf"
                )
                st.metric("Letzter Testlauf", bench_txt, help="Letztes Benchmark-Ergebnis (gesunde Tools / Modelle).")

        run_pipeline = st.button(
            "🚀 Sync & Optimize",
            key="ai_top_btn_pipeline",
            disabled=not is_admin,
            width="stretch",
            help="Führt den kompletten Ablauf aus: Discover (stabile Modelle, ohne Previews) → Benchmark & Optimize → Unblock Tools → Prune Failing.",
        )

        with st.expander("⚙️ Ablauf-Optionen (Schritte, Kontingent & Paid-Check)", expanded=False):
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                step_discover = st.checkbox("🔎 Discover", value=True, key="ai_opt_discover", disabled=not is_admin)
            with s2:
                step_bench = st.checkbox("🚀 Benchmark", value=True, key="ai_opt_benchmark", disabled=not is_admin)
            with s3:
                step_unblock = st.checkbox("🔓 Unblock", value=True, key="ai_opt_unblock", disabled=not is_admin)
            with s4:
                step_prune = st.checkbox("🧹 Prune Failing", value=True, key="ai_opt_prune", disabled=not is_admin)
            include_quota_tools = st.checkbox(
                "🎟️ Kontingent-Tools im Benchmark mittesten (verbraucht 1 Query/Session)",
                value=False,
                key="ai_opt_include_quota_tools",
                disabled=not is_admin,
                help=(
                    "Standardmäßig AUS zum Schonen begrenzter Free-Kontingente (z. B. Browserbase 60/Mo, "
                    "Serper 2.500 Total, Tavily 1.000/Mo, Gemini Grounding). Unlimited-Tools werden immer live geprüft."
                ),
            )
            include_paid = st.checkbox(
                "💳 Paid-Only-Modelle im Benchmark mitprüfen (nur Paid-Key · minimale Token-Kosten)",
                value=False,
                key="ai_opt_include_paid",
                disabled=not is_admin,
                help=(
                    "Standardmäßig strikt $0 (nur Free-Tier-Key). Aktiviert sendet der Benchmark je 1 Mini-Prompt "
                    "(< $0,001) an Paid-Only-Modelle über den Paid-Key; Free-Tier-Modelle & Tools bleiben kostenfrei."
                ),
            )

        if is_admin:
            render_instant_overlay_bridge()

        # Overlay, Spinner & Alerts strikt außerhalb schmaler Spalten rendern
        if run_pipeline:
            _run_pipeline_action(
                brain,
                run_discover=step_discover,
                run_benchmark=step_bench,
                run_unblock=step_unblock,
                run_prune=step_prune,
                include_paid=include_paid,
                include_quota_tools=include_quota_tools,
            )

        _render_feedback_banner()
        if not is_admin:
            st.caption("Modus: Read-Only")

        tab_seq, tab_mod, tab_tool, tab_vault, tab_check, tab_logs = _open_ai_tabs()
        with tab_seq:
            render_sequences_tab(brain, is_admin=is_admin, models=models, tools=tools, sequences=seqs)
        with tab_mod:
            render_models_tab(brain, is_admin=is_admin, models=models)
        with tab_tool:
            render_tools_tab(
                brain,
                is_admin=is_admin,
                tools=tools,
                usage_quotas=ui_state.usage_quotas,
            )
        with tab_vault:
            render_vault_tab(
                brain,
                is_admin=is_admin,
                statuses=secret_statuses,
                fleet_client_record=ui_state.fleet_client,
            )
        with tab_check:
            render_playground_tab(brain, is_admin=is_admin, ui_state=ui_state)
        with tab_logs:
            render_logs_tab(brain, is_admin=is_admin, access_logs=ui_state.access_logs)
    finally:
        brain.close()

    render_active_secret_dialog(settings, is_admin=is_admin, secret_statuses=secret_statuses)
