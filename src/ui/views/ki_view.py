"""KI-Brain: Zentrales KI & Websearch Brain Control Center."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, KIWebsearchBrain

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_models_tab import render_models_tab
from src.ui.views.ki_playground_tab import render_playground_tab
from src.ui.views.ki_secret_dialog import render_active_secret_dialog
from src.ui.views.ki_sequences_tab import render_sequences_tab
from src.ui.views.ki_task_overlay import (
    format_pipeline_feedback,
    invalidate_ki_ui_cache,
    load_cached_ui_state,
    render_instant_overlay_bridge,
    run_with_task_overlay,
)
from src.ui.views.ki_tools_tab import render_tools_tab
from src.ui.views.ki_vault_tab import render_vault_tab

KI_TAB_SEQ = "🎯 Sequencing"
KI_TAB_MODELS = "🧩 KI"
KI_TAB_TOOLS = "🔧 Tools"
KI_TAB_VAULT = "🔐 Vault"
KI_TAB_CHECK = "🧪 Check"
KI_TAB_LABELS: tuple[str, ...] = (
    KI_TAB_SEQ,
    KI_TAB_MODELS,
    KI_TAB_TOOLS,
    KI_TAB_VAULT,
    KI_TAB_CHECK,
)


def _open_ki_tabs() -> tuple[object, ...]:
    """Rendert die 5 KI-Brain-Tabs mit stabiler Client-ID ohne Server-Reload beim Wechsel."""
    try:
        return tuple(st.tabs(list(KI_TAB_LABELS), key="ki_brain_tabs"))
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


def _run_pipeline_action(
    brain: KIWebsearchBrain,
    *,
    run_discover: bool,
    run_benchmark: bool,
    run_unblock: bool,
    run_prune: bool,
    exclude_previews: bool,
    include_paid: bool,
) -> None:
    if not any((run_discover, run_benchmark, run_unblock, run_prune)):
        _flash("warning", "Kein Ablauf-Schritt ausgewählt — bitte unter 'Ablauf-Optionen' mindestens einen Schritt aktivieren.")
        st.rerun()
    try:
        summary = run_with_task_overlay(
            "🚀 Sync & Optimize",
            "Starte kombinierten KI-Brain-Ablauf (Discover → Benchmark → Unblock → Prune)...",
            lambda cb: brain.run_full_pipeline(
                run_discover=run_discover,
                run_benchmark=run_benchmark,
                run_unblock=run_unblock,
                run_prune=run_prune,
                exclude_previews=exclude_previews,
                include_paid=include_paid,
                on_progress=cb,
            ),
        )
        _flash(*format_pipeline_feedback(summary))
    except (BrainError, Exception) as exc:  # noqa: BLE001
        _flash("error", f"Sync & Optimize fehlgeschlagen: {exc}")
    invalidate_ki_ui_cache()
    st.rerun()


def render_ki_view(settings: AppSettings, role: AuthRole = AuthRole.ADMIN) -> None:
    """Rendert das zentrale KI-Brain zur Modell-, Tool-, Vault- & Sequenz-Steuerung."""
    st.header("🧠 KI-Brain")
    st.caption(
        "Zentrale Schaltstelle für kostenfreie & optionale Paid-KI-Modelle, Websearch-Tools, "
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
        with st.container(key="ki_kpi_row"):
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("KI-Modelle", f"{sum(1 for m in models if m.enabled)} / {len(models)}")
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
            key="ki_top_btn_pipeline",
            disabled=not is_admin,
            width="stretch",
            help="Führt den kompletten Ablauf aus: Discover → Benchmark & Optimize → Unblock Tools → Prune Failing.",
        )

        with st.expander("⚙️ Ablauf-Optionen (Schritte, Preview-Filter & Paid-Check)", expanded=False):
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                step_discover = st.checkbox("🔎 Discover", value=True, key="ki_opt_discover", disabled=not is_admin)
            with s2:
                step_bench = st.checkbox("🚀 Benchmark", value=True, key="ki_opt_benchmark", disabled=not is_admin)
            with s3:
                step_unblock = st.checkbox("🔓 Unblock", value=True, key="ki_opt_unblock", disabled=not is_admin)
            with s4:
                step_prune = st.checkbox("🧹 Prune Failing", value=True, key="ki_opt_prune", disabled=not is_admin)
            exclude_previews = st.checkbox(
                "Preview- & Experimental-Modelle ausschließen",
                value=True,
                key="ki_exclude_previews",
                disabled=not is_admin,
                help="Ignoriert Preview-/Experimental-Modelle beim Discover (kanonischer Deep Research Agent bleibt erhalten).",
            )
            include_paid = st.checkbox(
                "💳 Paid-Tier-Modelle im Benchmark mitprüfen (nur mit Paid-Key)",
                value=False,
                key="ki_opt_include_paid",
                disabled=not is_admin,
                help="Standardmäßig kostenlos über den Free-Tier-Key. Aktivieren, um Paid-Only-Modelle über den Paid-Tier-Key mit zu testen.",
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
                exclude_previews=exclude_previews,
                include_paid=include_paid,
            )

        _render_feedback_banner()
        if not is_admin:
            st.caption("Modus: Read-Only")

        tab_seq, tab_mod, tab_tool, tab_vault, tab_check = _open_ki_tabs()
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
                app_client_records=ui_state.app_clients,
            )
        with tab_check:
            render_playground_tab(brain, is_admin=is_admin, ui_state=ui_state)
    finally:
        brain.close()

    render_active_secret_dialog(settings, is_admin=is_admin, secret_statuses=secret_statuses)
