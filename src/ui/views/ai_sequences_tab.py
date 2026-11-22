"""AI-Brain Tab 1 (Sequencing): Ausführungs-Sequenzen für Fast, Standard & Deep + Editor."""

from __future__ import annotations

import streamlit as st
from ai_brain import (
    AIModelSpec,
    AIWebsearchBrain,
    BrainTier,
    ProbeStatus,
    TierSequenceConfig,
    ToolSpec,
)

from src.ui.views.ai_task_overlay import invalidate_ai_ui_cache

_STATUS_BADGES: dict[ProbeStatus, str] = {
    ProbeStatus.OK: "🟢 OK",
    ProbeStatus.QUOTA_LIMITED: "🟠 Quota 429",
    ProbeStatus.AUTH_MISSING: "🔑 Missing Key",
    ProbeStatus.BLOCKED: "🔴 Blocked",
    ProbeStatus.ERROR: "🔴 Error",
    ProbeStatus.UNTESTED: "⚪ Untested",
}

_TIER_TITLES: dict[BrainTier, tuple[str, str]] = {
    BrainTier.FAST: ("⚡ Fast", "Maximale Geschwindigkeit (< 2s) für schnelle Alerts & Checks"),
    BrainTier.STANDARD: ("⚖️ Standard", "Ausgewogene Multi-Source-Suche inkl. Seitenabruf"),
    BrainTier.DEEP: ("🔬 Deep", "Maximale Tiefe (Web + News + Wissen + Volltext + Deep Agent)"),
}


def format_probe_badge(status: ProbeStatus, *, enabled: bool = True) -> str:
    """Liefert kurzes englisches Status-Badge für Modelle, Tools & Benchmark-Einträge."""
    if not enabled:
        return "⛔ Disabled"
    return _STATUS_BADGES.get(status, f"⚪ {status.value}")


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ai_brain_feedback"] = {"level": level, "message": message}


def _render_sequence_card(
    seq: TierSequenceConfig,
    *,
    valid_model_ids: set[str] | None = None,
    valid_tool_ids: set[str] | None = None,
) -> None:
    title, subtitle = _TIER_TITLES[seq.tier]
    lock_badge = " · 🔒 Manual" if seq.manual_override else " · 🤖 Auto"
    s_ids = [t for t in seq.search_tool_sequence if valid_tool_ids is None or t in valid_tool_ids]
    e_ids = [t for t in seq.extractor_tool_sequence if valid_tool_ids is None or t in valid_tool_ids]
    m_ids = [m for m in seq.model_sequence if valid_model_ids is None or m in valid_model_ids]
    ps_ids = [t for t in (seq.paid_search_tool_sequence or seq.search_tool_sequence) if valid_tool_ids is None or t in valid_tool_ids]
    pm_ids = [m for m in (seq.paid_model_sequence or seq.model_sequence) if valid_model_ids is None or m in valid_model_ids]

    with st.expander(f"{title}{lock_badge}", expanded=False):
        st.caption(subtitle)
        search_chain = " → ".join(s_ids) or "–"
        ext_chain = " → ".join(e_ids) or "–"
        model_chain = " → ".join(m_ids) or "Heuristik"
        st.markdown(f"**Such-Sequenz (`max={seq.max_search_tools}`):** `{search_chain}`")
        st.markdown(f"**Extraktoren (`scrape={seq.enable_page_scraping}`, `pages={seq.max_pages_to_scrape}`):** `{ext_chain}`")
        st.markdown(f"**AI-Modelle (`deep_agent={seq.use_deep_agent}`):** `{model_chain}`")
        paid_diff = (
            (ps_ids and ps_ids != s_ids)
            or (pm_ids and pm_ids != m_ids)
            or (seq.paid_use_deep_agent != seq.use_deep_agent)
        )
        if paid_diff:
            paid_s = " → ".join(ps_ids or s_ids) or "–"
            paid_m = " → ".join(pm_ids or m_ids) or "Heuristik"
            st.caption(f"💳 Paid-Opt-In (`allow_paid=True`): Suche `{paid_s}` · AI `{paid_m}` (`deep_agent={seq.paid_use_deep_agent}`)")
        if seq.optimization_rationale:
            st.caption(f"Logik: {seq.optimization_rationale}")


def _render_sequence_editor(
    brain: AIWebsearchBrain,
    seqs: dict[BrainTier, TierSequenceConfig],
    *,
    models: list[AIModelSpec],
    tools: list[ToolSpec],
) -> None:
    all_tool_ids = [t.id for t in tools]
    all_model_ids = [m.id for m in models]

    with st.expander("✏️ Sequenz manuell überschreiben oder feinjustieren", expanded=False):
        tier_choice = st.selectbox(
            "Stufe wählen",
            options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP],
            format_func=lambda t: t.value.capitalize(),
            key="ai_seq_editor_tier",
        )
        cur = seqs[tier_choice]
        with st.form(f"ai_sequence_editor_form_{tier_choice.value}"):
            raw_search = st.text_input(
                "Such-Tool-Reihenfolge (kommagetrennt)",
                value=", ".join(cur.search_tool_sequence),
                help=f"Verfügbare Tools: {', '.join(all_tool_ids)}",
            )
            raw_ext = st.text_input(
                "Extraktor-Reihenfolge (kommagetrennt)",
                value=", ".join(cur.extractor_tool_sequence),
            )
            raw_models = st.text_input(
                "Modell-Reihenfolge (kommagetrennt)",
                value=", ".join(cur.model_sequence),
                help=f"Verfügbare Modelle: {', '.join(all_model_ids)}",
            )
            c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
            with c1:
                max_tools = st.number_input("Max Such-Tools", min_value=1, max_value=25, value=cur.max_search_tools)
            with c2:
                max_pages = st.number_input("Max Scrape-Seiten", min_value=0, max_value=10, value=cur.max_pages_to_scrape)
            with c3:
                manual_lock = st.checkbox("Manuelle Sperre (Auto-Testlauf überschreibt nicht)", value=cur.manual_override)
            c4, c5 = st.columns(2)
            with c4:
                scrape_on = st.checkbox("Seiten-Scraping aktiv", value=cur.enable_page_scraping)
            with c5:
                deep_agent_on = st.checkbox("Deep Agent aktiv (Standard / Free-Tier)", value=cur.use_deep_agent)
            submitted = st.form_submit_button("💾 Sequenz speichern", width="stretch")

        if submitted:
            s_list = [x.strip() for x in raw_search.split(",") if x.strip() in all_tool_ids]
            e_list = [x.strip() for x in raw_ext.split(",") if x.strip() in all_tool_ids]
            m_list = [x.strip() for x in raw_models.split(",") if x.strip() in all_model_ids]
            updated = cur.model_copy(
                update={
                    "search_tool_sequence": s_list,
                    "extractor_tool_sequence": e_list,
                    "model_sequence": m_list,
                    "max_search_tools": int(max_tools),
                    "enable_page_scraping": bool(scrape_on),
                    "max_pages_to_scrape": int(max_pages),
                    "use_deep_agent": bool(deep_agent_on),
                    "manual_override": bool(manual_lock),
                    "optimization_rationale": "Manuell über AI-Brain angepasst.",
                }
            )
            brain.update_sequence(updated)
            invalidate_ai_ui_cache()
            _set_feedback("success", f"Sequenz für Stufe '{tier_choice.value.capitalize()}' gespeichert.")
            st.rerun()


def render_sequences_tab(
    brain: AIWebsearchBrain,
    is_admin: bool,
    *,
    models: list[AIModelSpec] | None = None,
    tools: list[ToolSpec] | None = None,
    sequences: dict[BrainTier, TierSequenceConfig] | None = None,
) -> None:
    """Rendert Fast/Standard/Deep-Sequenzen untereinander (standardmäßig zugeklappt) & den Sequenz-Editor."""
    seqs = sequences if sequences is not None else brain.list_sequences()
    resolved_models = models if models is not None else brain.list_models()
    resolved_tools = tools if tools is not None else brain.list_tools()
    valid_m_ids = {m.id for m in resolved_models}
    valid_t_ids = {t.id for t in resolved_tools}

    for tier in (BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP):
        _render_sequence_card(seqs[tier], valid_model_ids=valid_m_ids, valid_tool_ids=valid_t_ids)

    if is_admin:
        _render_sequence_editor(
            brain,
            seqs,
            models=resolved_models,
            tools=resolved_tools,
        )
