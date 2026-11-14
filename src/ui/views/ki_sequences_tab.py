"""KI-View Tab 1: Ausführungs-Sequenzen für Fast, Standard & Deep + Benchmark-Report."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainTier, KIWebsearchBrain, ProbeStatus, TierSequenceConfig

from src.ui.formatters import format_iso_timestamp
from src.ui.views.ki_secret_dialog import render_missing_keys_action_bar

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


def format_probe_badge(status: ProbeStatus) -> str:
    """Liefert kurzes englisches Status-Badge für Benchmark-Einträge."""
    return _STATUS_BADGES.get(status, f"⚪ {status.value}")


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_sequence_card(seq: TierSequenceConfig) -> None:
    title, subtitle = _TIER_TITLES[seq.tier]
    lock_badge = " · 🔒 Manual" if seq.manual_override else " · 🤖 Auto"
    with st.expander(f"{title}{lock_badge}", expanded=True):
        st.caption(subtitle)
        search_chain = " → ".join(seq.search_tool_sequence) or "–"
        ext_chain = " → ".join(seq.extractor_tool_sequence) or "–"
        model_chain = " → ".join(seq.model_sequence) or "Heuristik"
        st.markdown(f"**Such-Sequenz (`max={seq.max_search_tools}`):** `{search_chain}`")
        st.markdown(f"**Extraktoren (`scrape={seq.enable_page_scraping}`, `pages={seq.max_pages_to_scrape}`):** `{ext_chain}`")
        st.markdown(f"**KI-Modelle (`deep_agent={seq.use_deep_agent}`):** `{model_chain}`")
        if seq.optimization_rationale:
            st.caption(f"Logik: {seq.optimization_rationale}")


def _render_sequence_editor(brain: KIWebsearchBrain, seqs: dict[BrainTier, TierSequenceConfig]) -> None:
    all_tool_ids = [t.id for t in brain.list_tools()]
    all_model_ids = [m.id for m in brain.list_models()]

    with st.expander("✏️ Sequenz manuell überschreiben oder feinjustieren", expanded=False):
        tier_choice = st.selectbox(
            "Stufe wählen",
            options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP],
            format_func=lambda t: t.value.capitalize(),
            key="ki_seq_editor_tier",
        )
        cur = seqs[tier_choice]
        with st.form(f"ki_sequence_editor_form_{tier_choice.value}"):
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
                deep_agent_on = st.checkbox("Deep Agent aktiv", value=cur.use_deep_agent)
            submitted = st.form_submit_button("💾 Sequenz speichern", use_container_width=True)

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
                    "optimization_rationale": "Manuell über KI-View angepasst.",
                }
            )
            brain.update_sequence(updated)
            _set_feedback("success", f"Sequenz für Stufe '{tier_choice.value.capitalize()}' gespeichert.")
            st.rerun()


def render_sequences_tab(brain: KIWebsearchBrain, is_admin: bool) -> None:
    """Rendert Fast/Standard/Deep-Sequenzen, Editor & den letzten Benchmark-Report."""
    seqs = brain.list_sequences()
    cols = st.columns(3)
    for idx, tier in enumerate((BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP)):
        with cols[idx]:
            _render_sequence_card(seqs[tier])

    if is_admin:
        _render_sequence_editor(brain, seqs)

    st.divider()
    st.subheader("📊 Letzter Benchmark-Testlauf")
    report = brain.get_latest_benchmark()
    if report is None:
        st.caption("Noch kein Benchmark-Testlauf ausgeführt. Starte oben den Testlauf zur automatischen Optimierung.")
        render_missing_keys_action_bar(brain, is_admin=is_admin, section_key="seq")
        return

    st.caption(
        f"Lauf `{report.run_id}` · Zeit: {format_iso_timestamp(report.finished_at)} · "
        f"Gesund: {report.healthy_tools_count} Tools / {report.healthy_models_count} Modelle · "
        f"Query: `{report.test_query}`"
    )
    rows = [
        {
            "Typ": "Model" if e.item_type == "model" else "Tool",
            "Name": e.name,
            "ID": e.item_id,
            "Status": format_probe_badge(e.status),
            "Latenz (ms)": e.latency_ms,
            "Treffer": e.result_count,
            "Score": e.quality_score,
            "Details": e.details,
        }
        for e in sorted(report.entries, key=lambda x: (x.item_type, -x.quality_score, x.latency_ms))
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    render_missing_keys_action_bar(brain, is_admin=is_admin, section_key="seq_bench")
