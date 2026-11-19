"""KI-Brain Tab 5 (Check): Live-Test (Fast/Standard/Deep), Benchmark-Report, Bots-API & Snapshot."""

from __future__ import annotations

import datetime

import streamlit as st
import yaml
from ki_brain import (
    BenchmarkReport,
    BrainConfigSnapshot,
    BrainDashboardState,
    BrainError,
    BrainTier,
    KIWebsearchBrain,
)

from src.ui.formatters import format_iso_timestamp
from src.ui.views.ki_sequences_tab import format_probe_badge
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _snapshot_from_state(brain: KIWebsearchBrain, ui_state: BrainDashboardState | None) -> BrainConfigSnapshot:
    if ui_state is None:
        return brain.export_snapshot()
    return BrainConfigSnapshot(
        models=ui_state.models,
        tools=ui_state.tools,
        sequences={t.value: cfg for t, cfg in ui_state.sequences.items()},
        last_benchmark=ui_state.latest_benchmark,
        exported_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )


def _render_live_test_section(brain: KIWebsearchBrain, is_admin: bool) -> None:
    st.subheader("🧪 Brain Live-Test (Search & Ask)")
    with st.form("ki_brain_playground_form"):
        query = st.text_input("Such- oder Forschungsfrage", placeholder="z. B. Aktuelle Entwicklungen bei Open-Source-LLMs in Deutschland")
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        with c1:
            tier = st.selectbox("Ausführungs-Stufe", options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP], index=1, format_func=lambda t: t.value.capitalize())
        with c2:
            mode = st.selectbox("Modus", options=["ask (Web + KI-Synthese)", "search (nur Tool-Sequenz)"])
        c3, c4 = st.columns(2, vertical_alignment="bottom")
        with c3:
            maps_mode = st.selectbox(
                "🗺️ Google Maps Grounding",
                options=["Auto (Geo-Smart)", "Immer aktiv", "Aus"],
                index=0,
                help="Geo-Smart aktiviert Google Maps Grounding (5.000/Mo in Gemini 3+) nur bei Orts-, Adress- oder Routenbezug.",
            )
        with c4:
            allow_paid = st.checkbox(
                "💳 Kostenpflichtige Modelle/Tools bei Bedarf einbeziehen (allow_paid)",
                value=False,
                help="Standard ist strikt 0% Kosten. Erst mit Opt-In werden Paid-Only-Modelle (z. B. Deep Research / Argon / Pro) und erschöpfte Kontingente genutzt.",
            )
        submitted = st.form_submit_button("▶️ Anfrage über Brain ausführen", disabled=not is_admin, width="stretch")

    if not submitted:
        return
    if not query.strip():
        st.warning("Bitte gib eine Such- oder Forschungsfrage ein.")
        return
    use_maps: bool | None = True if maps_mode == "Immer aktiv" else (False if maps_mode == "Aus" else None)
    budget_note = "Paid Opt-In" if allow_paid else "0% Kosten"
    try:
        if mode.startswith("search"):
            with st.spinner(f"Führe Websearch ({tier.value.capitalize()} · {budget_note}) aus..."):
                res_s = brain.search(query, tier=tier, allow_paid=allow_paid, use_maps=use_maps)
            invalidate_ki_ui_cache()
            st.caption(f"Tier: `{res_s.tier.value}` · Modus: `{budget_note}` · Latenz: `{res_s.latency_ms} ms` · Tools: `{', '.join(res_s.tools_executed) or '–'}`")
            for cite in res_s.citations:
                st.markdown(f"- **[{cite.title or cite.url}]({cite.url})** (`{cite.source_tool}`) – {cite.snippet}")
            if not res_s.citations:
                st.info("Keine Treffer von den aktiven Suchtools erhalten.")
            return
        with st.spinner(f"Führe Brain Ask ({tier.value.capitalize()} · {budget_note}) aus..."):
            res_a = brain.ask(query, tier=tier, allow_paid=allow_paid, use_maps=use_maps)
        invalidate_ki_ui_cache()
        st.caption(f"Modell: `{res_a.model_used}` · Modus: `{budget_note}` · Tier: `{res_a.tier.value}` · Seiten gescraped: `{res_a.pages_fetched}` · Latenz: `{res_a.latency_ms} ms`")
        st.markdown(res_a.text)
        if res_a.citations:
            with st.expander(f"🔗 Verwendete Quellen ({len(res_a.citations)})", expanded=False):
                for cite in res_a.citations:
                    st.markdown(f"- [{cite.title or cite.url}]({cite.url}) (`{cite.source_tool}`)")
    except BrainError as exc:
        st.error(f"Fehler bei Ausführung: {exc}")


def _render_latest_benchmark_section(brain: KIWebsearchBrain, ui_state: BrainDashboardState | None = None) -> None:
    st.subheader("📊 Letzter Benchmark-Testlauf")
    report: BenchmarkReport | None = ui_state.latest_benchmark if ui_state is not None else brain.get_latest_benchmark()
    if report is None:
        st.caption("Noch kein Benchmark-Testlauf ausgeführt. Starte oben 'Sync & Optimize' zur automatischen Optimierung.")
        return
    st.caption(
        f"Lauf `{report.run_id}` · Zeit: {format_iso_timestamp(report.finished_at)} · "
        f"Gesund: {report.healthy_tools_count} Tools / {report.healthy_models_count} Modelle · Query: `{report.test_query}`"
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
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn("Score", format="%.1f", help="Qualitätsscore 0–100 (höher = besser) aus Antwortzeit, Ertrag & Modell-Tier.")},
    )


def _render_bot_api_section() -> None:
    st.subheader("🔌 Einbindung in News-Bot, Research-Bot & Alert-Bot")
    st.caption(
        "Zugriff ausschließlich für deine 3 freigegebenen Apps (`news_bot`, `research_bot`, `alert_bot`) "
        "mit einem im Tab **🔐 Vault** generierten Access-Token (`BRAIN_APP_ID` + `BRAIN_APP_TOKEN`)."
    )
    st.code('pip install "git+https://github.com/hschenke/bot-dashboard.git"', language="bash")
    st.code(
        "from ki_brain import KIWebsearchBrain\n\n"
        "# Liest BRAIN_APP_ID & BRAIN_APP_TOKEN (sowie BRAIN_TURSO_URL / BRAIN_TURSO_KEY) aus ENV/Secrets.\n"
        "# Fremde Apps oder ungültige Tokens werden strikt mit BrainAuthError abgelehnt.\n"
        "# Standardmäßig strikt 0% Kosten (Free-Tier-Key); Paid-Tier-Key erst bei allow_paid=True.\n"
        "with KIWebsearchBrain.from_env() as brain:\n"
        '    fast_hits = brain.search("Aktuelle KI-News", tier="fast")\n'
        '    local_places = brain.search("Cafés am Alexanderplatz Berlin", tier="standard")  # Geo-Smart Maps\n'
        '    free_answer = brain.ask("Vergleich lokaler LLM-Frameworks", tier="deep")\n'
        '    paid_deep = brain.ask("Tiefenanalyse Quantencomputer", tier="deep", allow_paid=True)\n',
        language="python",
    )


def _render_snapshot_controls(brain: KIWebsearchBrain, is_admin: bool, ui_state: BrainDashboardState | None = None) -> None:
    st.subheader("📦 Export, Import & Standardkatalog")
    snapshot = _snapshot_from_state(brain, ui_state)
    snap_json = snapshot.model_dump_json(indent=2)
    snap_yaml = yaml.safe_dump(snapshot.model_dump(mode="json"), allow_unicode=True, sort_keys=False, default_flow_style=False)
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    with c1:
        st.download_button("⬇️ Brain-Snapshot (JSON)", data=snap_json, file_name="ki_brain_snapshot.json", mime="application/json", width="stretch")
    with c2:
        st.download_button("⬇️ Brain-Snapshot (YAML)", data=snap_yaml, file_name="ki_brain_snapshot.yaml", mime="text/yaml", width="stretch")
    if not is_admin:
        return
    with st.expander("📥 Brain-Snapshot (JSON) importieren", expanded=False):
        raw_import = st.text_area("JSON-Snapshot einfügen", key="ki_import_json_text")
        if st.button("📥 Snapshot importieren", key="ki_import_json_btn"):
            try:
                brain.import_snapshot_json(raw_import)
                invalidate_ki_ui_cache()
                _set_feedback("success", "Brain-Snapshot erfolgreich importiert.")
            except (BrainError, ValueError) as exc:
                _set_feedback("error", f"Import fehlgeschlagen: {exc}")
            st.rerun()
    if st.button(
        "♻️ Standardkatalog (Research-Bot & Alert-Bot) zurücksetzen",
        key="ki_reset_defaults_btn",
        help="Setzt Modelle, Tools, ausgeschlossene Einträge und Tier-Sequenzen auf den Werkskatalog aus Research-Bot & Alert-Bot zurück. Gespeicherte API-Keys im Tresor und Benchmark-Logs bleiben erhalten.",
    ):
        brain.reset_defaults()
        invalidate_ki_ui_cache()
        _set_feedback("success", "Standardmodelle, Tools & Startsequenzen wiederhergestellt.")
        st.rerun()


def render_playground_tab(brain: KIWebsearchBrain, is_admin: bool, *, ui_state: BrainDashboardState | None = None) -> None:
    """Ermöglicht Live-Queries in Fast/Standard/Deep, Benchmark-Report, Bots-Einbindung & Export/Import/Reset."""
    _render_live_test_section(brain, is_admin)
    st.divider()
    _render_latest_benchmark_section(brain, ui_state=ui_state)
    st.divider()
    _render_bot_api_section()
    st.divider()
    _render_snapshot_controls(brain, is_admin, ui_state=ui_state)
