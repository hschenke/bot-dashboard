"""KI-View Tab 4: Live-Test (Fast/Standard/Deep), Bots-API & Snapshot-Verwaltung."""

from __future__ import annotations

import datetime

import streamlit as st
import yaml
from ki_brain import (
    BrainConfigSnapshot,
    BrainDashboardState,
    BrainError,
    BrainTier,
    KIWebsearchBrain,
)

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


def _render_snapshot_controls(
    brain: KIWebsearchBrain,
    is_admin: bool,
    ui_state: BrainDashboardState | None = None,
) -> None:
    st.subheader("📦 Export, Import & Standardkatalog")
    snapshot = _snapshot_from_state(brain, ui_state)
    snap_json = snapshot.model_dump_json(indent=2)
    snap_yaml = yaml.safe_dump(snapshot.model_dump(mode="json"), allow_unicode=True, sort_keys=False, default_flow_style=False)
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    with c1:
        st.download_button(
            "⬇️ Brain-Snapshot (JSON)",
            data=snap_json,
            file_name="ki_brain_snapshot.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Brain-Snapshot (YAML)",
            data=snap_yaml,
            file_name="ki_brain_snapshot.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
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


def render_playground_tab(
    brain: KIWebsearchBrain,
    is_admin: bool,
    *,
    ui_state: BrainDashboardState | None = None,
) -> None:
    """Ermöglicht Live-Queries in Fast/Standard/Deep, Bots-Einbindung & Export/Import/Reset."""
    st.subheader("🧪 Brain Live-Test (Search & Ask)")
    with st.form("ki_brain_playground_form"):
        query = st.text_input(
            "Such- oder Forschungsfrage",
            placeholder="z. B. Aktuelle Entwicklungen bei Open-Source-LLMs in Deutschland",
        )
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        with c1:
            tier = st.selectbox(
                "Ausführungs-Stufe",
                options=[BrainTier.FAST, BrainTier.STANDARD, BrainTier.DEEP],
                index=1,
                format_func=lambda t: t.value.capitalize(),
            )
        with c2:
            mode = st.selectbox(
                "Modus",
                options=["ask (Web + KI-Synthese)", "search (nur Tool-Sequenz)"],
            )
        submitted = st.form_submit_button("▶️ Anfrage über Brain ausführen", disabled=not is_admin, use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Bitte gib eine Such- oder Forschungsfrage ein.")
        else:
            try:
                if mode.startswith("search"):
                    with st.spinner(f"Führe Websearch ({tier.value.capitalize()}) aus..."):
                        res_s = brain.search(query, tier=tier)
                    st.caption(
                        f"Tier: `{res_s.tier.value}` · Latenz: `{res_s.latency_ms} ms` · "
                        f"Tools: `{', '.join(res_s.tools_executed) or '–'}`"
                    )
                    for cite in res_s.citations:
                        st.markdown(f"- **[{cite.title or cite.url}]({cite.url})** (`{cite.source_tool}`) – {cite.snippet}")
                    if not res_s.citations:
                        st.info("Keine Treffer von den aktiven Suchtools erhalten.")
                else:
                    with st.spinner(f"Führe Brain Ask ({tier.value.capitalize()}) aus..."):
                        res_a = brain.ask(query, tier=tier)
                    st.caption(
                        f"Modell: `{res_a.model_used}` · Tier: `{res_a.tier.value}` · "
                        f"Seiten gescraped: `{res_a.pages_fetched}` · Latenz: `{res_a.latency_ms} ms`"
                    )
                    st.markdown(res_a.text)
                    if res_a.citations:
                        with st.expander(f"🔗 Verwendete Quellen ({len(res_a.citations)})", expanded=False):
                            for cite in res_a.citations:
                                st.markdown(f"- [{cite.title or cite.url}]({cite.url}) (`{cite.source_tool}`)")
            except BrainError as exc:
                st.error(f"Fehler bei Ausführung: {exc}")

    st.divider()
    st.subheader("🔌 Einbindung in Research-Bot, Alert-Bot & News-Bot")
    st.caption(
        "Installiere das eigenständige Paket `ki_brain` aus diesem Repository. "
        "Dank eigenem Top-Level-Paketnamen gibt es keine Kollision mit `src/` der Bot-Repos."
    )
    st.code(
        'pip install "git+https://github.com/hschenke/bot-dashboard.git"',
        language="bash",
    )
    st.code(
        "from ki_brain import KIWebsearchBrain\n\n"
        "# Nutzt dieselbe zentrale Turso-Cloud-DB (BRAIN_TURSO_URL / TURSO_URL)\n"
        "# und lädt automatisch optimierte Fast-, Standard- und Deep-Sequenzen.\n"
        "with KIWebsearchBrain.from_env() as brain:\n"
        '    fast_hits = brain.search("Aktuelle KI-News", tier="fast")\n'
        '    answer = brain.ask("Vergleich lokaler LLM-Frameworks", tier="deep")\n',
        language="python",
    )

    st.divider()
    _render_snapshot_controls(brain, is_admin, ui_state=ui_state)
