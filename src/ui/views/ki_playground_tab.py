"""KI-View Tab 4: Live-Test (Fast/Standard/Deep) & Integrations-Anleitung für Bots."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, BrainTier, KIWebsearchBrain


def render_playground_tab(brain: KIWebsearchBrain, is_admin: bool) -> None:
    """Ermöglicht Live-Queries in Fast/Standard/Deep und zeigt die Bot-Integration."""
    st.subheader("🧪 Brain Live-Test (Search & Ask)")
    with st.form("ki_brain_playground_form"):
        query = st.text_input(
            "Such- oder Forschungsfrage",
            placeholder="z. B. Aktuelle Entwicklungen bei Open-Source-LLMs in Deutschland",
        )
        c1, c2 = st.columns(2)
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
