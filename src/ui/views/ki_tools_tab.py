"""KI-View Tab 3: Dynamische Tool-Verwaltung, Custom JSON/RSS Tools & Snapshot-Backup."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, KIWebsearchBrain, SecretStatusEntry, ToolCategory, ToolSpec

from src.ui.views.ki_secret_dialog import render_missing_keys_action_bar
from src.ui.views.ki_sequences_tab import format_probe_badge
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache

_EXECUTOR_OPTIONS: tuple[str, ...] = (
    "custom_json_api",
    "custom_rss_feed",
    "searxng_search",
    "brave_search",
    "tavily_search",
    "serper_search",
    "yahoo_search",
    "duckduckgo_html",
    "duckduckgo_lite",
    "duckduckgo_instant",
    "google_news_rss",
    "wikipedia_api",
    "wikidata_api",
    "hackernews_algolia",
    "reddit_search",
    "arxiv_api",
    "direct_http_fetch",
    "jina_reader",
    "browserbase_fetch",
    "gemini_google_search",
    "gemini_url_context",
    "open_meteo",
)


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def render_tools_tab(
    brain: KIWebsearchBrain,
    is_admin: bool,
    *,
    tools: list[ToolSpec] | None = None,
    secret_statuses: list[SecretStatusEntry] | None = None,
    active_tab_label: str | None = None,
) -> None:
    """Zeigt alle Websearch- & Extraktions-Tools inkl. CRUD & HTTP-Diagnose."""
    resolved = tools if tools is not None else brain.list_tools()
    rows = [
        {
            "Name": t.name,
            "ID": t.id,
            "Kategorie": t.category.value,
            "Status": format_probe_badge(t.last_status, enabled=t.enabled),
            "Latenz (ms)": t.last_latency_ms,
            "Quality": t.quality_score,
            "Reliability": t.reliability_score,
            "Key": t.api_key_env or "Keyless",
            "Diagnose": t.last_error or "–",
        }
        for t in resolved
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "ℹ️ **SearXNG (`searxng_public`)** nutzt den öffentlichen `searx.space`-Pool inkl. HTML-Fallback "
        "(eigene Instanz-URL unten im Editor unter `searxng_public` eintragbar). "
        "Bei DuckDuckGo-Cloud-Sperre (`HTTP 202`): kostenfreien Key (`TAVILY_API_KEY`, `SERPER_API_KEY`, `BRAVE_SEARCH_API_KEY`) "
        "setzen oder auf `yahoo_search` / `searxng_public` vertrauen & DDG optional deaktivieren."
    )

    tool_envs = {t.api_key_env.strip().upper() for t in resolved if t.api_key_env.strip()}
    render_missing_keys_action_bar(
        brain,
        is_admin=is_admin,
        env_filter=tool_envs,
        section_key="tools",
        statuses=secret_statuses,
        active_tab_label=active_tab_label,
    )

    if is_admin:
        _render_tool_controls(brain, resolved)
        _render_tool_form(brain, resolved)


def _render_tool_controls(brain: KIWebsearchBrain, tools: list[ToolSpec]) -> None:
    tool_ids = [t.id for t in tools]
    if not tool_ids:
        return
    st.subheader("🔧 Tool aktivieren, bearbeiten oder löschen")
    c_sel, c_tog, c_del = st.columns([2, 1, 1], vertical_alignment="bottom")
    with c_sel:
        sel_id = st.selectbox("Tool wählen", options=tool_ids, key="ki_tool_quick_sel")
    cur = next(t for t in tools if t.id == sel_id)
    with c_tog:
        do_tog = st.button("⛔ Deaktivieren" if cur.enabled else "✅ Aktivieren", key="ki_tool_tog_btn", use_container_width=True)
    with c_del:
        do_del = st.button("🗑️ Entfernen", key="ki_tool_del_btn", use_container_width=True)
    if do_tog:
        brain.set_tool_enabled(sel_id, not cur.enabled)
        invalidate_ki_ui_cache()
        _set_feedback("success", f"Tool '{sel_id}' aktualisiert.")
        st.rerun()
    if do_del:
        brain.remove_tool(sel_id)
        invalidate_ki_ui_cache()
        _set_feedback("success", f"Tool '{sel_id}' entfernt.")
        st.rerun()


def _render_tool_form(brain: KIWebsearchBrain, tools: list[ToolSpec]) -> None:
    tool_ids = [t.id for t in tools]
    with st.expander("➕ Neues Tool manuell eintragen oder bearbeiten", expanded=False):
        mode = st.selectbox("Vorlage / Bearbeitung", options=["➕ Neues Tool", *tool_ids], key="ki_tool_edit_sel")
        existing = next((t for t in tools if t.id == mode), None)
        cur_exec = existing.executor_type if (existing and existing.executor_type in _EXECUTOR_OPTIONS) else _EXECUTOR_OPTIONS[0]

        with st.form(f"ki_tool_crud_form_{mode}"):
            c1, c2 = st.columns(2)
            with c1:
                t_id = st.text_input("Tool-ID", value=existing.id if existing else "", placeholder="z. B. meine_json_api")
                t_name = st.text_input("Anzeigename", value=existing.name if existing else "", placeholder="z. B. Tagesschau RSS / Custom API")
                t_cat = st.selectbox("Kategorie", options=list(ToolCategory), index=list(ToolCategory).index(existing.category) if existing else 0, format_func=lambda c: c.value)
                t_exec = st.selectbox("Executor-Typ", options=list(_EXECUTOR_OPTIONS), index=list(_EXECUTOR_OPTIONS).index(cur_exec))
            with c2:
                t_url = st.text_input("Endpoint-URL (SSRF-geprüft)", value=existing.endpoint_url if existing else "", placeholder="https://www.tagesschau.de/infoservices/alle-meldungen-100~rss2.xml")
                t_tmo = st.number_input("Timeout (Sek.)", min_value=1.0, max_value=60.0, value=float(existing.timeout_seconds if existing else 5.0))
                t_max = st.number_input("Max Treffer", min_value=1, max_value=20, value=int(existing.max_results if existing else 5))
                t_key_env = st.text_input("Optionaler Key-Env", value=existing.api_key_env if existing else "")
            cfg = existing.config if existing else {}
            c3, c4, c5 = st.columns(3)
            with c3:
                q_param = st.text_input("JSON Query-Param", value=str(cfg.get("query_param", "q")))
            with c4:
                items_path = st.text_input("JSON Items-Pfad", value=str(cfg.get("items_path", "results")))
            with c5:
                url_field = st.text_input("JSON URL/Title Felder", value=f"{cfg.get('url_field', 'url')},{cfg.get('title_field', 'title')}")
            c6, c7 = st.columns(2)
            with c6:
                snippet_field = st.text_input("JSON Snippet-Feld (optional)", value=str(cfg.get("snippet_field", "snippet")), placeholder="snippet / description")
            with c7:
                probe_query = st.text_input("Probe Query Rule (optional)", value=str(cfg.get("probe_query", "")), placeholder="z. B. Python oder Berlin")
            t_desc = st.text_input("Beschreibung", value=existing.description if existing else "")
            t_enabled = st.checkbox("Aktiviert", value=existing.enabled if existing else True)
            submitted = st.form_submit_button("💾 Tool speichern", use_container_width=True)

        if submitted:
            try:
                uf, tf = (url_field.split(",", 1) + ["title"])[:2]
                merged_cfg = {
                    **cfg,
                    "query_param": q_param.strip(),
                    "items_path": items_path.strip(),
                    "url_field": uf.strip() or "url",
                    "title_field": tf.strip() or "title",
                    "snippet_field": snippet_field.strip() or "snippet",
                }
                if probe_query.strip():
                    merged_cfg["probe_query"] = probe_query.strip()
                else:
                    merged_cfg.pop("probe_query", None)
                spec = ToolSpec(
                    id=t_id.strip().lower(),
                    name=t_name.strip() or t_id.strip(),
                    category=t_cat,
                    executor_type=t_exec,
                    description=t_desc.strip(),
                    endpoint_url=t_url.strip(),
                    requires_api_key=bool(t_key_env.strip()),
                    api_key_env=t_key_env.strip(),
                    origin_apps=existing.origin_apps if existing else ["manual"],
                    enabled=bool(t_enabled),
                    timeout_seconds=float(t_tmo),
                    max_results=int(t_max),
                    source="custom" if (existing is None or t_url.strip() != existing.endpoint_url) else existing.source,
                    config=merged_cfg,
                )
                saved = brain.add_or_update_tool(spec)
                invalidate_ki_ui_cache()
                _set_feedback("success", f"Tool '{saved.id}' gespeichert.")
                st.rerun()
            except (BrainError, ValueError) as exc:
                _set_feedback("error", f"Ungültiges Tool: {exc}")
                st.rerun()
