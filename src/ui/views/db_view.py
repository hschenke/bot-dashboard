"""Datenbank-Inspektor-Ansicht für die 3 Bot-Datenbanken (Turso Cloud & SQLite)."""

from __future__ import annotations

import streamlit as st

from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.ui.formatters import format_health_badge, format_iso_timestamp


def _render_single_bot_db_tab(snap: AppFleetSnapshot) -> None:
    db = snap.db_check
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DB-Status", format_health_badge(db.status))
    col2.metric("Gesamt-Zeilen", f"{db.total_rows:,}".replace(",", "."))
    col3.metric("Aktive Elemente", db.active_count)
    col4.metric("Abfrage-Latenz", f"{db.latency_ms} ms")

    st.caption(
        f"**Backend:** `{db.backend_label}` • "
        f"**Letzte Aktivität:** {format_iso_timestamp(db.last_activity_at)} • "
        f"**Hinweis:** {db.message}"
    )

    if db.table_metrics:
        st.markdown("#### 📊 Tabellen-Bestand")
        table_rows = [
            {"Tabelle": m.table_name, "Anzahl Datensätze": m.row_count}
            for m in db.table_metrics
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 🕒 Jüngste Datensätze & Aktivitäten")
    if not db.recent_items:
        st.info("Keine aktuellen Datensätze gefunden oder Datenbank noch leer.")
        return

    for item in db.recent_items:
        with st.container(border=True):
            left, right = st.columns([4, 1.2])
            with left:
                st.markdown(f"**[{item.category}] {item.title}**")
                st.caption(
                    f"{item.subtitle} • Status: `{item.status}` • "
                    f"Zeit: {format_iso_timestamp(item.timestamp)}"
                )
            with right:
                safe_link = sanitize_display_url(item.url)
                if safe_link != "#":
                    st.link_button("🔗 Quelle", safe_link, use_container_width=True)


def render_db_view(snapshots: list[AppFleetSnapshot]) -> None:
    """Rendert den Datenbank-Inspektor über alle überwachten Apps."""
    st.header("🗄️ Datenbank-Inspektor (Turso LibSQL & SQLite)")
    st.caption(
        "Live-Einblick in die Tabellenbestände, Latenzen und jüngsten Einträge deiner "
        "überwachten Apps."
    )

    overview_rows = [
        {
            "App": f"{s.spec.icon} {s.spec.title}",
            "Backend": s.db_check.backend_label,
            "Status": format_health_badge(s.db_check.status),
            "Tabellen": len(s.db_check.table_metrics),
            "Gesamtzeilen": s.db_check.total_rows,
            "Aktiv": s.db_check.active_count,
            "Fehler": s.db_check.error_count,
            "Latenz (ms)": s.db_check.latency_ms,
        }
        for s in snapshots
    ]
    st.dataframe(overview_rows, use_container_width=True, hide_index=True)
    st.divider()

    if not snapshots:
        return

    tabs = st.tabs([f"{s.spec.icon} {s.spec.title} DB" for s in snapshots])
    for idx, snap in enumerate(snapshots):
        with tabs[idx]:
            _render_single_bot_db_tab(snap)
