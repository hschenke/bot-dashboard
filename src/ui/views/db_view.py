"""Datenbank-Inspektor-Ansicht für die 3 Bot-Datenbanken (Turso Cloud & SQLite)."""

from __future__ import annotations

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import HealthStatus
from src.domain.models import AppFleetSnapshot, DatabaseHealthSummary
from src.security.sanitization import sanitize_display_url
from src.storage.repository import SqlDashboardRepository
from src.ui.formatters import format_health_badge, format_iso_timestamp, render_markdown_table
from src.ui.views.live_poller import render_live_status_poller


def _render_db_issue_breakdown(db: DatabaseHealthSummary) -> None:
    """Schlüsselt DB-Verbindungs- und App-Daten-Fehler vollbreit unter den KPIs auf."""
    if db.status == HealthStatus.OFFLINE:
        st.error(f"**Datenbank-Verbindung Offline:** {db.message or 'Keine Verbindung möglich.'}")
        return
    if db.status != HealthStatus.DEGRADED:
        return

    lines = [
        f"**DB-Verbindung:** 🟢 Online (`{db.backend_label}`, {db.latency_ms} ms) — die Datenbank selbst funktioniert.",
        f"**Ursache für 🟡 Degraded:** {db.error_count} offene(r) Fehler in den gespeicherten App-Daten "
        "(wird beim nächsten Check automatisch auf 🟢 Online zurückgesetzt, sobald kein Fehler mehr vorliegt):",
    ]
    for issue in db.issues:
        ts_txt = format_iso_timestamp(issue.timestamp)
        entry = f"- **{issue.source}: {issue.title}** · Status: `{issue.status}` · Stand: {ts_txt}"
        if issue.detail:
            entry += f"  \n  Grund: {issue.detail}"
        lines.append(entry)
    if not db.issues and db.message:
        lines.append(f"- {db.message}")
    st.warning("\n\n".join(lines))


def _render_single_bot_db_tab(snap: AppFleetSnapshot) -> None:
    db = snap.db_check
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DB-Status", format_health_badge(db.status))
    col2.metric("Gesamt-Zeilen", f"{db.total_rows:,}".replace(",", "."))
    col3.metric("Aktive Elemente", db.active_count)
    col4.metric("Abfrage-Latenz", f"{db.latency_ms} ms")

    conn_label = "🔴 Offline" if db.status == HealthStatus.OFFLINE else "🟢 Online"
    st.caption(
        f"**Backend:** `{db.backend_label}` • **Verbindung:** {conn_label} • "
        f"**Letzte Aktivität:** {format_iso_timestamp(db.last_activity_at)} • "
        f"**Letzter Check:** {format_iso_timestamp(db.checked_at)}"
    )
    _render_db_issue_breakdown(db)

    if db.table_metrics:
        st.markdown("#### 📊 Tabellen-Bestand")
        table_rows = [
            {"Tabelle": m.table_name, "Anzahl Datensätze": m.row_count}
            for m in db.table_metrics
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

    st.markdown("#### 🕒 Jüngste Datensätze & Aktivitäten")
    if not db.recent_items:
        st.info("Keine aktuellen Datensätze gefunden oder Datenbank noch leer.")
        return

    for item in db.recent_items:
        with st.container(border=True):
            left, right = st.columns([4, 1.2])
            with left:
                is_err = item.status.lower() in {"error", "failed"}
                prefix = "⚠️ " if is_err else ""
                st.markdown(f"**{prefix}[{item.category}] {item.title}**")
                st.caption(
                    f"{item.subtitle} • Status: `{item.status}` • "
                    f"Zeit: {format_iso_timestamp(item.timestamp)}"
                )
            with right:
                safe_link = sanitize_display_url(item.url)
                if safe_link != "#":
                    st.link_button("🔗 Quelle", safe_link, width="stretch")


def render_db_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator | None = None,
    repository: SqlDashboardRepository | None = None,
) -> None:
    """Rendert den Datenbank-Inspektor über alle überwachten Apps inkl. Fehler-Aufschlüsselung."""
    st.header("🗄️ Datenbank-Inspektor (Turso LibSQL & SQLite)")
    st.caption(
        "Live-Einblick in Verbindung, Tabellenbestände, offene App-Daten-Fehler, Latenzen und jüngste Einträge."
    )

    if coordinator is not None and st.button(
        "🔄 DBs & Status jetzt prüfen",
        key="db_view_refresh_btn",
        width="stretch",
        help="Prüft alle Apps, Worker & Datenbanken neu und setzt behobene Fehler sofort auf Online zurück.",
    ):
        with st.spinner("Checking Apps, Workers & DBs..."):
            refreshed = coordinator.check_all_apps(repository=repository)
        st.session_state["fleet_snapshots"] = refreshed
        st.rerun()

    overview_rows = [
        {
            "App": f"{s.spec.icon} {s.spec.title}",
            "Backend": s.db_check.backend_label,
            "Verbindung": "🔴 Offline" if s.db_check.status == HealthStatus.OFFLINE else "🟢 Online",
            "Status": format_health_badge(s.db_check.status),
            "Tabellen": len(s.db_check.table_metrics),
            "Gesamtzeilen": s.db_check.total_rows,
            "Aktiv": s.db_check.active_count,
            "Fehler": s.db_check.error_count,
            "Latenz (ms)": s.db_check.latency_ms,
        }
        for s in snapshots
    ]
    render_markdown_table(overview_rows)
    st.divider()

    if not snapshots:
        return

    tab_labels = [f"{s.spec.icon} {s.spec.title} DB" for s in snapshots]
    try:
        tabs = st.tabs(tab_labels, key="db_view_tabs")
    except TypeError:
        tabs = st.tabs(tab_labels)
    for idx, snap in enumerate(snapshots):
        with tabs[idx]:
            _render_single_bot_db_tab(snap)

    if coordinator is not None:
        render_live_status_poller(snapshots, coordinator)
