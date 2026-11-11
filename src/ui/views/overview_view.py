"""Flotten-Übersicht über alle 3 Streamlit-Apps mit Schnell-Checks und Direktlinks."""

from __future__ import annotations

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import HealthStatus
from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.storage.repository import SqlDashboardRepository
from src.ui.formatters import (
    format_health_badge,
    format_iso_timestamp,
    format_workflow_badge,
)


def _render_fleet_kpi_row(snapshots: list[AppFleetSnapshot]) -> None:
    online_apps = sum(1 for s in snapshots if s.overall_status == HealthStatus.ONLINE)
    awake_urls = sum(1 for s in snapshots if s.url_check.is_streamlit_awake)
    connected_dbs = sum(1 for s in snapshots if s.db_check.status != HealthStatus.OFFLINE)
    total_records = sum(s.db_check.total_rows for s in snapshots)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Einsatzbereite Apps", f"{online_apps} / {len(snapshots)}")
    col2.metric("Streamlit-URLs wach", f"{awake_urls} / {len(snapshots)}")
    col3.metric("Verbundene DBs", f"{connected_dbs} / {len(snapshots)}")
    col4.metric("Gesamte DB-Einträge", f"{total_records:,}".replace(",", "."))


def _render_app_card(
    snap: AppFleetSnapshot,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    idx: int,
) -> None:
    with st.container(border=True):
        st.markdown(f"### {snap.spec.icon} {snap.spec.title}")
        st.caption(snap.spec.description)
        st.markdown(f"**Gesamtstatus:** {format_health_badge(snap.overall_status)}")
        st.divider()

        st.markdown(
            f"**🌐 Web-App URL:** {format_health_badge(snap.url_check.status)}  \n"
            f"`{snap.url_check.latency_ms} ms` • {snap.url_check.message}"
        )
        st.markdown(
            f"**⚙️ GitHub Worker:** {format_workflow_badge(snap.workflow_check.state)}  \n"
            f"`{snap.workflow_check.workflow_file}` • {snap.workflow_check.status_label}"
        )
        st.markdown(
            f"**🗄️ Datenbank:** {format_health_badge(snap.db_check.status)}  \n"
            f"`{snap.db_check.total_rows} Zeilen` • {snap.db_check.backend_label}"
        )
        st.caption(f"Geprüft: {format_iso_timestamp(snap.checked_at)}")

        btn_col1, btn_col2 = st.columns(2)
        safe_url = sanitize_display_url(snap.spec.streamlit_url)
        with btn_col1:
            if safe_url != "#":
                st.link_button("🚀 App öffnen", safe_url, use_container_width=True)
        with btn_col2:
            if st.button(
                "🔄 Prüfen",
                key=f"refresh_single_{snap.spec.app_id.value}",
                use_container_width=True,
            ):
                with st.spinner(f"Prüfe {snap.spec.title}..."):
                    updated = coordinator.check_single_app(snap.spec, repository=repository)
                    current = list(st.session_state.get("fleet_snapshots", []))
                    if idx < len(current):
                        current[idx] = updated
                        st.session_state["fleet_snapshots"] = current
                st.rerun()


def render_overview_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Rendert die zentrale Kommandozentrale für News-Bot, Research-Bot und Alert-Bot."""
    head_col, action_col = st.columns([3, 1])
    with head_col:
        st.header("📡 Flotten-Übersicht (Streamlit Cloud Suite)")
        st.caption(
            "Echtzeit-Status deiner 3 gehosteten Apps: Erreichbarkeit, GitHub Actions Worker & Turso/SQLite-Datenbanken."
        )
    with action_col:
        if st.button("⚡ Alle 3 Apps jetzt prüfen", type="primary", use_container_width=True):
            with st.spinner("Prüfe URLs, GitHub Workflows & Datenbanken aller 3 Bots..."):
                st.session_state["fleet_snapshots"] = coordinator.check_all_apps(
                    repository=repository
                )
            st.rerun()

    _render_fleet_kpi_row(snapshots)
    st.write("")

    cols = st.columns(len(snapshots) or 1)
    for idx, snap in enumerate(snapshots):
        with cols[idx]:
            _render_app_card(snap, coordinator, repository, idx)
