"""Zentrale Dashboard-Übersicht: KPIs, Streamlit Cloud URLs & GitHub Actions Worker."""

from __future__ import annotations

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import AuthRole, HealthStatus
from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.storage.repository import SqlDashboardRepository
from src.ui.formatters import (
    format_health_badge,
    format_iso_timestamp,
    format_workflow_badge,
)
from src.ui.views.overview_feedback import (
    render_feedback_banner,
    run_wake_all_action,
    run_wake_single_action,
    run_worker_trigger_action,
    update_single_snapshot_state,
)


def _has_waking_candidate(snap: AppFleetSnapshot) -> bool:
    return (
        bool(snap.spec.streamlit_url.strip())
        and not snap.url_check.is_streamlit_awake
        and snap.url_check.status in {HealthStatus.ASLEEP, HealthStatus.DEGRADED}
    )


def _render_header(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
) -> None:
    has_sleeping = any(_has_waking_candidate(s) for s in snapshots)
    show_wake_all = role == AuthRole.ADMIN and has_sleeping
    cols = st.columns([2.4, 1.0, 1.1] if show_wake_all else [3.4, 1.1], vertical_alignment="center")
    with cols[0]:
        st.title("Holli Dashboard")
    with cols[1]:
        if st.button("⚡ Alle prüfen", type="primary", use_container_width=True):
            with st.spinner("Prüfe URLs, GitHub Workflows & Datenbanken..."):
                st.session_state["fleet_snapshots"] = coordinator.check_all_apps(
                    repository=repository
                )
                st.session_state.pop("overview_feedback", None)
            st.rerun()
    if show_wake_all:
        with cols[2]:
            if st.button("☀️ Alle aufwecken", use_container_width=True):
                run_wake_all_action(snapshots, coordinator, repository)


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


def _render_url_table_section(snapshots: list[AppFleetSnapshot]) -> None:
    st.subheader("🌐 Streamlit Cloud URLs & Health-Endpunkte")
    rows = [
        {
            "App": f"{snap.spec.icon} {snap.spec.title}",
            "Status": format_health_badge(snap.url_check.status),
            "Wach (Awake)": "✅ Ja" if snap.url_check.is_streamlit_awake else "🌙 Nein (Schläft)",
            "/_stcore/health": "✅ OK" if snap.url_check.health_endpoint_ok else "⚠️ Nicht bestätigt",
            "HTTP-Code": snap.url_check.http_status_code or "–",
            "Latenz (ms)": snap.url_check.latency_ms,
        }
        for snap in snapshots
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_card_actions(
    snap: AppFleetSnapshot,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
    idx: int,
) -> None:
    safe_app_url = sanitize_display_url(snap.spec.streamlit_url)
    safe_run_url = sanitize_display_url(snap.workflow_check.run_url)
    is_admin = role == AuthRole.ADMIN
    can_wake = is_admin and _has_waking_candidate(snap) and safe_app_url != "#"
    app_key = snap.spec.app_id.value

    cols = st.columns(5 if can_wake else (4 if is_admin else 3))
    with cols[0]:
        if safe_app_url != "#":
            st.link_button("🌐 Web-App öffnen", safe_app_url, use_container_width=True)
    with cols[1]:
        if safe_run_url != "#":
            st.link_button("📋 Action Log", safe_run_url, use_container_width=True)
    with cols[2]:
        if st.button("🔄 App prüfen", key=f"refresh_single_{app_key}", use_container_width=True):
            with st.spinner(f"Prüfe {snap.spec.title}..."):
                updated = coordinator.check_single_app(snap.spec, repository=repository)
            update_single_snapshot_state(idx, updated)
            st.rerun()

    if not is_admin:
        return
    with cols[3]:
        if st.button("▶️ Worker starten", key=f"trigger_wf_{app_key}", use_container_width=True):
            run_worker_trigger_action(snap, coordinator)
    if can_wake:
        with cols[4]:
            if st.button("☀️ App aufwecken", key=f"wake_app_{app_key}", use_container_width=True):
                run_wake_single_action(snap, idx, coordinator, repository)


def _render_worker_card(
    snap: AppFleetSnapshot,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
    idx: int,
) -> None:
    wf = snap.workflow_check
    safe_app_url = sanitize_display_url(snap.spec.streamlit_url)
    safe_repo_url = sanitize_display_url(snap.spec.github_repo_url)
    expander_title = (
        f"{snap.spec.icon} {snap.spec.title} — {format_health_badge(snap.overall_status)} "
        f"• Worker: {format_workflow_badge(wf.state)}"
    )

    with st.expander(expander_title, expanded=True):
        col_worker, col_urls = st.columns([1.5, 1.5])
        with col_worker:
            st.markdown(
                f"<div style='font-size: 0.83rem; color: #94a3b8; margin-bottom: 6px;'>"
                f"⚙️ Workflow: <code>{wf.workflow_file}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Worker-Status:** {format_workflow_badge(wf.state)} ({wf.status_label})")
            st.markdown(f"**Worker gestartet:** {format_iso_timestamp(wf.started_at)}")
            st.markdown(f"**Worker beendet:** {format_iso_timestamp(wf.updated_at)}")

        with col_urls:
            if safe_app_url != "#":
                st.markdown(f"**Streamlit URL:** [{snap.spec.streamlit_url}]({safe_app_url})")
            if safe_repo_url != "#":
                st.markdown(f"**GitHub Repo:** [{snap.spec.github_repo}]({safe_repo_url})")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        _render_card_actions(snap, coordinator, repository, role, idx)


def _render_running_workers_section(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
) -> None:
    st.subheader("⚙️ Running Status & GitHub Actions Hintergrund-Worker")
    st.caption(
        "Überwacht die automatischen Cloud-Worker aller Apps und erlaubt den direkten Sofort-Start per 1-Klick-Trigger."
    )
    for idx, snap in enumerate(snapshots):
        _render_worker_card(snap, coordinator, repository, role, idx)


def render_overview_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole = AuthRole.READONLY,
) -> None:
    """Rendert die zentrale Dashboard-Übersicht (KPIs, URL-Status & Cloud-Worker)."""
    _render_header(snapshots, coordinator, repository, role)
    render_feedback_banner(coordinator, repository)
    _render_fleet_kpi_row(snapshots)
    st.divider()
    _render_url_table_section(snapshots)
    st.divider()
    _render_running_workers_section(snapshots, coordinator, repository, role)
