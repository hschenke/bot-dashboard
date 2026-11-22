"""Zentrale Dashboard-Übersicht: KPIs, Streamlit Cloud URLs & GitHub Actions Worker."""

from __future__ import annotations

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.checks.diagnostics import format_degraded_cause_short, get_issue_diagnostics
from src.domain.enums import AuthRole, HealthStatus, WorkflowRunState
from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.storage.repository import SqlDashboardRepository
from src.ui.formatters import (
    build_workflow_file_url,
    format_health_badge,
    format_iso_timestamp,
    format_worker_status,
    format_workflow_badge,
    render_markdown_table,
)
from src.ui.task_overlay import render_instant_overlay_bridge
from src.ui.views.live_poller import render_live_status_poller
from src.ui.views.overview_actions import (
    run_card_action,
    run_check_all_action,
    run_wake_all_action,
)
from src.ui.views.overview_feedback import render_feedback_banner


def _has_waking_candidate(snap: AppFleetSnapshot) -> bool:
    return (
        bool(snap.spec.streamlit_url.strip())
        and not snap.url_check.is_streamlit_awake
        and snap.url_check.status in {HealthStatus.ASLEEP, HealthStatus.DEGRADED}
    )


def _render_fleet_kpi_row(snapshots: list[AppFleetSnapshot]) -> None:
    online_apps = sum(1 for s in snapshots if s.overall_status == HealthStatus.ONLINE)
    awake_urls = sum(1 for s in snapshots if s.url_check.is_streamlit_awake)
    connected_dbs = sum(1 for s in snapshots if s.db_check.status != HealthStatus.OFFLINE)

    with st.container(key="overview_kpi_row"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Apps Online", f"{online_apps} / {len(snapshots)}")
        col2.metric("Apps Awake", f"{awake_urls} / {len(snapshots)}")
        col3.metric("DBs Online", f"{connected_dbs} / {len(snapshots)}")


def _render_action_buttons(snapshots: list[AppFleetSnapshot], role: AuthRole) -> str | None:
    has_sleeping = any(_has_waking_candidate(s) for s in snapshots)
    can_wake_all = role == AuthRole.ADMIN and has_sleeping
    clicked: str | None = None
    with st.container(key="overview_action_row"):
        cols = st.columns(2, vertical_alignment="center")
        with cols[0]:
            if st.button("⚡ Check All", key="overview_top_btn_check_all", width="stretch"):
                clicked = "check_all"
        with cols[1]:
            if st.button(
                "☀️ Wake All",
                key="overview_top_btn_wake_all",
                disabled=not can_wake_all,
                width="stretch",
                help=(
                    "Weckt alle schlafenden Streamlit-Apps automatisch auf."
                    if can_wake_all
                    else "Aktuell schlafen keine Apps (wird automatisch aktiv, sobald eine App schläft)."
                ),
            ):
                clicked = "wake_all"
    return clicked


def _format_app_table_cell(snap: AppFleetSnapshot) -> str:
    safe_app_url = sanitize_display_url(snap.spec.streamlit_url)
    app_label = f"[{snap.spec.title}]({safe_app_url})" if safe_app_url != "#" else snap.spec.title
    return f"{snap.spec.icon} {app_label}"


def _render_url_table_section(snapshots: list[AppFleetSnapshot]) -> None:
    st.subheader("🌐 Health")
    rows = [
        {
            "App": _format_app_table_cell(snap),
            "Status": format_health_badge(snap.url_check.status),
            "Awake": "✅ Yes" if snap.url_check.is_streamlit_awake else "🌙 Sleeping",
            "/_stcore/health": "✅ OK" if snap.url_check.health_endpoint_ok else "⚠️ Down",
            "HTTP": snap.url_check.http_status_code or "–",
            "Latenz (ms)": snap.url_check.latency_ms,
        }
        for snap in snapshots
    ]
    st.markdown("<span class='bot-dash-health-marker'></span>", unsafe_allow_html=True)
    render_markdown_table(rows)


def _render_card_actions(
    snap: AppFleetSnapshot,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
    idx: int,
) -> None:
    safe_app_url = sanitize_display_url(snap.spec.streamlit_url)
    safe_run_url = sanitize_display_url(snap.workflow_check.run_url or snap.spec.github_actions_url)
    is_admin = role == AuthRole.ADMIN
    can_wake = is_admin and _has_waking_candidate(snap) and safe_app_url != "#"
    app_key = snap.spec.app_id.value

    cols = st.columns(5 if can_wake else (4 if is_admin else 3))
    clicked: str | None = None
    with cols[0]:
        if safe_app_url != "#":
            st.link_button("🌐 Web-App öffnen", safe_app_url, width="stretch")
    with cols[1]:
        if safe_run_url != "#":
            st.link_button("📋 Action Log", safe_run_url, width="stretch")
    with cols[2]:
        if st.button("🔄 App prüfen", key=f"refresh_single_{app_key}", width="stretch"):
            clicked = "check_single"
    if is_admin:
        with cols[3]:
            if st.button("▶️ Worker starten", key=f"trigger_wf_{app_key}", width="stretch"):
                clicked = "trigger_worker"
        if can_wake:
            with cols[4]:
                if st.button("☀️ App aufwecken", key=f"wake_app_{app_key}", width="stretch"):
                    clicked = "wake_single"
    run_card_action(clicked, snap, idx, coordinator, repository)


def _render_card_health_breakdown(snap: AppFleetSnapshot) -> None:
    """Zeigt pro Komponente (Web-App, Worker, DB) den Status und volle Degraded/Offline-Ursachen."""
    st.caption(
        f"Web-App: {format_health_badge(snap.url_check.status)} · "
        f"Worker: {format_workflow_badge(snap.workflow_check.state)} · "
        f"DB: {format_health_badge(snap.db_check.status)}"
    )
    if snap.overall_status not in {HealthStatus.DEGRADED, HealthStatus.OFFLINE}:
        return
    issues = get_issue_diagnostics(snap)
    if not issues:
        return
    lines = ["**Aufschlüsselung (wird beim nächsten fehlerfreien Check automatisch zurückgesetzt):**"]
    for diag in issues:
        lines.append(f"- **{diag.component} ({format_health_badge(diag.status)}):** {diag.summary}")
        for detail in diag.details:
            lines.append(f"  - {detail}")
    body = "\n".join(lines)
    if snap.overall_status == HealthStatus.OFFLINE:
        st.error(body)
    else:
        st.warning(body)


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
    safe_status_url = sanitize_display_url(wf.run_url or snap.spec.github_actions_url)
    safe_workflow_url = build_workflow_file_url(snap.spec.github_repo_url, wf.workflow_file, wf.branch)
    is_active_worker = wf.state in {WorkflowRunState.QUEUED, WorkflowRunState.RUNNING}
    status_text = format_worker_status(wf.state, wf.status_label)
    wf_link = f"[`{wf.workflow_file}`]({safe_workflow_url})" if safe_workflow_url != "#" else f"`{wf.workflow_file}`"
    status_link = f"[{status_text}]({safe_status_url})" if safe_status_url != "#" else status_text
    cause_short = format_degraded_cause_short(snap) if snap.overall_status == HealthStatus.DEGRADED else ""
    health_badge = (
        f"{format_health_badge(snap.overall_status)} ({cause_short})"
        if cause_short
        else format_health_badge(snap.overall_status)
    )
    expander_title = (
        f"{snap.spec.icon} {snap.spec.title} — {health_badge} "
        f"• Worker: {format_workflow_badge(wf.state)}"
    )

    with st.expander(expander_title, expanded=False):
        col_worker, col_urls = st.columns([1.5, 1.5])
        with col_worker:
            st.caption(f"⚙️ Workflow: {wf_link}")
            st.markdown(f"**Worker Status:** {status_link}")
            st.markdown(f"**Started:** {format_iso_timestamp(wf.started_at)}")
            end_text = "⏳ Active" if is_active_worker else format_iso_timestamp(wf.updated_at)
            st.markdown(f"**Finished:** {end_text}")

        with col_urls:
            if safe_app_url != "#":
                st.markdown(f"**Streamlit URL:** [{snap.spec.streamlit_url}]({safe_app_url})")
            if safe_repo_url != "#":
                st.markdown(f"**GitHub Repo:** [{snap.spec.github_repo}]({safe_repo_url})")

        _render_card_health_breakdown(snap)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        _render_card_actions(snap, coordinator, repository, role, idx)


def _render_running_workers_section(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole,
) -> None:
    st.subheader("⚙️ Action")
    st.caption(
        "Live-Status aller Cloud-Worker inkl. Direktlink zur Status-Seite und 1-Klick-Start."
    )
    for idx, snap in enumerate(snapshots):
        _render_worker_card(snap, coordinator, repository, role, idx)


def render_overview_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
    role: AuthRole = AuthRole.READONLY,
) -> None:
    """Rendert die zentrale Dashboard-Übersicht (kompakte KPIs, Action-Buttons, URL-Status & Worker)."""
    st.title("Holli Dashboard")
    _render_fleet_kpi_row(snapshots)
    action = _render_action_buttons(snapshots, role)
    render_instant_overlay_bridge()
    if action == "check_all":
        run_check_all_action(coordinator, repository)
    elif action == "wake_all":
        run_wake_all_action(snapshots, coordinator, repository)
    render_feedback_banner()

    st.divider()
    _render_url_table_section(snapshots)
    st.divider()
    _render_running_workers_section(snapshots, coordinator, repository, role)
    render_live_status_poller(snapshots, coordinator)
