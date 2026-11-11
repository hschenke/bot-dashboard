"""Detailansicht für URL-Erreichbarkeit, Streamlit-Health-Check & GitHub Actions Worker."""

from __future__ import annotations

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import AuthRole
from src.domain.exceptions import ProbeError
from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.ui.formatters import (
    format_health_badge,
    format_iso_timestamp,
    format_workflow_badge,
)


def _render_url_table_section(snapshots: list[AppFleetSnapshot]) -> None:
    st.subheader("🌐 Streamlit Cloud URLs & Health-Endpunkte")
    rows = []
    for snap in snapshots:
        rows.append(
            {
                "App": f"{snap.spec.icon} {snap.spec.title}",
                "Status": format_health_badge(snap.url_check.status),
                "Wach (Awake)": "✅ Ja" if snap.url_check.is_streamlit_awake else "🌙 Nein (Schläft)",
                "/_stcore/health": "✅ OK" if snap.url_check.health_endpoint_ok else "⚠️ Nicht bestätigt",
                "HTTP-Code": snap.url_check.http_status_code or "–",
                "Latenz (ms)": snap.url_check.latency_ms,
                "Streamlit URL": snap.spec.streamlit_url,
                "GitHub Repo": snap.spec.github_repo_url,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_running_workers_section(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    role: AuthRole,
) -> None:
    st.subheader("⚙️ Running Status & GitHub Actions Hintergrund-Worker")
    st.caption(
        "Überwacht die automatischen Cloud-Worker (`daily_digest.yaml`, `run_research.yaml`, `check_alerts.yaml`) "
        "und erlaubt den direkten Sofort-Start per `workflow_dispatch`."
    )

    for snap in snapshots:
        wf = snap.workflow_check
        with st.container(border=True):
            col_info, col_meta, col_actions = st.columns([2.2, 1.8, 1.3])
            with col_info:
                st.markdown(f"#### {snap.spec.icon} {snap.spec.title} — `{wf.workflow_file}`")
                st.markdown(f"**Worker-Status:** {format_workflow_badge(wf.state)} ({wf.status_label})")
                if snap.spec.secondary_workflows:
                    sec_list = ", ".join(f"`{w}`" for w in snap.spec.secondary_workflows)
                    st.caption(f"Weitere Workflows im Repo: {sec_list}")

            with col_meta:
                st.markdown(f"**Branch:** `{wf.branch}`")
                st.markdown(f"**Gestartet:** {format_iso_timestamp(wf.started_at)}")
                st.markdown(f"**Aktualisiert:** {format_iso_timestamp(wf.updated_at)}")

            with col_actions:
                safe_app_url = sanitize_display_url(snap.spec.streamlit_url)
                safe_run_url = sanitize_display_url(wf.run_url)
                if safe_app_url != "#":
                    st.link_button("🌐 Web-App öffnen", safe_app_url, use_container_width=True)
                if safe_run_url != "#":
                    st.link_button("📋 GitHub Action Log", safe_run_url, use_container_width=True)

                if role == AuthRole.ADMIN and st.button(
                    "▶️ Worker starten",
                    key=f"trigger_wf_{snap.spec.app_id.value}",
                    use_container_width=True,
                ):
                    try:
                        msg = coordinator.github_probe.trigger_workflow_dispatch(snap.spec)
                        st.success(msg)
                    except ProbeError as exc:
                        st.error(str(exc))


def render_urls_running_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    role: AuthRole,
) -> None:
    """Rendert die Detailansicht für URL-Erreichbarkeit und laufende Hintergrund-Worker."""
    st.header("🌐 Running & URLs (Web-Apps & Cloud-Worker)")
    _render_url_table_section(snapshots)
    st.divider()
    _render_running_workers_section(snapshots, coordinator, role)
