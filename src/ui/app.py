"""Haupt-Controller der Bot-Dashboard Streamlit Web-App (reine Präsentationsschicht)."""

from __future__ import annotations

import streamlit as st

from src.__version__ import __version__
from src.checks.coordinator import FleetCheckCoordinator
from src.config.settings import AppSettings, load_app_settings
from src.domain.models import AppFleetSnapshot
from src.storage.repository import SqlDashboardRepository, create_dashboard_repository
from src.ui.auth_view import enforce_authentication
from src.ui.styles import inject_global_styles
from src.ui.views.db_view import render_db_view
from src.ui.views.ki_view import render_ki_view
from src.ui.views.overview_view import render_overview_view
from src.ui.views.settings_view import render_settings_view
from src.ui.views.urls_running_view import render_urls_running_view


def _ensure_initial_snapshots(
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> list[AppFleetSnapshot]:
    if "fleet_snapshots" not in st.session_state:
        with st.spinner("Initialisiere Live-Status für News-Bot, Research-Bot und Alert-Bot..."):
            st.session_state["fleet_snapshots"] = coordinator.check_all_apps(repository=repository)
    return list(st.session_state["fleet_snapshots"])


def run_streamlit_app() -> None:
    """Initialisiert Seite, Authentifizierung, Navigation und die gewählte Dashboard-Ansicht."""
    st.set_page_config(
        page_title="Bot-Dashboard | Streamlit Fleet Monitor",
        page_icon="🎛️",
        layout="wide",
    )
    inject_global_styles()

    settings: AppSettings = load_app_settings()
    role = enforce_authentication(settings)
    repository = create_dashboard_repository(settings)
    coordinator = FleetCheckCoordinator(settings)

    try:
        snapshots = _ensure_initial_snapshots(coordinator, repository)

        with st.sidebar:
            st.title("🎛️ Bot-Dashboard")
            st.caption(f"v{__version__} • Control Center (3 Apps)")

            pages = [
                "📡 Flotten-Übersicht",
                "🌐 Running & URLs",
                "🗄️ DB-Inspektor",
                "🤖 KI-View",
                "⚙️ System & Historie",
            ]
            selected_page = st.radio("Navigation", options=pages, index=0)

            st.divider()
            if st.button("🔄 Alle Checks neu ausführen", use_container_width=True):
                with st.spinner("Aktualisiere alle 3 Apps..."):
                    st.session_state["fleet_snapshots"] = coordinator.check_all_apps(
                        repository=repository
                    )
                    st.session_state.pop("fleet_ai_analysis", None)
                st.rerun()

            st.caption(f"Dashboard-DB: **{repository.backend_name}**")
            for spec in settings.managed_apps:
                st.markdown(f"- {spec.icon} [{spec.title}]({spec.streamlit_url})")

        if selected_page == "📡 Flotten-Übersicht":
            render_overview_view(snapshots, coordinator, repository)
        elif selected_page == "🌐 Running & URLs":
            render_urls_running_view(snapshots, coordinator, role)
        elif selected_page == "🗄️ DB-Inspektor":
            render_db_view(snapshots)
        elif selected_page == "🤖 KI-View":
            render_ki_view(settings, snapshots)
        else:
            render_settings_view(settings, repository)
    finally:
        repository.close()
