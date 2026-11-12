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


def _ensure_initial_snapshots(
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> list[AppFleetSnapshot]:
    if "fleet_snapshots" not in st.session_state:
        with st.spinner("Initialisiere Live-Status aller Apps..."):
            st.session_state["fleet_snapshots"] = coordinator.check_all_apps(repository=repository)
    return list(st.session_state["fleet_snapshots"])


def run_streamlit_app() -> None:
    """Initialisiert Seite, Authentifizierung, Navigation und die gewählte Dashboard-Ansicht."""
    st.set_page_config(
        page_title="Holli Dashboard",
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
            st.title("🎛️ Holli Dashboard")
            st.caption(f"v{__version__} • Control Center")

            pages = [
                "📡 Dashboard",
                "🗄️ DB-Inspektor",
                "🤖 KI-View",
                "⚙️ System & Historie",
            ]
            selected_page = st.radio(
                "Navigation",
                options=pages,
                index=0,
                label_visibility="collapsed",
            )

        if selected_page in {
            "📡 Dashboard",
            "📡 Übersicht",
            "📡 Flotten-Übersicht",
            "🌐 Running & URLs",
        }:
            render_overview_view(snapshots, coordinator, repository, role)
        elif selected_page == "🗄️ DB-Inspektor":
            render_db_view(snapshots)
        elif selected_page == "🤖 KI-View":
            render_ki_view(settings, snapshots)
        else:
            render_settings_view(settings, repository)
    finally:
        repository.close()
