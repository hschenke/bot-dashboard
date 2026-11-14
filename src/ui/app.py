"""Haupt-Controller der Bot-Dashboard Streamlit Web-App (reine Präsentationsschicht)."""

from __future__ import annotations

import streamlit as st

from src.__version__ import __version__
from src.checks.coordinator import FleetCheckCoordinator
from src.config.settings import AppSettings, load_app_settings
from src.domain.enums import AuthRole
from src.domain.models import AppFleetSnapshot
from src.storage.repository import SqlDashboardRepository, create_dashboard_repository
from src.ui.auth_view import enforce_authentication
from src.ui.styles import inject_global_styles
from src.ui.views.db_view import render_db_view
from src.ui.views.ki_view import render_ki_view
from src.ui.views.overview_view import render_overview_view
from src.ui.views.settings_view import render_settings_view

_NAV_PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "📡", "Dashboard"),
    ("db", "🗄️", "DB-Inspektor"),
    ("ki", "🤖", "KI-View"),
    ("settings", "⚙️", "System & Historie"),
)
_NAV_VALID_KEYS: frozenset[str] = frozenset(k for k, _, _ in _NAV_PAGES)


def _ensure_initial_snapshots(
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> list[AppFleetSnapshot]:
    if "fleet_snapshots" not in st.session_state:
        with st.spinner("Checking Apps, Workers & DBs..."):
            st.session_state["fleet_snapshots"] = coordinator.check_all_apps(repository=repository)
    snapshots = list(st.session_state["fleet_snapshots"])
    pending_persist = st.session_state.pop("_persist_snapshot_indices", None)
    if pending_persist:
        for idx in pending_persist:
            if 0 <= idx < len(snapshots):
                repository.record_snapshot(snapshots[idx])
    return snapshots


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

        if st.session_state.get("active_page") not in _NAV_VALID_KEYS:
            st.session_state["active_page"] = "dashboard"

        with st.sidebar:
            st.title("🎛️ Holli Dashboard")
            st.caption(f"v{__version__} • Control Center")

            for page_key, page_icon, page_label in _NAV_PAGES:
                is_active = st.session_state["active_page"] == page_key
                if st.button(
                    page_label,
                    icon=page_icon,
                    key=f"nav_page_{page_key}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["active_page"] = page_key
                    st.rerun()

            if role == AuthRole.ADMIN and settings.app_password.strip():
                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                if st.button("Abmelden", icon="🚪", key="nav_logout_btn", use_container_width=True):
                    from src.ui.auth_view import clear_auth_cookie

                    clear_auth_cookie()
                    st.session_state["just_logged_out"] = True
                    st.rerun()

        selected_page = st.session_state["active_page"]
        if selected_page == "dashboard":
            render_overview_view(snapshots, coordinator, repository, role)
        elif selected_page == "db":
            render_db_view(snapshots)
        elif selected_page == "ki":
            render_ki_view(settings, role=role)
        else:
            render_settings_view(settings, repository)
    finally:
        repository.close()
