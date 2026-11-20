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
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache
from src.ui.views.ki_view import render_ki_view
from src.ui.views.overview_view import render_overview_view

_NAV_PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "📡", "Dashboard"),
    ("db", "🗄️", "DB-Inspektor"),
    ("ki", "🧠", "KI-Brain"),
)
_NAV_VALID_KEYS: frozenset[str] = frozenset(k for k, _, _ in _NAV_PAGES)


def _select_nav_page(page_key: str) -> None:
    """Button-Callback (`on_click`): wechselt die Seite in genau einem Render-Durchlauf."""
    st.session_state["active_page"] = page_key
    if page_key != "ki":
        invalidate_ki_ui_cache()


def _ensure_initial_snapshots(
    coordinator: FleetCheckCoordinator,
    get_repository: callable,  # type: ignore[valid-type]
) -> list[AppFleetSnapshot]:
    if "fleet_snapshots" not in st.session_state:
        with st.spinner("Checking Apps, Workers & DBs..."):
            st.session_state["fleet_snapshots"] = coordinator.check_all_apps(
                repository=get_repository()
            )
    snapshots = list(st.session_state["fleet_snapshots"])
    pending_persist = st.session_state.pop("_persist_snapshot_indices", None)
    if pending_persist:
        repo = get_repository()
        for idx in pending_persist:
            if 0 <= idx < len(snapshots):
                repo.record_snapshot(snapshots[idx])
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
    coordinator = FleetCheckCoordinator(settings)
    repo_holder: dict[str, SqlDashboardRepository | None] = {"repo": None}

    def _get_repository() -> SqlDashboardRepository:
        if repo_holder["repo"] is None:
            repo_holder["repo"] = create_dashboard_repository(settings)
        return repo_holder["repo"]

    try:
        snapshots = _ensure_initial_snapshots(coordinator, _get_repository)

        if st.session_state.get("active_page") not in _NAV_VALID_KEYS:
            st.session_state["active_page"] = "dashboard"

        with st.sidebar:
            st.title("🎛️ Holli Dashboard")
            st.caption(f"v{__version__} • Control Center")

            for page_key, page_icon, page_label in _NAV_PAGES:
                is_active = st.session_state["active_page"] == page_key
                st.button(
                    page_label,
                    icon=page_icon,
                    key=f"nav_page_{page_key}",
                    type="primary" if is_active else "secondary",
                    on_click=_select_nav_page,
                    args=(page_key,),
                    width="stretch",
                )

            if role == AuthRole.ADMIN and settings.app_password.strip():
                st.divider()
                if st.button("Abmelden", icon="🚪", key="nav_logout_btn", width="stretch"):
                    from src.ui.auth_view import clear_auth_cookie

                    clear_auth_cookie(settings)
                    st.session_state["just_logged_out"] = True
                    st.rerun()

        selected_page = st.session_state["active_page"]
        if selected_page == "dashboard":
            render_overview_view(snapshots, coordinator, _get_repository(), role)
        elif selected_page == "db":
            render_db_view(snapshots)
        else:
            render_ki_view(settings, role=role)
    finally:
        if repo_holder["repo"] is not None:
            repo_holder["repo"].close()
