"""Veraltet: Zusammengeführt in overview_view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ui.views.overview_view import render_overview_view

if TYPE_CHECKING:
    from src.checks.coordinator import FleetCheckCoordinator
    from src.domain.enums import AuthRole
    from src.domain.models import AppFleetSnapshot
    from src.storage.repository import SqlDashboardRepository


def render_urls_running_view(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    role: AuthRole,
    repository: SqlDashboardRepository | None = None,
) -> None:
    """Kompatibilitäts-Weiterleitung zur neuen konsolidierten Dashboard-Übersicht."""
    if repository is None:
        from src.config.settings import load_app_settings
        from src.storage.repository import create_dashboard_repository

        with create_dashboard_repository(load_app_settings()) as repo:
            render_overview_view(snapshots, coordinator, repo, role)
    else:
        render_overview_view(snapshots, coordinator, repository, role)


__all__ = ["render_urls_running_view"]
