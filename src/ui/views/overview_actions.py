"""Benutzer-Aktionen für die Dashboard-Übersicht (Check All, Wake All & Einzelkarten)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import HealthStatus, WorkflowRunState
from src.domain.exceptions import ProbeError
from src.domain.models import AppFleetSnapshot
from src.storage.repository import SqlDashboardRepository
from src.ui.task_overlay import run_with_task_overlay
from src.ui.views.overview_feedback import (
    build_wake_feedback,
    build_worker_feedback,
    update_single_snapshot_state,
)

_ACTIVE_WORKER_STATES = {WorkflowRunState.QUEUED, WorkflowRunState.RUNNING}


def _run_fleet_overlay(
    title: str,
    status_msg: str,
    done_msg: str,
    action_fn: Callable[[Callable[[float, str], None]], object],
) -> object:
    def _task(cb: Callable[[float, str], None]) -> object:
        cb(0.10, status_msg)
        result = action_fn(cb)
        cb(1.0, done_msg)
        return result

    return run_with_task_overlay(title, status_msg, _task)


def run_check_all_action(
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Prüft alle Apps im Task-Overlay (gibt Hintergrund bei Abschluss oder Fehler garantiert frei)."""
    try:
        snapshots = cast(
            list[AppFleetSnapshot],
            _run_fleet_overlay(
                "⚡ Check All",
                "Checking Apps, Workers & DBs...",
                "All checks completed.",
                lambda _cb: coordinator.check_all_apps(repository=repository),
            ),
        )
        st.session_state["fleet_snapshots"] = snapshots
        st.session_state.pop("overview_feedback", None)
    except Exception as exc:  # noqa: BLE001
        st.session_state["overview_feedback"] = {
            "level": "error",
            "message": f"Check All failed: {exc}",
            "wake_titles": [],
            "worker_titles": [],
        }
    st.rerun()


def run_wake_all_action(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Weckt schlafende Apps im Task-Overlay bis alle wach sind (oder Fehler/Timeout den Hintergrund freigibt)."""
    sleeping = [
        s.spec.title
        for s in snapshots
        if not s.url_check.is_streamlit_awake
        and s.url_check.status in {HealthStatus.ASLEEP, HealthStatus.DEGRADED}
    ]
    status_msg = (
        f"Waking sleeping Apps ({', '.join(sleeping)})..." if sleeping else "Waking sleeping Apps..."
    )
    try:
        wake_results, updated = cast(
            tuple[list[tuple[str, bool, str]], list[AppFleetSnapshot]],
            _run_fleet_overlay(
                "☀️ Wake All",
                status_msg,
                "Wakeup & status refresh completed.",
                lambda cb: coordinator.wake_and_refresh_apps(
                    snapshots, repository=repository, on_progress=cb
                ),
            ),
        )
        st.session_state["fleet_snapshots"] = updated
        st.session_state["overview_feedback"] = build_wake_feedback(wake_results, updated)
    except Exception as exc:  # noqa: BLE001
        st.session_state["overview_feedback"] = {
            "level": "error",
            "message": f"Wake All failed: {exc}",
            "wake_titles": [],
            "worker_titles": [],
        }
    st.rerun()


def run_wake_single_action(
    snap: AppFleetSnapshot,
    idx: int,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Weckt eine einzelne App auf, aktualisiert ihren Snapshot und setzt Feedback."""
    with st.spinner(f"Waking {snap.spec.title}..."):
        ok, msg, updated = coordinator.wake_and_refresh_single_app(snap.spec, repository=repository)
    update_single_snapshot_state(idx, updated)
    st.session_state["overview_feedback"] = build_wake_feedback([(snap.spec.title, ok, msg)], [updated])
    st.rerun()


def run_worker_trigger_action(
    snap: AppFleetSnapshot,
    idx: int,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Startet den GitHub Actions Worker, holt den neuen Run-Status und aktiviert Live-Polling."""
    try:
        with st.spinner(f"Starting Worker for {snap.spec.title}..."):
            _, updated = coordinator.trigger_and_refresh_worker(snap, repository=repository)
        update_single_snapshot_state(idx, updated)
        wf = updated.workflow_check
        is_new_run = wf.run_url != snap.workflow_check.run_url or wf.started_at != snap.workflow_check.started_at
        pending = wf.state in _ACTIVE_WORKER_STATES or not is_new_run
        st.session_state["overview_feedback"] = build_worker_feedback(updated, pending=pending)
    except ProbeError as exc:
        st.session_state["overview_feedback"] = {
            "level": "error", "message": f"{snap.spec.title}: {exc}", "wake_titles": [], "worker_titles": [],
        }
    st.rerun()


def run_card_action(
    action: str | None,
    snap: AppFleetSnapshot,
    idx: int,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Führt Karten-Aktionen außerhalb der Button-Spalten aus, um Verschiebungen zu vermeiden."""
    if action == "check_single":
        with st.spinner(f"Checking {snap.spec.title}..."):
            updated = coordinator.check_single_app(snap.spec, repository=repository)
        update_single_snapshot_state(idx, updated)
        st.rerun()
    elif action == "trigger_worker":
        run_worker_trigger_action(snap, idx, coordinator, repository)
    elif action == "wake_single":
        run_wake_single_action(snap, idx, coordinator, repository)
