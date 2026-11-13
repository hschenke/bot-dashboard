"""Automatisches Hintergrund-Polling für laufende Worker, App-Wakeups & Live-Status."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import WorkflowRunState
from src.domain.models import AppFleetSnapshot
from src.ui.views.overview_feedback import build_wake_feedback, build_worker_feedback

_ACTIVE_WORKER_STATES = {WorkflowRunState.QUEUED, WorkflowRunState.RUNNING}


def snapshot_fingerprint(snap: AppFleetSnapshot) -> tuple[object, ...]:
    """Erzeugt einen kompakten Vergleichsschlüssel für sichtbare Status-Änderungen."""
    return (
        snap.overall_status,
        snap.url_check.status,
        snap.url_check.is_streamlit_awake,
        snap.url_check.health_endpoint_ok,
        snap.workflow_check.state,
        snap.workflow_check.run_url,
        snap.workflow_check.updated_at,
        snap.db_check.status,
        snap.db_check.total_rows,
    )


def has_pending_activity(snapshots: list[AppFleetSnapshot]) -> bool:
    """Prüft, ob ein Worker (`Queued`/`Running`) oder ein App-Wakeup aktiv ist."""
    if any(s.workflow_check.state in _ACTIVE_WORKER_STATES for s in snapshots):
        return True
    feedback = st.session_state.get("overview_feedback")
    if not isinstance(feedback, dict):
        return False
    return bool(feedback.get("wake_titles") or feedback.get("worker_titles"))


def _refresh_feedback_on_change(
    old_snapshots: list[AppFleetSnapshot],
    new_snapshots: list[AppFleetSnapshot],
    changed_indices: list[int],
) -> None:
    raw = st.session_state.get("overview_feedback")
    feedback = dict(raw) if isinstance(raw, dict) else {}
    wake_titles = [str(t) for t in feedback.get("wake_titles", [])]
    worker_titles = [str(t) for t in feedback.get("worker_titles", [])]

    if wake_titles:
        st.session_state["overview_feedback"] = build_wake_feedback(
            [(t, True, "") for t in wake_titles],
            new_snapshots,
            booting_prefix="Booting: ",
        )
        return

    by_title = {s.spec.title: s for s in new_snapshots}
    for title in worker_titles:
        if title in by_title:
            snap = by_title[title]
            is_still_active = snap.workflow_check.state in _ACTIVE_WORKER_STATES
            st.session_state["overview_feedback"] = build_worker_feedback(
                snap, pending=is_still_active
            )
            return

    for idx in changed_indices:
        old_s, new_s = old_snapshots[idx], new_snapshots[idx]
        if (
            old_s.workflow_check.state in _ACTIVE_WORKER_STATES
            and new_s.workflow_check.state not in _ACTIVE_WORKER_STATES
        ):
            st.session_state["overview_feedback"] = build_worker_feedback(new_s, pending=False)
            return


def poll_live_updates(coordinator: FleetCheckCoordinator, *, fast_only: bool) -> bool:
    """Prüft laufende Worker/Wakeups (fast) oder alle Apps (steady); True bei Änderung."""
    current = list(st.session_state.get("fleet_snapshots", []))
    if not current:
        return False

    raw_fb = st.session_state.get("overview_feedback")
    wake_titles = set(raw_fb.get("wake_titles", [])) if isinstance(raw_fb, dict) else set()
    worker_titles = set(raw_fb.get("worker_titles", [])) if isinstance(raw_fb, dict) else set()

    if fast_only:
        pending_indices = [
            idx
            for idx, snap in enumerate(current)
            if snap.workflow_check.state in _ACTIVE_WORKER_STATES
            or snap.spec.title in wake_titles
            or snap.spec.title in worker_titles
        ]
        if not pending_indices:
            return False
        with ThreadPoolExecutor(max_workers=len(pending_indices)) as pool:
            fetched = list(
                pool.map(
                    lambda i: coordinator.check_single_app(current[i].spec, None),
                    pending_indices,
                )
            )
        new_snapshots = list(current)
        for idx, updated in zip(pending_indices, fetched, strict=True):
            new_snapshots[idx] = updated
    else:
        new_snapshots = coordinator.refresh_quiet_all(current)
        if len(new_snapshots) != len(current):
            return False

    changed_indices = [
        idx
        for idx, (old_s, new_s) in enumerate(zip(current, new_snapshots, strict=True))
        if snapshot_fingerprint(old_s) != snapshot_fingerprint(new_s)
    ]
    if not changed_indices:
        return False

    st.session_state["fleet_snapshots"] = new_snapshots
    st.session_state["_persist_snapshot_indices"] = changed_indices
    _refresh_feedback_on_change(current, new_snapshots, changed_indices)
    return True


def _run_poller_tick(
    coordinator: FleetCheckCoordinator,
    *,
    fast_only: bool,
    min_elapsed: float,
) -> None:
    now = time.monotonic()
    last_ts = float(st.session_state.get("_live_poll_last_ts", 0.0))
    if now - last_ts < min_elapsed:
        return
    st.session_state["_live_poll_last_ts"] = now
    if poll_live_updates(coordinator, fast_only=fast_only):
        st.rerun(scope="app")


@st.fragment(run_every=6)
def _fast_live_poller(coordinator: FleetCheckCoordinator) -> None:
    _run_poller_tick(coordinator, fast_only=True, min_elapsed=4.0)


@st.fragment(run_every=60)
def _steady_live_poller(coordinator: FleetCheckCoordinator) -> None:
    _run_poller_tick(coordinator, fast_only=False, min_elapsed=45.0)


def render_live_status_poller(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
) -> None:
    """Aktiviert automatisches Live-Polling (6s bei aktivem Worker/Wakeup, sonst 60s)."""
    st.session_state["_live_poll_last_ts"] = time.monotonic()
    if has_pending_activity(snapshots):
        _fast_live_poller(coordinator)
    elif snapshots:
        _steady_live_poller(coordinator)
