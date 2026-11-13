"""Kompakte Status-Rückmeldungen und Wakeup-/Worker-Aktionen für die Übersicht."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.checks.coordinator import FleetCheckCoordinator
from src.domain.exceptions import ProbeError
from src.domain.models import AppFleetSnapshot
from src.storage.repository import SqlDashboardRepository


def build_wake_feedback(
    wake_results: list[tuple[str, bool, str]],
    updated: list[AppFleetSnapshot],
    *,
    booting_prefix: str = "Wakeup gesendet (HTTP 204): ",
) -> dict[str, Any]:
    """Erstellt ein einzelnes kompaktes Feedback-Objekt aus Wakeup- und Snapshot-Daten."""
    if not wake_results:
        return {
            "level": "success",
            "message": "Alle geprüften Apps sind bereits aktiv.",
            "wake_titles": [],
            "needs_refresh": False,
        }

    awake_map = {s.spec.title: s.url_check.is_streamlit_awake for s in updated}
    awake_now = [t for t, ok, _ in wake_results if ok and awake_map.get(t, False)]
    booting = [t for t, ok, _ in wake_results if ok and not awake_map.get(t, False)]
    failed = [f"{t} ({msg})" for t, ok, msg in wake_results if not ok]

    parts: list[str] = []
    if awake_now:
        parts.append(f"Aktiv & wach: {', '.join(awake_now)}")
    if booting:
        verb = "fährt noch hoch" if len(booting) == 1 else "fahren noch hoch"
        parts.append(f"{booting_prefix}{', '.join(booting)} {verb}")
    if failed:
        parts.append(f"Fehlgeschlagen: {', '.join(failed)}")

    return {
        "level": "warning" if failed else "success",
        "message": " · ".join(parts) + ".",
        "wake_titles": booting,
        "needs_refresh": bool(booting),
    }


def update_single_snapshot_state(idx: int, updated: AppFleetSnapshot) -> None:
    """Aktualisiert einen Snapshot im Session-State und bereinigt offene Wakeup-Hinweise."""
    current = list(st.session_state.get("fleet_snapshots", []))
    if idx < len(current):
        current[idx] = updated
        st.session_state["fleet_snapshots"] = current

    raw_feedback = st.session_state.get("overview_feedback")
    if not isinstance(raw_feedback, dict):
        return
    feedback = dict(raw_feedback)
    wake_titles = [str(t) for t in feedback.get("wake_titles", [])]
    if updated.spec.title in wake_titles and updated.url_check.is_streamlit_awake:
        remaining = [t for t in wake_titles if t != updated.spec.title]
        feedback["wake_titles"] = remaining
        feedback["needs_refresh"] = bool(remaining)
        if not remaining:
            feedback["message"] = f"Aktiv & wach: {updated.spec.title}."
        else:
            verb = "fährt noch hoch" if len(remaining) == 1 else "fahren noch hoch"
            feedback["message"] = (
                f"Aktiv & wach: {updated.spec.title} · "
                f"Noch im Startvorgang: {', '.join(remaining)} {verb}."
            )
        st.session_state["overview_feedback"] = feedback


def run_wake_all_action(
    snapshots: list[AppFleetSnapshot],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Weckt alle schlafenden Apps auf, aktualisiert die Tabelle und setzt kompaktes Feedback."""
    with st.spinner("Wecke Apps auf & aktualisiere Status..."):
        wake_results, updated = coordinator.wake_and_refresh_apps(snapshots, repository=repository)
    st.session_state["fleet_snapshots"] = updated
    st.session_state["overview_feedback"] = build_wake_feedback(wake_results, updated)
    st.rerun()


def run_wake_single_action(
    snap: AppFleetSnapshot,
    idx: int,
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Weckt eine einzelne App auf, aktualisiert ihren Snapshot und setzt Feedback."""
    with st.spinner(f"Wecke {snap.spec.title} auf & aktualisiere Status..."):
        ok, msg, updated = coordinator.wake_and_refresh_single_app(
            snap.spec, repository=repository
        )
    update_single_snapshot_state(idx, updated)
    st.session_state["overview_feedback"] = build_wake_feedback(
        [(snap.spec.title, ok, msg)], [updated]
    )
    st.rerun()


def run_worker_trigger_action(snap: AppFleetSnapshot, coordinator: FleetCheckCoordinator) -> None:
    """Startet den GitHub Actions Workflow und zeigt das Ergebnis im kompakten Banner."""
    try:
        msg = coordinator.github_probe.trigger_workflow_dispatch(snap.spec)
        feedback = {"level": "success", "message": f"{snap.spec.title}: {msg}"}
    except ProbeError as exc:
        feedback = {"level": "error", "message": f"{snap.spec.title}: {exc}"}
    st.session_state["overview_feedback"] = {**feedback, "wake_titles": [], "needs_refresh": False}
    st.rerun()


def _render_alert_box(level: str, message: str) -> None:
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.success(message)


def _handle_post_wake_refresh(
    feedback: dict[str, Any],
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    with st.spinner("Aktualisiere App-Status..."):
        updated = coordinator.check_all_apps(repository=repository)
    st.session_state["fleet_snapshots"] = updated

    wake_titles = [str(t) for t in feedback.get("wake_titles", [])]
    if not wake_titles:
        st.session_state.pop("overview_feedback", None)
    else:
        synthetic = [(title, True, "") for title in wake_titles]
        st.session_state["overview_feedback"] = build_wake_feedback(
            synthetic, updated, booting_prefix="Noch im Startvorgang: "
        )
    st.rerun()


def render_feedback_banner(
    coordinator: FleetCheckCoordinator,
    repository: SqlDashboardRepository,
) -> None:
    """Zeigt genau ein kompaktes Status-Banner direkt unter dem Header (optional mit Refresh-Button)."""
    feedback = st.session_state.get("overview_feedback")
    if not isinstance(feedback, dict) or not feedback.get("message"):
        return

    level = str(feedback.get("level", "success"))
    message = str(feedback["message"])
    if feedback.get("needs_refresh"):
        col_msg, col_btn = st.columns([4.2, 1.0], vertical_alignment="center")
        with col_msg:
            _render_alert_box(level, message)
        with col_btn:
            if st.button("🔄 Aktualisieren", key="refresh_after_wake", use_container_width=True):
                _handle_post_wake_refresh(feedback, coordinator, repository)
        return

    _render_alert_box(level, message)
