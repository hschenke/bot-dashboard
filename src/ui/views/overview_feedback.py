"""Kompakte Status-Rückmeldungen für die Dashboard-Übersicht."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.domain.enums import WorkflowRunState
from src.domain.models import AppFleetSnapshot
from src.security.sanitization import sanitize_display_url
from src.ui.formatters import format_workflow_badge

_ACTIVE_WORKER_STATES = {WorkflowRunState.QUEUED, WorkflowRunState.RUNNING}


def format_worker_status_link(snap: AppFleetSnapshot, *, force_queued: bool = False) -> str:
    """Erstellt ein kompaktes Status-Badge mit Direktlink zur GitHub Actions Worker-Seite."""
    wf = snap.workflow_check
    if force_queued:
        badge = format_workflow_badge(WorkflowRunState.QUEUED)
        raw_url = snap.spec.github_actions_url
    else:
        badge = format_workflow_badge(wf.state)
        raw_url = wf.run_url or snap.spec.github_actions_url
    safe_url = sanitize_display_url(raw_url)
    return f"[{badge}]({safe_url})" if safe_url != "#" else badge


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
            "message": "Alle geprüften Apps sind Online.",
            "wake_titles": [],
            "worker_titles": [],
        }

    awake_map = {s.spec.title: s.url_check.is_streamlit_awake for s in updated}
    awake_now = [t for t, ok, _ in wake_results if ok and awake_map.get(t, False)]
    booting = [t for t, ok, _ in wake_results if ok and not awake_map.get(t, False)]
    failed = [f"{t} ({msg})" for t, ok, msg in wake_results if not ok]

    parts: list[str] = []
    if awake_now:
        parts.append(f"Online & Awake: {', '.join(awake_now)}")
    if booting:
        verb = "startet (Live-Update aktiv)" if len(booting) == 1 else "starten (Live-Update aktiv)"
        parts.append(f"{booting_prefix}{', '.join(booting)} {verb}")
    if failed:
        parts.append(f"Failed: {', '.join(failed)}")

    return {
        "level": "warning" if failed else "success",
        "message": " · ".join(parts) + ".",
        "wake_titles": booting,
        "worker_titles": [],
    }


def build_worker_feedback(snap: AppFleetSnapshot, *, pending: bool) -> dict[str, Any]:
    """Erstellt ein einzelnes kompaktes Feedback-Objekt nach Worker-Start oder -Abschluss."""
    force_queued = pending and snap.workflow_check.state not in _ACTIVE_WORKER_STATES
    link = format_worker_status_link(snap, force_queued=force_queued)
    is_failure = not pending and snap.workflow_check.state == WorkflowRunState.FAILURE
    suffix = " · Live-Update aktiv." if pending else "."
    return {
        "level": "warning" if is_failure else "success",
        "message": f"{snap.spec.title}: Worker {link}{suffix}",
        "wake_titles": [],
        "worker_titles": [snap.spec.title] if pending else [],
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
        feedback["message"] = (
            f"Online & Awake: {updated.spec.title}."
            if not remaining
            else f"Online & Awake: {updated.spec.title} · Startet noch: {', '.join(remaining)}."
        )
        st.session_state["overview_feedback"] = feedback


def render_feedback_banner() -> None:
    """Zeigt genau ein kompaktes Status-Banner direkt unter dem Header."""
    feedback = st.session_state.get("overview_feedback")
    if not isinstance(feedback, dict) or not feedback.get("message"):
        return
    level, message = str(feedback.get("level", "success")), str(feedback["message"])
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.success(message)
