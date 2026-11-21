"""KI-Brain Tab 6 (Logs): Zugriffsprotokoll (Wann, Wer, Wo, Wie, Budget & Erfolg)."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainAccessLogEntry, BrainError, KIWebsearchBrain

from src.ui.formatters import format_iso_timestamp
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache

_STATUS_BADGES: dict[str, str] = {
    "Success": "🟢 Success",
    "Failed": "🔴 Failed",
}
_BUDGET_BADGES: dict[str, str] = {
    "free": "🆓 Free",
    "paid": "💳 Paid",
}


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _format_engine_col(entry: BrainAccessLogEntry) -> str:
    parts: list[str] = []
    if entry.model_used:
        parts.append(entry.model_used)
    if entry.tools_used:
        parts.append(", ".join(entry.tools_used[:3]))
    return " · ".join(parts) if parts else "–"


def _render_log_kpis(logs: list[BrainAccessLogEntry]) -> None:
    total = len(logs)
    ok_count = sum(1 for row in logs if row.status == "Success")
    paid_count = sum(1 for row in logs if row.budget_mode == "paid")
    rate_txt = f"{round((ok_count / total) * 100)}%" if total else "–"
    latest_txt = (
        f"{logs[0].caller_app} · {format_iso_timestamp(logs[0].created_at)}"
        if logs
        else "Noch kein Zugriff"
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Zugriffe", str(total))
    with k2:
        st.metric("Success-Rate", rate_txt)
    with k3:
        st.metric("Paid-Calls", str(paid_count))
    with k4:
        st.metric("Letzter Zugriff", latest_txt)


def _filter_logs(
    logs: list[BrainAccessLogEntry],
    *,
    caller_filter: str,
    status_filter: str,
    action_filter: str,
) -> list[BrainAccessLogEntry]:
    return [
        entry
        for entry in logs
        if (caller_filter == "Alle" or entry.caller_app == caller_filter)
        and (status_filter == "Alle" or entry.status == status_filter)
        and (action_filter == "Alle" or entry.action == action_filter)
    ]


def render_logs_tab(
    brain: KIWebsearchBrain,
    *,
    is_admin: bool,
    access_logs: list[BrainAccessLogEntry] | None = None,
) -> None:
    """Rendert das Live-Zugriffsprotokoll aller angeschlossenen Apps und Dashboard-Tests."""
    st.subheader("📜 KI-Brain Zugriffs-Logs")
    st.caption(
        "Protokolliert automatisch jeden Zugriff über `ki_brain` (Alert-Bot, News-Bot, Research-Bot & Dashboard) "
        "inkl. Herkunft, Umgebung, Methode, Free/Paid-Modus, Latenz & Ergebnis (max. 500 Einträge)."
    )

    logs = access_logs if access_logs is not None else brain.list_access_logs(limit=200)
    _render_log_kpis(logs)

    known_callers = sorted({entry.caller_app for entry in logs if entry.caller_app})
    known_actions = sorted({entry.action for entry in logs if entry.action})

    f1, f2, f3, b_refresh, b_clear = st.columns([1.4, 1.1, 1.3, 1.0, 1.0], vertical_alignment="bottom")
    with f1:
        caller_filter = st.selectbox("Wer (App)", options=["Alle", *known_callers], key="ki_logs_filter_caller")
    with f2:
        status_filter = st.selectbox("Status", options=["Alle", "Success", "Failed"], key="ki_logs_filter_status")
    with f3:
        action_filter = st.selectbox("Aktion", options=["Alle", *known_actions], key="ki_logs_filter_action")
    with b_refresh:
        refresh_clicked = st.button("🔄 Aktualisieren", key="ki_logs_refresh_btn", width="stretch")
    with b_clear:
        clear_clicked = st.button(
            "🗑️ Leeren",
            key="ki_logs_clear_btn",
            disabled=(not is_admin or not logs),
            width="stretch",
            help="Löscht das gespeicherte Zugriffsprotokoll.",
        )

    # Aktionen & Feedback strikt außerhalb der schmalen Spalten ausführen
    if refresh_clicked:
        invalidate_ki_ui_cache()
        st.rerun()

    if clear_clicked:
        try:
            removed = brain.clear_access_logs()
            invalidate_ki_ui_cache()
            _set_feedback("info", f"Zugriffs-Logs geleert ({removed} Einträge entfernt).")
        except BrainError as exc:
            _set_feedback("error", f"Logs konnten nicht geleert werden: {exc}")
        st.rerun()

    filtered = _filter_logs(
        logs,
        caller_filter=caller_filter,
        status_filter=status_filter,
        action_filter=action_filter,
    )
    if not filtered:
        st.info(
            "Noch keine passenden Zugriffs-Logs vorhanden. Sobald Alert-Bot, News-Bot, Research-Bot oder der "
            "Tab **🧪 Check** eine Anfrage über `ki_brain` ausführen, erscheinen alle Details automatisch hier."
        )
        return

    rows = [
        {
            "Wann": format_iso_timestamp(entry.created_at),
            "Wer": entry.caller_app,
            "Wo": entry.environment,
            "Aktion": entry.action,
            "Tier": entry.tier.capitalize() if entry.tier else "–",
            "Budget": _BUDGET_BADGES.get(entry.budget_mode, entry.budget_mode),
            "Status": _STATUS_BADGES.get(entry.status, entry.status),
            "Latenz (ms)": entry.latency_ms,
            "Modell / Tools": _format_engine_col(entry),
            "Details": entry.summary or "–",
        }
        for entry in filtered
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
