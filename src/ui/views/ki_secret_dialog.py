"""KI-Brain: Interaktiver Write-Only API-Key-Dialog mit Direkt-Testlauf."""

from __future__ import annotations

from typing import Any

import streamlit as st
from ki_brain import BenchmarkEntry, BrainError, ProbeStatus, SecretStatusEntry

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache

_DIALOG_STATE_KEY = "ki_active_dialog_env"
_DIALOG_META_KEY = "ki_active_dialog_meta"

_STATUS_SHORT: dict[ProbeStatus, str] = {
    ProbeStatus.OK: "🟢 OK",
    ProbeStatus.QUOTA_LIMITED: "🟠 Quota 429",
    ProbeStatus.AUTH_MISSING: "🔑 Missing Key",
    ProbeStatus.BLOCKED: "🔴 Blocked",
    ProbeStatus.ERROR: "🔴 Error",
    ProbeStatus.UNTESTED: "⚪ Untested",
}

_TIER_BADGES: dict[str, str] = {
    "free": "🆓 Free-Tier Key (0% Kosten Standard)",
    "paid": "💳 Paid-Tier Key (nur bei allow_paid=True / Paid-Modellen)",
    "legacy": "🔁 Legacy-Fallback Key (Fallback für Free & Paid)",
    "tool": "🔧 Websearch / Tool API-Key",
}


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _clear_active_dialog() -> None:
    st.session_state.pop(_DIALOG_STATE_KEY, None)
    st.session_state.pop(_DIALOG_META_KEY, None)


def queue_secret_dialog_on_click(
    env_name: str,
    _tab_label: str | None = None,
    meta_dict: dict[str, Any] | None = None,
) -> None:
    """Button-Callback (`on_click`): Öffnet das Modal ohne Extra-Rerun (React hält aktiven Tab)."""
    st.session_state[_DIALOG_STATE_KEY] = env_name.strip().upper()
    if meta_dict is not None:
        st.session_state[_DIALOG_META_KEY] = meta_dict
    else:
        st.session_state.pop(_DIALOG_META_KEY, None)


def _resolve_dialog_status(clean_env: str, settings: AppSettings) -> SecretStatusEntry | None:
    cached = st.session_state.get(_DIALOG_META_KEY)
    if isinstance(cached, dict) and str(cached.get("env_name") or "").upper() == clean_env:
        return SecretStatusEntry.model_validate(cached)
    with create_dashboard_brain(settings) as dlg_brain:
        for entry in dlg_brain.list_secret_statuses():
            if entry.env_name == clean_env:
                return entry
    return None


def render_active_secret_dialog(
    settings: AppSettings,
    *,
    is_admin: bool,
    secret_statuses: list[SecretStatusEntry] | None = None,
) -> None:
    """Öffnet einmalig pro Render-Zyklus das angeforderte Secret-Dialog-Modal."""
    if not is_admin:
        _clear_active_dialog()
        return
    env_name = str(st.session_state.get(_DIALOG_STATE_KEY) or "").strip().upper()
    if not env_name:
        return
    if secret_statuses and _DIALOG_META_KEY not in st.session_state:
        match = next((s for s in secret_statuses if s.env_name == env_name), None)
        if match is not None:
            st.session_state[_DIALOG_META_KEY] = match.model_dump()
    open_secret_dialog(env_name, settings)


def _format_probe_summary(env_name: str, entries: list[BenchmarkEntry]) -> tuple[str, str]:
    if not entries:
        return "success", f"Secret `{env_name}` verschlüsselt gespeichert. Keine aktivierten Einträge zum Testen gefunden."
    ok_count = sum(1 for e in entries if e.status == ProbeStatus.OK)
    parts = [f"`{e.item_id}`: {_STATUS_SHORT.get(e.status, e.status.value)} ({e.latency_ms} ms)" for e in entries[:5]]
    level = "success" if ok_count == len(entries) else ("warning" if ok_count > 0 else "error")
    return level, f"Secret `{env_name}` gespeichert & getestet ({ok_count}/{len(entries)} OK): {' · '.join(parts)}"


@st.dialog("🔐 API-Key im Vault eintragen (Write-Only)", on_dismiss=_clear_active_dialog)
def open_secret_dialog(env_name: str, settings: AppSettings) -> None:
    """Öffnet ein GitHub-Secrets-ähnliches Write-Only-Modal inkl. sofortigem Testlauf."""
    clean_env = env_name.strip().upper()
    cur_status = _resolve_dialog_status(clean_env, settings)

    st.markdown(f"### Secret `{clean_env}`")
    if cur_status:
        st.caption(f"Typ: **{_TIER_BADGES.get(cur_status.key_tier, '🔑 API-Key')}**")
        if cur_status.affected_items:
            st.caption(f"Schaltet frei: **{', '.join(cur_status.affected_items[:6])}**")
        if cur_status.is_configured:
            st.caption(f"Aktive Quelle: **{cur_status.source}**")
    st.caption(
        "Write-Only Sicherheitsmodus: Bestehende Keys werden niemals im Klartext angezeigt "
        "und immer komplett neu verschlüsselt im Vault überschrieben."
    )

    secret_val = st.text_input(
        f"Neuen Wert für `{clean_env}` komplett eintragen",
        value="",
        type="password",
        placeholder="Vollständigen API-Key hier einfügen...",
        key=f"ki_dlg_secret_input_{clean_env}",
    )

    c_test, c_save = st.columns(2, vertical_alignment="bottom")
    with c_test:
        save_and_test = st.button("🔒 Speichern & testen", key=f"ki_dlg_test_btn_{clean_env}", use_container_width=True)
    with c_save:
        save_only = st.button("🔒 Nur speichern", key=f"ki_dlg_save_btn_{clean_env}", use_container_width=True)

    # Spinner & Validierungsfehler strikt außerhalb schmaler Spalten rendern
    if save_and_test or save_only:
        if not secret_val.strip():
            st.error("Bitte gib den vollständigen API-Key ein.")
            return
        try:
            with create_dashboard_brain(settings) as dlg_brain:
                dlg_brain.set_secret(clean_env, secret_val)
                if save_and_test:
                    with st.spinner(f"Teste freigeschaltete Modelle/Tools für `{clean_env}`..."):
                        entries = dlg_brain.probe_items_for_secret(clean_env)
                    level, msg = _format_probe_summary(clean_env, entries)
                else:
                    level, msg = "success", f"Secret `{clean_env}` verschlüsselt im Vault gespeichert (Write-Only)."
            invalidate_ki_ui_cache()
            _set_feedback(level, msg)
            _clear_active_dialog()
            st.rerun()
        except BrainError as exc:
            st.error(f"Fehler: {exc}")
            return

    if cur_status and cur_status.is_configured and "Vault" in cur_status.source:
        st.divider()
        if st.button("🗑️ Aus Vault löschen", key=f"ki_dlg_del_btn_{clean_env}", use_container_width=True):
            with create_dashboard_brain(settings) as dlg_brain:
                dlg_brain.delete_secret(clean_env)
            invalidate_ki_ui_cache()
            _clear_active_dialog()
            _set_feedback("info", f"Secret `{clean_env}` aus dem Brain-Vault entfernt.")
            st.rerun()
