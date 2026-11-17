"""KI-View: Interaktiver Write-Only Missing-Key-Dialog mit Direkt-Testlauf."""

from __future__ import annotations

from typing import Any

import streamlit as st
from ki_brain import BenchmarkEntry, BrainError, KIWebsearchBrain, ProbeStatus, SecretStatusEntry

from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings

_DIALOG_STATE_KEY = "ki_active_dialog_env"
_DIALOG_META_KEY = "ki_active_dialog_meta"
_TABS_STATE_KEY = "ki_view_tabs"

_STATUS_SHORT: dict[ProbeStatus, str] = {
    ProbeStatus.OK: "🟢 OK",
    ProbeStatus.QUOTA_LIMITED: "🟠 Quota 429",
    ProbeStatus.AUTH_MISSING: "🔑 Missing Key",
    ProbeStatus.BLOCKED: "🔴 Blocked",
    ProbeStatus.ERROR: "🔴 Error",
    ProbeStatus.UNTESTED: "⚪ Untested",
}


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _clear_active_dialog() -> None:
    st.session_state.pop(_DIALOG_STATE_KEY, None)
    st.session_state.pop(_DIALOG_META_KEY, None)


def queue_secret_dialog_on_click(
    env_name: str,
    tab_label: str | None = None,
    meta_dict: dict[str, Any] | None = None,
) -> None:
    """Button-Callback (`on_click`): Öffnet das Modal ohne Extra-Rerun und hält den aktiven Tab fest."""
    st.session_state[_DIALOG_STATE_KEY] = env_name.strip().upper()
    if meta_dict is not None:
        st.session_state[_DIALOG_META_KEY] = meta_dict
    else:
        st.session_state.pop(_DIALOG_META_KEY, None)
    if tab_label:
        st.session_state[_TABS_STATE_KEY] = tab_label


def trigger_secret_dialog(
    env_name: str,
    *,
    tab_label: str | None = None,
    meta: SecretStatusEntry | None = None,
) -> None:
    """Merkt das gewählte Secret für das Write-Only-Modal und triggert ein Rerun."""
    queue_secret_dialog_on_click(env_name, tab_label=tab_label, meta_dict=meta.model_dump() if meta else None)
    st.rerun()


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


@st.dialog("🔑 API-Key sicher eintragen (Write-Only)", on_dismiss=_clear_active_dialog)
def open_secret_dialog(env_name: str, settings: AppSettings) -> None:
    """Öffnet ein GitHub-Secrets-ähnliches Write-Only-Modal inkl. sofortigem Testlauf."""
    clean_env = env_name.strip().upper()
    cur_status = _resolve_dialog_status(clean_env, settings)

    st.markdown(f"### Secret `{clean_env}`")
    if cur_status and cur_status.affected_items:
        st.caption(f"Schaltet frei: **{', '.join(cur_status.affected_items)}**")
    st.caption(
        "Sicherheitsmodus (Write-Only wie GitHub Secrets): Bestehende Keys werden "
        "niemals im Klartext angezeigt und immer komplett neu verschlüsselt überschrieben."
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
                    level, msg = "success", f"Secret `{clean_env}` verschlüsselt gespeichert (Write-Only)."
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
            _clear_active_dialog()
            _set_feedback("info", f"Secret `{clean_env}` aus dem Brain-Vault entfernt.")
            st.rerun()


def render_missing_keys_action_bar(
    brain: KIWebsearchBrain,
    *,
    is_admin: bool,
    env_filter: set[str] | None = None,
    section_key: str,
    statuses: list[SecretStatusEntry] | None = None,
    active_tab_label: str | None = None,
) -> None:
    """Zeigt direkt klickbare `Missing Key`-Buttons zum Eintragen & sofortigen Testen."""
    resolved = statuses if statuses is not None else brain.list_secret_statuses()
    missing = [
        s
        for s in resolved
        if not s.is_configured and s.affected_items and (env_filter is None or s.env_name in env_filter)
    ]
    if not missing:
        return
    if not is_admin:
        names = ", ".join(s.env_name for s in missing)
        st.caption(f"🔑 Fehlende API-Keys (Read-Only): `{names}`")
        return

    st.caption("🔑 Fehlende API-Keys — direkt auf **Missing Key** klicken zum Eintragen & Testen:")
    for idx in range(0, len(missing), 2):
        chunk = missing[idx : idx + 2]
        cols = st.columns(len(chunk), vertical_alignment="center")
        for col, entry in zip(cols, chunk, strict=True):
            with col:
                st.button(
                    f"🔑 Missing Key: {entry.env_name}",
                    key=f"ki_missing_btn_{section_key}_{entry.env_name}",
                    help=f"Klicken zum Eintragen & Testen · Verwendet von: {', '.join(entry.affected_items)}",
                    on_click=queue_secret_dialog_on_click,
                    args=(entry.env_name, active_tab_label, entry.model_dump()),
                    use_container_width=True,
                )
