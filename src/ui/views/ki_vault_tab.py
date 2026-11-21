"""KI-Brain Tab 4 (Vault): Zentraler Write-Only API-Key-Tresor (Free/Paid) & einheitlicher Flotten-Key."""

from __future__ import annotations

from typing import Any

import streamlit as st
from ki_brain import BrainError, FleetAccessStatus, KIWebsearchBrain, SecretStatusEntry

from src.ui.formatters import format_iso_timestamp
from src.ui.views.ki_secret_dialog import queue_secret_dialog_on_click
from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache

_TIER_LABELS: dict[str, str] = {
    "free": "🆓 Free-Tier",
    "paid": "💳 Paid-Tier",
    "legacy": "🔁 Legacy-Fallback",
    "tool": "🔧 Tool-Key",
}
_ISSUED_KEY_SESSION_KEY = "ki_vault_issued_fleet_key"  # noqa: S105


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_secret_table_rows(entries: list[SecretStatusEntry], *, is_admin: bool) -> None:
    for s in entries:
        tier_badge = _TIER_LABELS.get(s.key_tier, "🔑 Key")
        c_name, c_action, c_meta, c_time = st.columns([1.8, 1.35, 2.25, 1.0], vertical_alignment="center")
        with c_name:
            st.markdown(f"**`{s.env_name}`**")
            st.caption(tier_badge)
        with c_action:
            if is_admin:
                btn_lbl = "🔄 Set (••••••••)" if s.is_configured else "🔑 Missing Key"
                st.button(
                    btn_lbl,
                    key=f"ki_vault_row_btn_{s.env_name}",
                    help="Klicken zum Eintragen, Überschreiben & Testen",
                    on_click=queue_secret_dialog_on_click,
                    args=(s.env_name, None, s.model_dump()),
                    width="stretch",
                )
            else:
                st.caption("🟢 Set (••••••••)" if s.is_configured else "🔑 Missing Key")
        with c_meta:
            usage = ", ".join(s.affected_items[:3]) or "Optional"
            st.caption(f"{s.source} · {usage}")
        with c_time:
            st.caption(format_iso_timestamp(s.updated_at) if s.updated_at else "–")


def _render_ai_keys_section(ai_entries: list[SecretStatusEntry], *, is_admin: bool) -> None:
    st.subheader("🧠 KI-Modelle API-Keys (Free-Tier & Paid-Tier)")
    st.caption(
        "Strikte Kostentrennung: Standardmäßig kostenlos (`allow_paid=False`) über den **Free-Tier-Key**. "
        "Der **Paid-Tier-Key** wird erst bei explizitem Opt-In (`allow_paid=True`) oder Paid-Only-Modellen aktiv."
    )
    has_vault_gemini = any(
        s.env_name in {"GEMINI_FREE_API_KEY", "GEMINI_PAID_API_KEY", "GEMINI_API_KEY"} and "Vault" in s.source
        for s in ai_entries
    )
    has_env_gemini = any(
        s.env_name == "GEMINI_API_KEY"
        and ("ENV / Streamlit Secrets" in s.source or "überschreibt" in s.source)
        for s in ai_entries
    )
    if has_vault_gemini and has_env_gemini:
        st.info(
            "💡 Ein Gemini-Key ist bereits verschlüsselt im Vault hinterlegt. "
            "`GEMINI_API_KEY` kann jetzt direkt in **Streamlit Cloud → App Settings → Secrets** gelöscht werden "
            "(die App hat dort keinen Schreibzugriff; danach verschwindet der Legacy-Eintrag automatisch)."
        )
    _render_secret_table_rows(ai_entries, is_admin=is_admin)


def _render_tool_keys_section(tool_entries: list[SecretStatusEntry], *, is_admin: bool) -> None:
    st.subheader("🔧 Websearch & Tool API-Keys")
    st.caption("Optionale API-Keys für Websearch- & Extraktions-Tools (alle Write-Only im Vault verschlüsselt).")
    _render_secret_table_rows(tool_entries, is_admin=is_admin)


def _render_fleet_access_section(
    brain: KIWebsearchBrain,
    fleet: FleetAccessStatus,
    *,
    is_admin: bool,
) -> None:
    st.subheader("🛡️ App-Zugriff (1 gemeinsamer Key für alle 3 Apps)")
    st.caption(
        "Genau **1 kompakter `KI_BRAIN_KEY`** für **News-Bot**, **Research-Bot** und **Alert-Bot** — "
        "inkl. Brain-DB-Verbindung & sofortiger Sperr-/Rotationsfunktion."
    )

    issued = st.session_state.get(_ISSUED_KEY_SESSION_KEY)
    if is_admin and isinstance(issued, dict) and issued.get("connection_key"):
        conn_key = str(issued["connection_key"])
        has_turso = bool(issued.get("has_turso", False))
        st.warning(
            "🔑 **Neuer `KI_BRAIN_KEY` für News-Bot, Research-Bot & Alert-Bot "
            "(wird nur einmalig angezeigt — dieses EINE Secret in Streamlit Cloud Secrets & GitHub Repository Secrets der 3 Apps einfügen):**"
        )
        st.code(f'KI_BRAIN_KEY = "{conn_key}"\n', language="toml")
        if not has_turso:
            st.info(
                "ℹ️ Hinweis: Das Dashboard nutzt aktuell lokales SQLite. Für Cloud-Apps (Streamlit Cloud / GitHub Actions) "
                "bitte zuerst `BRAIN_TURSO_URL` & `BRAIN_TURSO_KEY` (oder `TURSO_URL` & `TURSO_KEY`) im Dashboard setzen "
                "und den Key danach neu erzeugen."
            )
        if st.button("✅ Key kopiert — Anzeige schließen", key="ki_vault_dismiss_token_btn"):
            st.session_state.pop(_ISSUED_KEY_SESSION_KEY, None)
            st.rerun()

    action_req: str | None = None
    c_app, c_state, c_issue, c_toggle = st.columns([1.8, 1.2, 1.4, 1.2], vertical_alignment="center")
    with c_app:
        st.markdown(f"**{fleet.display_name}**")
        ts_parts = [
            f"Erstellt: {format_iso_timestamp(fleet.created_at)}" if fleet.created_at else "Kein Key",
            f"Zuletzt genutzt: {format_iso_timestamp(fleet.last_used_at)}" if fleet.last_used_at else "",
        ]
        st.caption(" · ".join(p for p in ts_parts if p))
    with c_state:
        if fleet.enabled:
            st.caption("🟢 Active")
        elif fleet.has_key:
            st.caption("⛔ Disabled")
        else:
            st.caption("⚪ No Key")
    with c_issue:
        if is_admin:
            lbl = "🔄 Key rotieren" if fleet.has_key else "🔑 Key erzeugen"
            if st.button(lbl, key="ki_fleet_issue_btn", width="stretch"):
                action_req = "issue"
    with c_toggle:
        if is_admin and fleet.has_key:
            if fleet.enabled:
                if st.button("⛔ Sperren", key="ki_fleet_revoke_btn", width="stretch"):
                    action_req = "disable"
            else:
                if st.button("✅ Freigeben", key="ki_fleet_enable_btn", width="stretch"):
                    action_req = "enable"

    if action_req is not None:
        try:
            if action_req == "issue":
                conn_key, has_turso = brain.issue_fleet_key()
                st.session_state[_ISSUED_KEY_SESSION_KEY] = {
                    "connection_key": conn_key,
                    "has_turso": has_turso,
                }
                _set_feedback("success", "Neuer kompakter `KI_BRAIN_KEY` für alle 3 Apps erzeugt (siehe Vault).")
            elif action_req == "disable":
                brain.set_fleet_enabled(False)
                _set_feedback("info", "Flotten-Zugriff (`KI_BRAIN_KEY`) sofort gesperrt.")
            elif action_req == "enable":
                brain.set_fleet_enabled(True)
                _set_feedback("success", "Flotten-Zugriff (`KI_BRAIN_KEY`) wieder freigegeben.")
            invalidate_ki_ui_cache()
        except BrainError as exc:
            _set_feedback("error", f"Flotten-Zugriff konnte nicht aktualisiert werden: {exc}")
        st.rerun()


def _render_custom_secret_form(brain: KIWebsearchBrain, known_names: list[str]) -> None:
    with st.expander("➕ Eigenen API-Key manuell speichern, ersetzen oder löschen", expanded=False):
        secret_options = [*known_names, "➕ Eigener Key-Name..."]
        c_sel, c_custom = st.columns([2, 2], vertical_alignment="bottom")
        with c_sel:
            selected_opt = st.selectbox("Secret auswählen", options=secret_options, key="ki_secret_target_select")
        with c_custom:
            custom_name = st.text_input(
                "Eigener Secret-Name (Großbuchstaben)",
                value="",
                placeholder="z. B. PERPLEXITY_API_KEY",
                disabled=(selected_opt != "➕ Eigener Key-Name..."),
                key="ki_secret_custom_name",
            )

        target_env = custom_name.strip().upper() if selected_opt == "➕ Eigener Key-Name..." else selected_opt
        with st.form(f"ki_secret_write_only_form_{selected_opt}", clear_on_submit=True):
            new_val = st.text_input(
                f"Neuen Wert für `{target_env or 'SECRET'}` komplett eintragen (überschreibt bestehenden Wert)",
                value="",
                type="password",
                placeholder="Neuen API-Key hier einfügen (bestehender Key wird nie angezeigt)...",
            )
            b_save, b_del = st.columns([2, 1], vertical_alignment="bottom")
            with b_save:
                save_clicked = st.form_submit_button("🔒 Secret sicher speichern / ersetzen", width="stretch")
            with b_del:
                del_clicked = st.form_submit_button("🗑️ Aus Vault löschen", width="stretch")

        if save_clicked:
            try:
                brain.set_secret(target_env, new_val)
                invalidate_ki_ui_cache()
                _set_feedback("success", f"Secret `{target_env}` verschlüsselt im Vault gespeichert (Write-Only).")
            except BrainError as exc:
                _set_feedback("error", f"Secret konnte nicht gespeichert werden: {exc}")
            st.rerun()

        if del_clicked:
            try:
                removed = brain.delete_secret(target_env)
                invalidate_ki_ui_cache()
                msg = f"Secret `{target_env}` aus dem Brain-Vault entfernt." if removed else f"Kein Vault-Eintrag für `{target_env}` vorhanden."
                _set_feedback("info", msg)
            except BrainError as exc:
                _set_feedback("error", f"Löschen fehlgeschlagen: {exc}")
            st.rerun()


def render_vault_tab(
    brain: KIWebsearchBrain,
    is_admin: bool,
    *,
    statuses: list[SecretStatusEntry] | None = None,
    fleet_client_record: dict[str, Any] | None = None,
) -> None:
    """Rendert den zentralen Vault-Tab für alle API-Keys (Free/Paid) & den gemeinsamen Flotten-Key."""
    resolved = statuses if statuses is not None else brain.list_secret_statuses()
    ai_entries = [s for s in resolved if s.category == "ai"]
    tool_entries = [s for s in resolved if s.category != "ai"]
    fleet = brain.get_fleet_access_status(client_record=fleet_client_record)

    _render_ai_keys_section(ai_entries, is_admin=is_admin)
    st.divider()
    _render_tool_keys_section(tool_entries, is_admin=is_admin)
    st.divider()
    _render_fleet_access_section(brain, fleet, is_admin=is_admin)

    if is_admin:
        st.divider()
        _render_custom_secret_form(brain, [s.env_name for s in resolved])
