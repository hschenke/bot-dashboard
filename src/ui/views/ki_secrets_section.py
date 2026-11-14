"""KI-View: Write-Only Secret-Verwaltung (wie GitHub Secrets) mit klickbaren Missing-Key-Einträgen."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, KIWebsearchBrain

from src.ui.formatters import format_iso_timestamp
from src.ui.views.ki_secret_dialog import trigger_secret_dialog


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def _render_secret_rows(brain: KIWebsearchBrain, is_admin: bool) -> list[str]:
    statuses = brain.list_secret_statuses()
    clicked_env: str | None = None
    for s in statuses:
        c_name, c_action, c_used, c_time = st.columns([1.6, 1.35, 2.25, 1.1], vertical_alignment="center")
        with c_name:
            st.markdown(f"**`{s.env_name}`**")
        with c_action:
            if is_admin:
                btn_lbl = "🔄 Set (••••••••)" if s.is_configured else "🔑 Missing Key"
                btn_help = "Klicken zum Eintragen, Überschreiben & Testen"
                if st.button(btn_lbl, key=f"ki_vault_row_btn_{s.env_name}", help=btn_help, use_container_width=True):
                    clicked_env = s.env_name
            else:
                st.caption("🟢 Set (••••••••)" if s.is_configured else "🔑 Missing Key")
        with c_used:
            usage = ", ".join(s.affected_items[:3]) or "Optional"
            st.caption(f"{s.source} · {usage}")
        with c_time:
            st.caption(format_iso_timestamp(s.updated_at) if s.updated_at else "–")
    if clicked_env:
        trigger_secret_dialog(clicked_env)
    return [s.env_name for s in statuses]


def render_secrets_manager_expander(brain: KIWebsearchBrain, is_admin: bool) -> None:
    """Rendert den GitHub-Secrets-artigen Write-Only-Key-Tresor für Missing-Key-Elemente."""
    statuses = brain.list_secret_statuses()
    missing_count = sum(1 for s in statuses if not s.is_configured)
    badge = f" ({missing_count} Missing Key)" if missing_count else " (Alle gesetzt)"

    with st.expander(f"🔑 API-Keys & Missing-Key Tresor (Write-Only wie GitHub Secrets){badge}", expanded=False):
        st.caption(
            "Gespeicherte Keys sind **Write-Only** (AEAD-verschlüsselt in der Brain-DB): "
            "Vorhandene Werte werden aus Sicherheitsgründen niemals im Klartext angezeigt und "
            "können immer nur **komplett neu überschrieben** oder gelöscht werden. "
            "Klicke auf einen `🔑 Missing Key`, um den Key direkt einzutragen und zu testen."
        )
        known_names = _render_secret_rows(brain, is_admin=is_admin)

        if not is_admin:
            return

        st.divider()
        secret_options = [*known_names, "➕ Eigener Key-Name..."]
        c_sel, c_custom = st.columns([2, 2], vertical_alignment="bottom")
        with c_sel:
            selected_opt = st.selectbox("Secret auswählen (Manuell / Custom)", options=secret_options, key="ki_secret_target_select")
        with c_custom:
            custom_name = st.text_input(
                "Eigener Secret-Name (optional, Großbuchstaben)",
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
                save_clicked = st.form_submit_button("🔒 Secret sicher speichern / komplett ersetzen", use_container_width=True)
            with b_del:
                del_clicked = st.form_submit_button("🗑️ Aus Vault löschen", use_container_width=True)

        if save_clicked:
            try:
                brain.set_secret(target_env, new_val)
                _set_feedback("success", f"Secret `{target_env}` verschlüsselt gespeichert (Write-Only). Missing-Key-Einträge wurden freigeschaltet.")
            except BrainError as exc:
                _set_feedback("error", f"Secret konnte nicht gespeichert werden: {exc}")
            st.rerun()

        if del_clicked:
            try:
                removed = brain.delete_secret(target_env)
                msg = f"Secret `{target_env}` aus dem Brain-Vault entfernt." if removed else f"Kein Vault-Eintrag für `{target_env}` vorhanden."
                _set_feedback("info", msg)
            except BrainError as exc:
                _set_feedback("error", f"Löschen fehlgeschlagen: {exc}")
            st.rerun()
