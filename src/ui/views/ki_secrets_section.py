"""KI-View: Write-Only Secret-Verwaltung (wie GitHub Secrets) für Missing-Key-Elemente."""

from __future__ import annotations

import streamlit as st
from ki_brain import BrainError, KIWebsearchBrain

from src.ui.formatters import format_iso_timestamp


def _set_feedback(level: str, message: str) -> None:
    st.session_state["ki_brain_feedback"] = {"level": level, "message": message}


def render_secrets_manager_expander(brain: KIWebsearchBrain, is_admin: bool) -> None:
    """Rendert den GitHub-Secrets-artigen Write-Only-Key-Tresor für Missing-Key-Elemente."""
    statuses = brain.list_secret_statuses()
    missing_count = sum(1 for s in statuses if not s.is_configured)
    badge = f" ({missing_count} Missing Key)" if missing_count else " (Alle gesetzt)"

    with st.expander(f"🔑 API-Keys & Missing-Key Tresor (Write-Only wie GitHub Secrets){badge}", expanded=False):
        st.caption(
            "Gespeicherte Keys sind **Write-Only** (AEAD-verschlüsselt in der Brain-DB): "
            "Vorhandene Werte werden aus Sicherheitsgründen niemals im Klartext angezeigt und "
            "können immer nur **komplett neu überschrieben** oder gelöscht werden."
        )
        rows = [
            {
                "Secret-Name": s.env_name,
                "Status": "🟢 Set (••••••••)" if s.is_configured else "🔑 Missing Key",
                "Quelle": s.source,
                "Aktualisiert": format_iso_timestamp(s.updated_at) if s.updated_at else "–",
                "Verwendet von": ", ".join(s.affected_items[:4]) or "Optional",
            }
            for s in statuses
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if not is_admin:
            return

        secret_options = [s.env_name for s in statuses] + ["➕ Eigener Key-Name..."]
        c_sel, c_custom = st.columns([2, 2])
        with c_sel:
            selected_opt = st.selectbox(
                "Secret auswählen",
                options=secret_options,
                key="ki_secret_target_select",
            )
        with c_custom:
            custom_name = st.text_input(
                "Eigener Secret-Name (optional, Großbuchstaben)",
                value="" if selected_opt != "➕ Eigener Key-Name..." else "",
                placeholder="z. B. PERPLEXITY_API_KEY",
                disabled=(selected_opt != "➕ Eigener Key-Name..."),
                key="ki_secret_custom_name",
            )

        target_env = (
            custom_name.strip().upper()
            if selected_opt == "➕ Eigener Key-Name..."
            else selected_opt
        )
        with st.form(f"ki_secret_write_only_form_{selected_opt}", clear_on_submit=True):
            new_val = st.text_input(
                f"Neuen Wert für `{target_env or 'SECRET'}` komplett eintragen (überschreibt bestehenden Wert)",
                value="",
                type="password",
                placeholder="Neuen API-Key hier einfügen (bestehender Key wird nie angezeigt)...",
            )
            b_save, b_del = st.columns([2, 1])
            with b_save:
                save_clicked = st.form_submit_button(
                    "🔒 Secret sicher speichern / komplett ersetzen",
                    type="primary",
                    use_container_width=True,
                )
            with b_del:
                del_clicked = st.form_submit_button(
                    "🗑️ Aus Vault löschen",
                    use_container_width=True,
                )

        if save_clicked:
            try:
                brain.set_secret(target_env, new_val)
                _set_feedback(
                    "success",
                    f"Secret `{target_env}` verschlüsselt gespeichert (Write-Only). Missing-Key-Einträge wurden freigeschaltet.",
                )
            except BrainError as exc:
                _set_feedback("error", f"Secret konnte nicht gespeichert werden: {exc}")
            st.rerun()

        if del_clicked:
            try:
                removed = brain.delete_secret(target_env)
                msg = (
                    f"Secret `{target_env}` aus dem Brain-Vault entfernt."
                    if removed
                    else f"Kein Vault-Eintrag für `{target_env}` vorhanden."
                )
                _set_feedback("info", msg)
            except BrainError as exc:
                _set_feedback("error", f"Löschen fehlgeschlagen: {exc}")
            st.rerun()
