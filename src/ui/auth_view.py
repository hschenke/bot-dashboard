"""Authentifizierungs-View mit HMAC-Session-Token und Brute-Force-Schutz."""

from __future__ import annotations

import streamlit as st

from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.security.auth import (
    LoginRateLimiter,
    generate_admin_token,
    resolve_credential_role,
)


@st.cache_resource
def _get_rate_limiter() -> LoginRateLimiter:
    return LoginRateLimiter()


def enforce_authentication(settings: AppSettings) -> AuthRole:
    """Prüft Passwortschutz / Query-Token und rendert bei Bedarf den Login-Dialog."""
    if not settings.app_password.strip():
        return AuthRole.ADMIN

    session_token = str(st.session_state.get("auth_token", ""))
    query_token = str(st.query_params.get("token", ""))
    active_credential = session_token or query_token

    role = resolve_credential_role(active_credential, settings.app_password)
    if role in {AuthRole.ADMIN, AuthRole.READONLY}:
        return role

    limiter = _get_rate_limiter()
    locked, remaining_sec = limiter.is_locked_out()

    st.subheader("🔒 Bot-Dashboard Anmeldung")
    if locked:
        st.error(f"Zu viele Fehlversuche. Bitte warte {remaining_sec} Sekunden.")
        st.stop()

    with st.form("dashboard_login_form"):
        entered = st.text_input("Dashboard-Passwort", type="password")
        submitted = st.form_submit_button("Anmelden", use_container_width=True)

    if submitted:
        checked_role = resolve_credential_role(entered, settings.app_password)
        if checked_role == AuthRole.ADMIN:
            limiter.reset()
            st.session_state["auth_token"] = generate_admin_token(settings.app_password)
            st.rerun()
        else:
            limiter.record_failure()
            st.error("Ungültiges Passwort.")

    st.stop()
    return AuthRole.GUEST
