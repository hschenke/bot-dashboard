"""Authentifizierungs-View mit 30-Tage-Browser-Persistenz, Session-Token und Brute-Force-Schutz."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import unquote

import streamlit as st

from src.config.constants import AUTH_COOKIE_NAME
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.security.auth import (
    LoginRateLimiter,
    generate_admin_token,
    resolve_credential_role,
    verify_admin_token,
)
from src.ui.auth_storage import is_valid_admin_token_format, mount_auth_storage_bridge

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_rate_limiter() -> LoginRateLimiter:
    return LoginRateLimiter()


def _get_cookie_val() -> str:
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            val = st.context.cookies.get(AUTH_COOKIE_NAME)
            if val:
                return unquote(str(val)).strip()
    except Exception as exc:
        logger.debug("Fehler beim Lesen aus st.context.cookies: %s", exc)
    return ""


def _schedule_storage_sync(action: str) -> None:
    st.session_state["_pending_storage_action"] = action
    st.session_state["_storage_sync_nonce"] = secrets.token_hex(4)


def clear_auth_cookie() -> None:
    """Entfernt das Authentifizierungs-Token aus Session, URL und Browser-Speicher."""
    st.session_state.pop("auth_token", None)
    st.session_state.pop("remember_me", None)
    st.session_state.pop("_auth_restore_nonce", None)
    st.session_state["_ignore_context_cookie"] = True
    if "token" in st.query_params:
        del st.query_params["token"]
    _schedule_storage_sync("clear")


def _sync_authenticated_storage(token: str) -> None:
    pending = st.session_state.pop("_pending_storage_action", None)
    sync_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or "")
    if pending == "save" and is_valid_admin_token_format(token):
        mount_auth_storage_bridge(action="save", token=token, nonce=sync_nonce)
    elif pending == "clear":
        mount_auth_storage_bridge(action="clear", nonce=sync_nonce)
    else:
        mount_auth_storage_bridge(action="noop")


def _try_authenticate_from_url(settings: AppSettings) -> AuthRole | None:
    url_token = str(st.query_params.get("token", "")).strip()
    if not url_token:
        return None
    token_role = resolve_credential_role(url_token, settings.app_password)
    if token_role == AuthRole.ADMIN:
        admin_token = (
            url_token
            if is_valid_admin_token_format(url_token)
            else generate_admin_token(settings.app_password)
        )
        st.session_state["auth_token"] = admin_token
        st.session_state["remember_me"] = True
        st.session_state.pop("_ignore_context_cookie", None)
        mount_auth_storage_bridge(action="save", token=admin_token, nonce=secrets.token_hex(4))
        return AuthRole.ADMIN
    if token_role == AuthRole.READONLY:
        st.session_state["auth_token"] = url_token
        return AuthRole.READONLY
    del st.query_params["token"]
    _schedule_storage_sync("clear")
    return None


def _try_authenticate_from_context_cookie(settings: AppSettings) -> AuthRole | None:
    if st.session_state.get("_ignore_context_cookie"):
        return None
    cookie_token = _get_cookie_val()
    if not cookie_token or not verify_admin_token(cookie_token, settings.app_password):
        return None
    st.session_state["auth_token"] = cookie_token
    st.session_state["remember_me"] = True
    mount_auth_storage_bridge(action="save", token=cookie_token, nonce=secrets.token_hex(4))
    return AuthRole.ADMIN


def _complete_admin_login(settings: AppSettings, remember_me: bool) -> None:
    st.session_state["auth_token"] = generate_admin_token(settings.app_password)
    st.session_state.pop("_auth_restore_nonce", None)
    if remember_me:
        st.session_state["remember_me"] = True
        st.session_state.pop("_ignore_context_cookie", None)
        _schedule_storage_sync("save")
    else:
        st.session_state["remember_me"] = False
        st.session_state["_ignore_context_cookie"] = True
        if "token" in st.query_params:
            del st.query_params["token"]
        _schedule_storage_sync("clear")
    st.rerun()


def _render_login_form(settings: AppSettings) -> None:
    limiter = _get_rate_limiter()
    st.subheader("🔒 Bot-Dashboard Anmeldung")
    st.caption("Geschützter Zugriff – bitte Dashboard-Passwort eingeben.")

    locked, remaining_sec = limiter.is_locked_out("web")
    if locked:
        st.error(f"Zu viele Fehlversuche. Bitte warte {remaining_sec} Sekunden.")
        return

    with st.form("dashboard_login_form", clear_on_submit=True):
        entered = st.text_input("Dashboard-Passwort", type="password")
        remember_me = st.checkbox("🍪 Auf diesem Gerät 30 Tage angemeldet bleiben", value=True)
        submitted = st.form_submit_button("Anmelden", type="primary", use_container_width=True)

    if not submitted:
        return
    if resolve_credential_role(entered, settings.app_password) != AuthRole.ADMIN:
        limiter.record_failure("web")
        st.error("Ungültiges Passwort.")
        return
    limiter.reset("web")
    _complete_admin_login(settings, remember_me)


def _sync_unauthenticated_storage(settings: AppSettings) -> None:
    pending = st.session_state.pop("_pending_storage_action", None)
    sync_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or "")
    if pending == "clear":
        mount_auth_storage_bridge(action="clear", nonce=sync_nonce or secrets.token_hex(4))
        return
    restore_nonce = str(st.session_state.setdefault("_auth_restore_nonce", secrets.token_hex(4)))
    restored = mount_auth_storage_bridge(action="restore", nonce=restore_nonce)
    if not restored:
        return
    st.session_state.pop("_auth_restore_nonce", None)
    if verify_admin_token(restored, settings.app_password):
        st.session_state["auth_token"] = restored
        st.session_state["remember_me"] = True
        st.session_state.pop("_ignore_context_cookie", None)
        _schedule_storage_sync("save")
    else:
        _schedule_storage_sync("clear")
    st.rerun()


def enforce_authentication(settings: AppSettings) -> AuthRole:
    """Prüft Session-State, URL-Token, Cookie und Browser-Storage und rendert Login bei Bedarf."""
    if not settings.app_password.strip():
        return AuthRole.ADMIN

    if st.session_state.pop("just_logged_out", False):
        clear_auth_cookie()
        st.session_state.pop("_pending_storage_action", None)
        clear_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or secrets.token_hex(4))
        mount_auth_storage_bridge(action="clear", nonce=clear_nonce)
        _render_login_form(settings)
        st.stop()
        return AuthRole.GUEST

    session_token = str(st.session_state.get("auth_token", ""))
    role = resolve_credential_role(session_token, settings.app_password)
    if role == AuthRole.ADMIN:
        _sync_authenticated_storage(session_token)
        return AuthRole.ADMIN
    if role == AuthRole.READONLY:
        return AuthRole.READONLY

    fallback_role = _try_authenticate_from_url(settings) or _try_authenticate_from_context_cookie(settings)
    if fallback_role is not None:
        return fallback_role

    _sync_unauthenticated_storage(settings)
    _render_login_form(settings)
    st.stop()
    return AuthRole.GUEST
