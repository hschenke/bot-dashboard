"""Authentifizierungs-View mit Geräte-PoP, Kontext-Bindung, Server-Sessions & Rate-Limiter."""

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
    parse_admin_token,
    verify_admin_token,
    verify_password_secret,
    verify_readonly_token,
)
from src.security.device_pop import compute_client_fingerprint
from src.storage.auth_sessions import get_auth_session_store
from src.ui.auth_storage import is_valid_admin_token_format, mount_auth_storage_bridge
from src.ui.auth_sync import (
    schedule_storage_sync,
    sync_authenticated_storage,
    sync_unauthenticated_storage,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_rate_limiter() -> LoginRateLimiter:
    return LoginRateLimiter()


def _current_client_fingerprint() -> str:
    try:
        ctx = getattr(st, "context", None)
        headers = getattr(ctx, "headers", None) if ctx is not None else None
        locale = getattr(ctx, "locale", None) if ctx is not None else None
        return compute_client_fingerprint(headers, locale)
    except Exception:  # noqa: BLE001
        return "no_client_context"


def _get_cookie_val() -> str:
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            val = st.context.cookies.get(AUTH_COOKIE_NAME)
            if val:
                return unquote(str(val)).strip()
    except Exception as exc:
        logger.debug("Fehler beim Lesen aus st.context.cookies: %s", exc)
    return ""


def clear_auth_cookie(settings: AppSettings | None = None) -> None:
    """Entfernt das Token aus Session, URL & Browser und widerruft die Server-Session."""
    old_token = str(st.session_state.pop("auth_token", "") or "")
    if settings is not None and old_token:
        parsed = parse_admin_token(old_token)
        if parsed is not None:
            get_auth_session_store(settings).revoke_session(parsed.session_id)

    for key in (
        "remember_me",
        "_auth_restore_nonce",
        "_auth_restore_challenge",
        "_auth_bind_challenge",
        "_auth_bind_nonce",
    ):
        st.session_state.pop(key, None)
    st.session_state["_ignore_context_cookie"] = True
    if "token" in st.query_params:
        del st.query_params["token"]
    schedule_storage_sync("clear")


def _try_authenticate_from_url(settings: AppSettings) -> AuthRole | None:
    url_token = str(st.query_params.get("token", "")).strip()
    if not url_token:
        return None
    if verify_readonly_token(url_token, settings.app_password):
        st.session_state["auth_token"] = url_token
        return AuthRole.READONLY
    del st.query_params["token"]
    st.session_state["_auth_notice"] = (
        "URL-Anmeldung mit Admin-Tokens ist deaktiviert. Bitte melde dich direkt mit dem Passwort an."
    )
    return None


def _complete_admin_login(settings: AppSettings, remember_me: bool, client_fp: str) -> None:
    st.session_state["auth_token"] = generate_admin_token(
        settings.app_password, client_fingerprint=client_fp
    )
    st.session_state.pop("_auth_restore_nonce", None)
    st.session_state.pop("_auth_restore_challenge", None)
    st.session_state.pop("_auth_notice", None)
    if "token" in st.query_params:
        del st.query_params["token"]
    if remember_me:
        st.session_state["remember_me"] = True
        st.session_state.pop("_ignore_context_cookie", None)
        st.session_state["_auth_bind_challenge"] = secrets.token_hex(16)
        st.session_state["_auth_bind_nonce"] = secrets.token_hex(4)
        st.session_state["_pending_storage_action"] = "bind"
    else:
        st.session_state["remember_me"] = False
        st.session_state["_ignore_context_cookie"] = True
        schedule_storage_sync("clear")
    st.rerun()


def _render_login_form(settings: AppSettings) -> None:
    limiter = _get_rate_limiter()
    st.subheader("🔒 Bot-Dashboard Anmeldung")
    st.caption("Geschützter Zugriff – bitte Dashboard-Passwort eingeben.")
    notice = st.session_state.pop("_auth_notice", None)
    if notice:
        st.warning(str(notice))

    locked, remaining_sec = limiter.is_locked_out("web")
    if locked:
        st.error(f"Zu viele Fehlversuche. Bitte warte {remaining_sec} Sekunden.")
        return

    with st.form("dashboard_login_form", clear_on_submit=True):
        entered = st.text_input("Dashboard-Passwort", type="password")
        remember_me = st.checkbox("🍪 Auf diesem Gerät 30 Tage angemeldet bleiben", value=True)
        submitted = st.form_submit_button("Anmelden", type="primary", width="stretch")

    if not submitted:
        return
    if not verify_password_secret(entered, settings.app_password):
        limiter.record_failure("web")
        st.error("Ungültiges Passwort.")
        return
    limiter.reset("web")
    _complete_admin_login(settings, remember_me, _current_client_fingerprint())


def enforce_authentication(settings: AppSettings) -> AuthRole:
    """Prüft Session-State, Geräte-PoP und Server-Session-Status und rendert Login bei Bedarf."""
    if not settings.app_password.strip():
        return AuthRole.ADMIN

    if st.session_state.pop("just_logged_out", False):
        clear_auth_cookie(settings)
        st.session_state.pop("_pending_storage_action", None)
        clear_nonce = str(st.session_state.pop("_storage_sync_nonce", "") or secrets.token_hex(4))
        mount_auth_storage_bridge(action="clear", nonce=clear_nonce)
        _render_login_form(settings)
        st.stop()
        return AuthRole.GUEST

    client_fp = _current_client_fingerprint()
    session_token = str(st.session_state.get("auth_token", ""))
    if verify_admin_token(
        session_token, settings.app_password, client_fingerprint=client_fp, allow_session_only=True
    ):
        parsed = parse_admin_token(session_token)
        store = get_auth_session_store(settings)
        if parsed is not None and store.is_session_active(
            parsed.session_id,
            parsed.issued_at,
            require_registered=parsed.is_persistent_device_bound and store.is_remote_persistent,
        ):
            if "token" in st.query_params:
                del st.query_params["token"]
            sync_authenticated_storage(settings, session_token, client_fp, mount_auth_storage_bridge)
            return AuthRole.ADMIN
        st.session_state.pop("auth_token", None)
        schedule_storage_sync("clear")

    if verify_readonly_token(session_token, settings.app_password):
        return AuthRole.READONLY

    if _try_authenticate_from_url(settings) == AuthRole.READONLY:
        return AuthRole.READONLY

    cookie_cand = "" if st.session_state.get("_ignore_context_cookie") else _get_cookie_val()
    if cookie_cand and not is_valid_admin_token_format(cookie_cand):
        cookie_cand = ""
    sync_unauthenticated_storage(settings, client_fp, cookie_cand, mount_auth_storage_bridge)
    _render_login_form(settings)
    st.stop()
    return AuthRole.GUEST
