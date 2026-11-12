"""Authentifizierungs-View mit 30-Tage-Cookie, Session-Token und Brute-Force-Schutz."""

from __future__ import annotations

import logging
import re

import streamlit as st

from src.config.constants import AUTH_COOKIE_MAX_AGE_SECONDS, AUTH_COOKIE_NAME
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.security.auth import (
    LoginRateLimiter,
    generate_admin_token,
    resolve_credential_role,
    verify_admin_token,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_rate_limiter() -> LoginRateLimiter:
    return LoginRateLimiter()


def _safe_html(js: str) -> None:
    try:
        st.html(js)
    except Exception as exc:
        logger.debug("Fehler beim Rendern des JS-Snippets: %s", exc)


def _render_save_storage_script(token: str) -> None:
    """Persistiert das Token in clientseitigem localStorage und document.cookie."""
    if not token or not token.startswith("adm:") or not re.fullmatch(r"adm:\d+:[0-9a-f]{64}", token):
        return
    js = (
        f'<script>try{{localStorage.setItem("{AUTH_COOKIE_NAME}","{token}");'
        f'const s=location.protocol==="https:"?"; Secure":"";'
        f'document.cookie="{AUTH_COOKIE_NAME}={token}; max-age={AUTH_COOKIE_MAX_AGE_SECONDS}; path=/; SameSite=Lax"+s;'
        f'}}catch(e){{console.debug("save err",e);}}</script>'
    )
    _safe_html(js)


def _render_clear_storage_script() -> None:
    """Entfernt Token aus clientseitigem localStorage und document.cookie."""
    js = (
        f'<script>try{{localStorage.removeItem("{AUTH_COOKIE_NAME}");'
        f'const s=location.protocol==="https:"?"; Secure":"";'
        f'document.cookie="{AUTH_COOKIE_NAME}=; max-age=0; path=/; SameSite=Lax"+s;'
        f'}}catch(e){{console.debug("clear err",e);}}</script>'
    )
    _safe_html(js)


def _render_auto_restore_script() -> None:
    """Stellt gespeichertes Token aus localStorage/Cookie beim App-Start wieder her."""
    js = (
        f'<script>(function(){{try{{'
        f'let t=localStorage.getItem("{AUTH_COOKIE_NAME}");'
        f'if(!t){{const m=document.cookie.match(new RegExp("(^|;\\\\s*){AUTH_COOKIE_NAME}=([^;]*)"));if(m)t=decodeURIComponent(m[2]);}}'
        f'if(t&&t.startsWith("adm:")){{const u=new URL(location.href);'
        f'if(u.searchParams.get("token")!==t){{u.searchParams.set("token",t);location.replace(u.toString());}}'
        f'}}}}catch(e){{console.debug("restore err",e);}}}})();</script>'
    )
    _safe_html(js)


def _get_cookie_val() -> str:
    try:
        if hasattr(st, "context") and hasattr(st.context, "cookies"):
            val = st.context.cookies.get(AUTH_COOKIE_NAME)
            if val:
                return str(val).strip()
    except Exception as exc:
        logger.debug("Fehler beim Lesen aus st.context.cookies: %s", exc)
    return ""


def clear_auth_cookie() -> None:
    """Entfernt das Authentifizierungs-Cookie und bereinigt den Speicher."""
    _render_clear_storage_script()
    st.session_state.pop("auth_token", None)
    if "token" in st.query_params:
        del st.query_params["token"]


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

    if submitted:
        checked_role = resolve_credential_role(entered, settings.app_password)
        if checked_role == AuthRole.ADMIN:
            limiter.reset("web")
            token = generate_admin_token(settings.app_password)
            st.session_state["auth_token"] = token
            if remember_me:
                st.query_params["token"] = token
                st.session_state["remember_me"] = True
                _render_save_storage_script(token)
            else:
                st.session_state["remember_me"] = False
                if "token" in st.query_params:
                    del st.query_params["token"]
                _render_clear_storage_script()
            st.rerun()
        else:
            limiter.record_failure("web")
            st.error("Ungültiges Passwort.")


def enforce_authentication(settings: AppSettings) -> AuthRole:
    """Prüft Session-State, 30-Tage-Browser-Persistenz oder URL-Token und rendert Login bei Bedarf."""
    if not settings.app_password.strip():
        return AuthRole.ADMIN

    if st.session_state.pop("just_logged_out", False):
        clear_auth_cookie()
        _render_login_form(settings)
        st.stop()
        return AuthRole.GUEST

    # 1. Aktive Session im Speicher
    session_token = str(st.session_state.get("auth_token", ""))
    role = resolve_credential_role(session_token, settings.app_password)
    if role == AuthRole.ADMIN:
        if st.session_state.get("remember_me") and not st.session_state.get("storage_synced"):
            _render_save_storage_script(session_token)
            st.session_state["storage_synced"] = True
        return AuthRole.ADMIN

    # 2. URL-Token (z. B. aus E-Mail-Direktlinks oder Mobile-Bookmark)
    url_token = str(st.query_params.get("token", "")).strip()
    if url_token:
        token_role = resolve_credential_role(url_token, settings.app_password)
        if token_role in {AuthRole.ADMIN, AuthRole.READONLY}:
            st.session_state["auth_token"] = url_token
            if token_role == AuthRole.ADMIN:
                _render_save_storage_script(url_token)
                st.session_state["storage_synced"] = True
            return token_role
        del st.query_params["token"]
        _render_clear_storage_script()

    # 3. Persistentes 30-Tage-Cookie im Browser via st.context.cookies
    cookie_token = _get_cookie_val()
    if cookie_token and verify_admin_token(cookie_token, settings.app_password):
        st.session_state["auth_token"] = cookie_token
        st.query_params["token"] = cookie_token
        _render_save_storage_script(cookie_token)
        st.session_state["storage_synced"] = True
        return AuthRole.ADMIN

    if role == AuthRole.READONLY:
        return AuthRole.READONLY

    # 4. Automatisches Wiederherstellen aus localStorage/Cookie anstoßen
    _render_auto_restore_script()

    _render_login_form(settings)
    st.stop()
    return AuthRole.GUEST
