"""Unit- & Integrations-Tests für die mobile Auth-Storage-Bridge und Auth-View."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from src.config.constants import AUTH_COOKIE_NAME
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.security.auth import LoginRateLimiter, generate_admin_token, generate_readonly_token
from src.ui import auth_storage, auth_view


class _StopCalled(Exception):
    pass


class _RerunCalled(Exception):
    pass


def _patch_streamlit_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_state: dict[str, Any] | None = None,
    query_params: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[tuple[str, str, str]]]:
    state: dict[str, Any] = {} if session_state is None else session_state
    params: dict[str, str] = {} if query_params is None else query_params
    calls: list[tuple[str, str, str]] = []

    fake_ctx = SimpleNamespace(cookies={} if cookies is None else cookies)
    monkeypatch.setattr(auth_view.st, "session_state", state)
    monkeypatch.setattr(auth_view.st, "query_params", params)
    monkeypatch.setattr(auth_view.st, "context", fake_ctx, raising=False)
    monkeypatch.setattr(auth_view.st, "stop", lambda: (_ for _ in ()).throw(_StopCalled()))
    monkeypatch.setattr(auth_view.st, "rerun", lambda: (_ for _ in ()).throw(_RerunCalled()))
    monkeypatch.setattr(auth_view, "_render_login_form", lambda settings: None)

    def _fake_bridge(action: str, token: str = "", nonce: str = "") -> str:
        calls.append((action, token, nonce))
        return ""

    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", _fake_bridge)
    return state, params, calls


def test_auth_storage_validation_and_error_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not auth_storage.is_valid_admin_token_format("")
    assert not auth_storage.is_valid_admin_token_format("adm:123:" + "a" * 63)
    valid = f"adm:1700000000:{'a' * 64}"
    assert auth_storage.is_valid_admin_token_format(f"  {valid}  ")

    monkeypatch.setattr(auth_storage, "_get_bridge_component", lambda: None)
    assert auth_storage.mount_auth_storage_bridge("restore") == ""

    monkeypatch.setattr(
        auth_storage,
        "_get_bridge_component",
        lambda: lambda **kwargs: SimpleNamespace(restored=valid),
    )
    assert auth_storage.mount_auth_storage_bridge("restore") == valid

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(auth_storage, "_get_bridge_component", lambda: _boom)
    assert auth_storage.mount_auth_storage_bridge("restore") == ""


def test_enforce_authentication_allows_open_mode_without_password() -> None:
    assert auth_view.enforce_authentication(AppSettings(app_password="  ")) == AuthRole.ADMIN


def test_enforce_authentication_restores_and_clears_expired_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret)
    valid_token = generate_admin_token(secret)

    # 1) Valid restored token from browser bridge sets session & schedules save
    state, _params, calls = _patch_streamlit_state(monkeypatch)
    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", lambda **kw: calls.append((kw["action"], kw.get("token", ""), kw.get("nonce", ""))) or valid_token)
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert state["auth_token"] == valid_token
    assert state["remember_me"] is True
    assert state["_pending_storage_action"] == "save"

    # Subsequent authenticated run flushes "save", then "noop"
    assert auth_view.enforce_authentication(settings) == AuthRole.ADMIN
    assert calls[-1][0] == "save"
    assert calls[-1][1] == valid_token
    assert auth_view.enforce_authentication(settings) == AuthRole.ADMIN
    assert calls[-1][0] == "noop"

    # 2) Expired / invalid token from browser bridge schedules clear cleanly
    expired = f"adm:1000000:{'0' * 64}"
    state2, _p2, calls2 = _patch_streamlit_state(monkeypatch)
    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", lambda **kw: calls2.append((kw["action"], "", "")) or expired)
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in state2
    assert state2["_pending_storage_action"] == "clear"

    # Next run executes "clear" without re-triggering restore
    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", lambda **kw: calls2.append((kw["action"], "", "")) or "")
    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert calls2[-1][0] == "clear"


def test_logout_ignores_stale_websocket_cookie_and_clears_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret)
    valid_token = generate_admin_token(secret)

    # Logout with stale WebSocket cookie present in st.context.cookies
    state, params, calls = _patch_streamlit_state(
        monkeypatch,
        session_state={"auth_token": valid_token, "just_logged_out": True},
        query_params={"token": valid_token},
        cookies={AUTH_COOKIE_NAME: valid_token.replace(":", "%3A")},
    )
    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in state
    assert "token" not in params
    assert state["_ignore_context_cookie"] is True
    assert calls[-1][0] == "clear"

    # Follow-up rerun without reload must NOT auto-login from stale WS cookie
    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in state


def test_context_cookie_and_url_token_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret)
    valid_token = generate_admin_token(secret)
    ro_token = generate_readonly_token(secret)

    # URL-encoded cookie restores immediately & saves via bridge
    state, _p, calls = _patch_streamlit_state(
        monkeypatch, cookies={AUTH_COOKIE_NAME: valid_token.replace(":", "%3A")}
    )
    assert auth_view.enforce_authentication(settings) == AuthRole.ADMIN
    assert state["auth_token"] == valid_token
    assert calls[-1] == ("save", valid_token, calls[-1][2])

    # Raw URL password upgrades to HMAC token cleanly
    state_pw, _p2, _c2 = _patch_streamlit_state(monkeypatch, query_params={"token": secret})
    assert auth_view.enforce_authentication(settings) == AuthRole.ADMIN
    assert auth_storage.is_valid_admin_token_format(state_pw["auth_token"])
    assert state_pw["auth_token"] != secret

    # Readonly token grants READONLY
    state_ro, _p3, _c3 = _patch_streamlit_state(monkeypatch, query_params={"token": ro_token})
    assert auth_view.enforce_authentication(settings) == AuthRole.READONLY
    assert state_ro["auth_token"] == ro_token


def test_login_form_remember_me_and_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret)
    limiter = LoginRateLimiter(max_attempts=2, lockout_seconds=60)
    monkeypatch.setattr(auth_view, "_get_rate_limiter", lambda: limiter)

    class _FormCtx:
        def __enter__(self) -> _FormCtx:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    monkeypatch.setattr(auth_view.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(auth_view.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(auth_view.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(auth_view.st, "form", lambda *a, **k: _FormCtx())
    monkeypatch.setattr(auth_view.st, "rerun", lambda: (_ for _ in ()).throw(_RerunCalled()))

    # Wrong password increments limiter
    state_err: dict[str, Any] = {}
    monkeypatch.setattr(auth_view.st, "session_state", state_err)
    monkeypatch.setattr(auth_view.st, "text_input", lambda *a, **k: "wrong")
    monkeypatch.setattr(auth_view.st, "checkbox", lambda *a, **k: True)
    monkeypatch.setattr(auth_view.st, "form_submit_button", lambda *a, **k: True)
    auth_view._render_login_form(settings)
    assert not limiter.is_locked_out("web")[0]

    # Second wrong password triggers lockout
    auth_view._render_login_form(settings)
    assert limiter.is_locked_out("web")[0]
    limiter.reset("web")

    # Successful login with remember_me=False clears storage & ignores stale WS cookie
    state_false: dict[str, Any] = {}
    params_false: dict[str, str] = {"token": "stale"}
    monkeypatch.setattr(auth_view.st, "session_state", state_false)
    monkeypatch.setattr(auth_view.st, "query_params", params_false)
    monkeypatch.setattr(auth_view.st, "text_input", lambda *a, **k: secret)
    monkeypatch.setattr(auth_view.st, "checkbox", lambda *a, **k: False)
    with pytest.raises(_RerunCalled):
        auth_view._render_login_form(settings)
    assert state_false["remember_me"] is False
    assert state_false["_pending_storage_action"] == "clear"
    assert state_false["_ignore_context_cookie"] is True
    assert "token" not in params_false


def test_streamlit_apptest_login_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_PASSWORD", "apptest-secret")

    def _app() -> None:
        import streamlit as st
        from src.config.settings import load_app_settings
        from src.ui.auth_view import enforce_authentication

        role = enforce_authentication(load_app_settings())
        st.write(f"ROLE:{role.value}")

    at = AppTest.from_function(_app).run()
    assert not at.exception
    assert at.subheader[0].value == "🔒 Bot-Dashboard Anmeldung"

    at.text_input[0].set_value("apptest-secret")
    at.button[0].click().run()
    assert not at.exception
    assert at.markdown[0].value == "ROLE:admin"
    assert at.session_state["remember_me"] is True
    assert auth_storage.is_valid_admin_token_format(str(at.session_state["auth_token"]))

