"""Unit- & Integrations-Tests für WebCrypto-Device-PoP, Session-Widerruf und Auth-View."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from src.config.constants import AUTH_COOKIE_NAME
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.security.auth import (
    LoginRateLimiter,
    generate_admin_token,
    generate_readonly_token,
    parse_admin_token,
)
from src.security.device_pop import compute_client_fingerprint, compute_device_key_tag
from src.storage.auth_sessions import get_auth_session_store, reset_auth_session_stores
from src.ui import auth_storage, auth_view


class _StopCalled(Exception):
    pass


class _RerunCalled(Exception):
    pass


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _make_device_key() -> tuple[ec.EllipticCurvePrivateKey, str, str]:
    priv = ec.generate_private_key(ec.SECP256R1())
    spki = _b64u(
        priv.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return priv, spki, compute_device_key_tag(spki)


def _sign_p256(priv: ec.EllipticCurvePrivateKey, msg: str) -> str:
    r, s = decode_dss_signature(priv.sign(msg.encode("utf-8"), ec.ECDSA(hashes.SHA256())))
    return _b64u(r.to_bytes(32, "big") + s.to_bytes(32, "big"))


@pytest.fixture(autouse=True)
def _clean_auth_stores() -> Any:
    reset_auth_session_stores()
    yield
    reset_auth_session_stores()


def _patch_streamlit_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_state: dict[str, Any] | None = None,
    query_params: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    locale: str | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    state: dict[str, Any] = {} if session_state is None else session_state
    params: dict[str, str] = {} if query_params is None else query_params
    calls: list[dict[str, Any]] = []

    fake_ctx = SimpleNamespace(
        cookies={} if cookies is None else cookies,
        headers={} if headers is None else headers,
        locale=locale,
    )
    monkeypatch.setattr(auth_view.st, "session_state", state)
    monkeypatch.setattr(auth_view.st, "query_params", params)
    monkeypatch.setattr(auth_view.st, "context", fake_ctx, raising=False)
    monkeypatch.setattr(auth_view.st, "stop", lambda: (_ for _ in ()).throw(_StopCalled()))
    monkeypatch.setattr(auth_view.st, "rerun", lambda: (_ for _ in ()).throw(_RerunCalled()))
    monkeypatch.setattr(auth_view, "_render_login_form", lambda settings: None)

    def _fake_bridge(
        action: str,
        token: str = "",
        nonce: str = "",
        *,
        challenge: str = "",
        candidate_token: str = "",
    ) -> str:
        calls.append(
            {
                "action": action,
                "token": token,
                "nonce": nonce,
                "challenge": challenge,
                "candidate_token": candidate_token,
            }
        )
        return ""

    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", _fake_bridge)
    return state, params, calls


def test_auth_storage_validation_and_error_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not auth_storage.is_valid_admin_token_format("")
    assert not auth_storage.is_valid_admin_token_format(f"adm:1794570264:{'a' * 64}")
    assert not auth_storage.is_valid_admin_token_format(
        f"adm2:1700000000:{'b' * 24}:{'c' * 16}:session:{'d' * 64}"
    )
    valid = f"adm2:1700000000:{'b' * 24}:{'c' * 16}:{'e' * 32}:{'d' * 64}"
    assert auth_storage.is_valid_admin_token_format(f"  {valid}  ")
    assert auth_storage.parse_bridge_message('{"kind":"restore"}') == {"kind": "restore"}
    assert auth_storage.parse_bridge_message("not-json") is None

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
    assert auth_view.enforce_authentication(AppSettings(app_password="  ", sqlite_path=":memory:")) == AuthRole.ADMIN


def test_admin_url_tokens_and_leaked_v1_token_are_strictly_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret, sqlite_path=":memory:")
    leaked_v1 = "adm:1794570264:bd4dc085291b01b2efb01b613b65a5ac77c03e6b3b7ac477724f5ef833730957"
    valid_v2_session = generate_admin_token(secret)
    ro_token = generate_readonly_token(secret)

    # 1) Exaktes geleaktes v1-Token in URL -> abgewiesen & aus ?token= entfernt
    st1, qp1, _ = _patch_streamlit_state(monkeypatch, query_params={"token": leaked_v1})
    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert "token" not in qp1
    assert "auth_token" not in st1

    # 2) Selbst gültige v2-Admin-Tokens oder Klartext-Passwort in URL -> strikt abgelehnt!
    for forbidden in (valid_v2_session, secret):
        st_x, qp_x, _ = _patch_streamlit_state(monkeypatch, query_params={"token": forbidden})
        with pytest.raises(_StopCalled):
            auth_view.enforce_authentication(settings)
        assert "token" not in qp_x
        assert "auth_token" not in st_x

    # 3) Readonly-Link bleibt als READONLY nutzbar (ohne Persistenz im Browser)
    st_ro, _qp_ro, _ = _patch_streamlit_state(monkeypatch, query_params={"token": ro_token})
    assert auth_view.enforce_authentication(settings) == AuthRole.READONLY
    assert st_ro["auth_token"] == ro_token


def test_cookie_alone_cannot_authenticate_without_webcrypto_pop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret, sqlite_path=":memory:")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"}
    fp = compute_client_fingerprint(headers, "de-DE")
    _priv, _spki, tag = _make_device_key()
    bound_tok = generate_admin_token(secret, client_fingerprint=fp, device_key_tag=tag)

    state, _p, calls = _patch_streamlit_state(
        monkeypatch,
        cookies={AUTH_COOKIE_NAME: bound_tok.replace(":", "%3A")},
        headers=headers,
        locale="de-DE",
    )
    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in state
    assert calls[-1]["action"] == "restore"
    assert calls[-1]["candidate_token"] == bound_tok


def test_pop_bind_restore_and_cross_device_or_revoked_replay_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret, sqlite_path=":memory:")
    chrome_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"}
    firefox_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Firefox/132.0"}
    chrome_fp = compute_client_fingerprint(chrome_headers, "de-DE")

    priv_owner, spki_owner, _tag_owner = _make_device_key()
    priv_attacker, spki_attacker, _ = _make_device_key()

    # 1) Login mit remember_me=True -> Bind-Phase erzeugt gerätegebundenes adm2-Token & registriert Session
    session_tok = generate_admin_token(secret, client_fingerprint=chrome_fp)
    state, _p, calls = _patch_streamlit_state(
        monkeypatch,
        session_state={
            "auth_token": session_tok,
            "remember_me": True,
            "_auth_bind_challenge": "bind-chal-1",
            "_auth_bind_nonce": "bn1",
            "_pending_storage_action": "bind",
        },
        headers=chrome_headers,
        locale="de-DE",
    )

    def _bind_bridge(action: str, **kw: Any) -> str:
        calls.append({"action": action, **kw})
        return json.dumps(
            {
                "kind": "bind",
                "pubkey": spki_owner,
                "signature": _sign_p256(priv_owner, f"bind:{kw['challenge']}"),
                "challenge": kw["challenge"],
            }
        )

    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", _bind_bridge)
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)

    bound_tok = state["auth_token"]
    assert auth_storage.is_valid_admin_token_format(bound_tok)
    assert state["_pending_storage_action"] == "save"
    parsed = parse_admin_token(bound_tok)
    assert parsed is not None

    # 2) Gültiger Restore auf demselben Browser & Gerät (mit eigenem non-exportable ECDSA-Key)
    st_ok, _p_ok, calls_ok = _patch_streamlit_state(monkeypatch, headers=chrome_headers, locale="de-DE")

    def _restore_ok(action: str, **kw: Any) -> str:
        calls_ok.append({"action": action, **kw})
        return json.dumps(
            {
                "kind": "restore",
                "token": bound_tok,
                "pubkey": spki_owner,
                "signature": _sign_p256(priv_owner, f"restore:{kw['challenge']}:{parsed.session_id}"),
                "challenge": kw["challenge"],
            }
        )

    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", _restore_ok)
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert st_ok["auth_token"] == bound_tok

    # 3) Abgefischtes Token auf anderem Rechner (gleicher Chrome, aber falscher IndexedDB-Key!) -> BLOCKED!
    st_pc2, _p2, _ = _patch_streamlit_state(monkeypatch, headers=chrome_headers, locale="de-DE")
    monkeypatch.setattr(
        auth_view,
        "mount_auth_storage_bridge",
        lambda action, **kw: json.dumps(
            {
                "kind": "restore",
                "token": bound_tok,
                "pubkey": spki_attacker,
                "signature": _sign_p256(priv_attacker, f"restore:{kw['challenge']}:{parsed.session_id}"),
                "challenge": kw["challenge"],
            }
        ),
    )
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in st_pc2
    assert st_pc2["_pending_storage_action"] == "clear"

    # 4) Abgefischtes Token in anderem Browser (Firefox statt Chrome) -> fp_tag Mismatch -> BLOCKED!
    st_ff, _p3, _ = _patch_streamlit_state(monkeypatch, headers=firefox_headers, locale="de-DE")
    monkeypatch.setattr(
        auth_view,
        "mount_auth_storage_bridge",
        lambda action, **kw: json.dumps(
            {
                "kind": "restore",
                "token": bound_tok,
                "pubkey": spki_owner,
                "signature": _sign_p256(priv_owner, f"restore:{kw['challenge']}:{parsed.session_id}"),
                "challenge": kw["challenge"],
            }
        ),
    )
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in st_ff
    assert st_ff["_pending_storage_action"] == "clear"

    # 5) Nach Logout / Session-Widerruf darf Token auch im Originalbrowser NICHT mehr funktionieren!
    get_auth_session_store(settings).revoke_session(parsed.session_id)
    st_rev, _p4, _ = _patch_streamlit_state(monkeypatch, headers=chrome_headers, locale="de-DE")
    monkeypatch.setattr(auth_view, "mount_auth_storage_bridge", _restore_ok)
    with pytest.raises(_RerunCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in st_rev
    assert st_rev["_pending_storage_action"] == "clear"


def test_logout_revokes_session_ignores_stale_cookie_and_clears_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret, sqlite_path=":memory:")
    _priv, _spki, tag = _make_device_key()
    valid_token = generate_admin_token(secret, device_key_tag=tag)
    parsed = parse_admin_token(valid_token)
    assert parsed is not None

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
    assert calls[-1]["action"] == "clear"
    assert not get_auth_session_store(settings).is_session_active(parsed.session_id, parsed.issued_at)

    with pytest.raises(_StopCalled):
        auth_view.enforce_authentication(settings)
    assert "auth_token" not in state


def test_login_form_remember_me_and_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "mobile-secret-pw"
    settings = AppSettings(app_password=secret, sqlite_path=":memory:")
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

    # Token darf nicht als Passwort im Login-Formular akzeptiert werden
    state_err: dict[str, Any] = {}
    monkeypatch.setattr(auth_view.st, "session_state", state_err)
    monkeypatch.setattr(auth_view.st, "text_input", lambda *a, **k: generate_admin_token(secret))
    monkeypatch.setattr(auth_view.st, "checkbox", lambda *a, **k: True)
    monkeypatch.setattr(auth_view.st, "form_submit_button", lambda *a, **k: True)
    auth_view._render_login_form(settings)
    assert not limiter.is_locked_out("web")[0]

    auth_view._render_login_form(settings)
    assert limiter.is_locked_out("web")[0]
    limiter.reset("web")

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
    monkeypatch.setenv("SQLITE_PATH", ":memory:")

    def _app() -> None:
        import streamlit as st
        from src.config.settings import load_app_settings
        from src.ui.auth_view import enforce_authentication

        role = enforce_authentication(load_app_settings())
        st.write(f"ROLE:{role.value}")

    at = AppTest.from_function(_app).run()
    assert not at.exception
    assert at.subheader[0].value == "🔒 Bot-Dashboard"
    assert not at.warning
    assert at.checkbox[0].label == "Angemeldet bleiben"

    at.text_input[0].set_value("apptest-secret")
    at.button[0].click().run()
    assert not at.exception
    assert at.markdown[0].value == "ROLE:admin"
    assert at.session_state["remember_me"] is True
    assert parse_admin_token(str(at.session_state["auth_token"])) is not None
    assert at.session_state["_pending_storage_action"] == "bind"


def test_auth_session_store_revoke_all_and_strict_registration(tmp_path: Any) -> None:
    from src.storage.auth_sessions import AuthSessionStore
    from src.storage.sqlite_executor import SqliteExecutor

    db_path = tmp_path / "sessions.db"
    store = AuthSessionStore(SqliteExecutor(db_path), is_remote_persistent=True)
    try:
        # Unregistrierte Session wird bei require_registered=True strikt abgelehnt
        assert not store.is_session_active("a" * 24, 1700000000, require_registered=True)

        store.register_session(
            session_id="a" * 24,
            fp_tag="b" * 16,
            device_key_tag="c" * 32,
            issued_at=1700000000,
            expires_at=1900000000,
        )
        assert store.is_session_active("a" * 24, 1700000000, require_registered=True)
        store.touch_session("a" * 24)

        # Globaler Widerruf invalidiert auch registrierte Sessions sofort über DB-Restart hinweg
        store.revoke_all_sessions()
    finally:
        store.close()

    reopened = AuthSessionStore(SqliteExecutor(db_path), is_remote_persistent=True)
    try:
        assert not reopened.is_session_active("a" * 24, 1700000000, require_registered=True)
    finally:
        reopened.close()
