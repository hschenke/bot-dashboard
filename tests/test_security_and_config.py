"""Tests für SSRF-Guard, HMAC-Authentifizierung, Sanitization und Konfiguration."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.config.settings import _lookup_in_mapping, load_app_settings
from src.domain.enums import AppId, AuthRole
from src.domain.exceptions import SSRFBlockedError, ValidationError
from src.security.auth import (
    LoginRateLimiter,
    generate_admin_token,
    generate_readonly_token,
    resolve_credential_role,
    verify_admin_token,
    verify_readonly_token,
)
from src.security.device_pop import compute_client_fingerprint
from src.security.sanitization import (
    resolve_safe_sqlite_path,
    sanitize_display_url,
    validate_github_repo_slug,
    validate_whitelisted_table_name,
    validate_workflow_filename,
)
from src.security.ssrf_guard import is_public_ip_address, validate_safe_external_url


def test_ssrf_guard_blocks_private_and_loopback_ips() -> None:
    assert not is_public_ip_address("127.0.0.1")
    assert not is_public_ip_address("::1")
    assert not is_public_ip_address("10.0.0.1")
    assert not is_public_ip_address("192.168.1.50")
    assert not is_public_ip_address("169.254.169.254")
    assert is_public_ip_address("8.8.8.8")


def test_ssrf_guard_validates_external_url_and_blocks_rebinding() -> None:
    url = validate_safe_external_url(
        "https://holli-news-bot.streamlit.app/",
        dns_resolver=lambda host, port: ["34.120.10.20"],
    )
    assert url == "https://holli-news-bot.streamlit.app/"

    with pytest.raises(SSRFBlockedError):
        validate_safe_external_url("http://localhost:8501")

    with pytest.raises(SSRFBlockedError):
        validate_safe_external_url(
            "https://evil.example.com",
            dns_resolver=lambda host, port: ["169.254.169.254"],
        )


def test_hmac_tokens_and_role_resolution() -> None:
    secret = "super-secret-pw"
    chrome_fp = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"},
        "de-DE",
    )
    firefox_fp = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Firefox/132.0"},
        "de-DE",
    )
    admin_tok = generate_admin_token(secret, client_fingerprint=chrome_fp)
    ro_tok = generate_readonly_token(secret)

    assert verify_admin_token(admin_tok, secret, client_fingerprint=chrome_fp)
    assert not verify_admin_token(admin_tok, secret, client_fingerprint=firefox_fp)
    assert not verify_admin_token(admin_tok, secret, client_fingerprint=chrome_fp, allow_session_only=False)
    assert not verify_admin_token(admin_tok, "wrong", client_fingerprint=chrome_fp)
    # Legacy v1-Tokens (inkl. geleakter Beispiel-Tokens) werden strikt abgelehnt
    leaked_v1 = "adm:1794570264:bd4dc085291b01b2efb01b613b65a5ac77c03e6b3b7ac477724f5ef833730957"
    assert not verify_admin_token(leaked_v1, secret)
    assert resolve_credential_role(leaked_v1, secret) == AuthRole.GUEST
    assert verify_readonly_token(ro_tok, secret)

    assert resolve_credential_role(secret, secret) == AuthRole.ADMIN
    assert resolve_credential_role(admin_tok, secret, client_fingerprint=chrome_fp) == AuthRole.ADMIN
    assert resolve_credential_role(ro_tok, secret) == AuthRole.READONLY
    assert resolve_credential_role("invalid", secret) == AuthRole.GUEST
    assert resolve_credential_role("", "") == AuthRole.ADMIN


def test_device_pop_ecdsa_p256_and_fingerprint() -> None:
    import base64

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from src.security.device_pop import (
        compute_client_fingerprint,
        compute_device_key_tag,
        verify_device_pop_signature,
    )

    def _b64u(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    priv1 = ec.generate_private_key(ec.SECP256R1())
    priv2 = ec.generate_private_key(ec.SECP256R1())
    spki1 = _b64u(
        priv1.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    spki2 = _b64u(
        priv2.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    tag1 = compute_device_key_tag(spki1)
    assert len(tag1) == 32
    assert tag1 != compute_device_key_tag(spki2)

    msg = "restore:challenge123:0123456789abcdef01234567"
    r, s = decode_dss_signature(priv1.sign(msg.encode("utf-8"), ec.ECDSA(hashes.SHA256())))
    raw_sig = _b64u(r.to_bytes(32, "big") + s.to_bytes(32, "big"))

    assert verify_device_pop_signature(spki1, raw_sig, msg)
    assert not verify_device_pop_signature(spki2, raw_sig, msg)
    assert not verify_device_pop_signature(spki1, raw_sig, "restore:other:0123456789abcdef01234567")

    # Chrome-Minor/Major-Updates bleiben stabil, Browser-/OS-Wechsel erzeugt neues Merkmal
    fp_c131 = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"}, "de-DE"
    )
    fp_c132 = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/132.0.0.0 Safari/537.36"}, "de-DE"
    )
    fp_edge = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"},
        "de-DE",
    )
    fp_ios = compute_client_fingerprint(
        {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) Safari/604.1"}, "de-DE"
    )
    assert fp_c131 == fp_c132
    assert len({fp_c131, fp_edge, fp_ios, compute_client_fingerprint({}, None)}) == 4


def test_login_rate_limiter_locks_after_max_attempts() -> None:
    limiter = LoginRateLimiter(max_attempts=3, lockout_seconds=60)
    assert not limiter.is_locked_out("client1")[0]
    limiter.record_failure("client1")
    limiter.record_failure("client1")
    assert not limiter.is_locked_out("client1")[0]
    limiter.record_failure("client1")
    locked, remaining = limiter.is_locked_out("client1")
    assert locked is True
    assert remaining > 0
    limiter.reset("client1")
    assert not limiter.is_locked_out("client1")[0]


def test_sanitization_and_path_traversal_protection(tmp_path: Path) -> None:
    assert validate_whitelisted_table_name("articles") == "articles"
    with pytest.raises(ValidationError):
        validate_whitelisted_table_name("users; DROP TABLE articles")

    assert validate_github_repo_slug("hschenke/news-bot") == "hschenke/news-bot"
    with pytest.raises(ValidationError):
        validate_github_repo_slug("invalid repo slug")

    assert validate_workflow_filename("check_alerts.yaml") == "check_alerts.yaml"
    with pytest.raises(ValidationError):
        validate_workflow_filename("../secret.txt")

    assert sanitize_display_url("https://example.com/a") == "https://example.com/a"
    assert sanitize_display_url("javascript:alert(1)") == "#"

    safe_file = resolve_safe_sqlite_path(tmp_path, "data/test.db")
    assert safe_file.is_relative_to(tmp_path)

    with pytest.raises(ValidationError):
        resolve_safe_sqlite_path(tmp_path, "../../etc/passwd")


def test_load_app_settings_populates_three_managed_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_BOT_URL", "https://holli-news-bot.streamlit.app/")
    monkeypatch.setenv("NEWS_BOT_REPO", "hschenke/news-bot")
    settings = load_app_settings(allow_sibling_env_fallback=False)
    assert len(settings.managed_apps) == 3
    ids = [spec.app_id for spec in settings.managed_apps]
    assert ids == [AppId.NEWS_BOT, AppId.RESEARCH_BOT, AppId.ALERT_BOT]
    news_spec = settings.get_app_spec(AppId.NEWS_BOT)
    assert "streamlit.app" in news_spec.streamlit_url
    assert news_spec.github_repo == "hschenke/news-bot"


def test_secret_mapping_lookup_flatter_and_nested() -> None:
    data = {
        "NEWS_BOT_URL": "https://flat.example.com",
        "research_bot": {"url": "https://nested.example.com"},
    }
    assert _lookup_in_mapping(data, "NEWS_BOT_URL") == "https://flat.example.com"
    assert _lookup_in_mapping(data, "RESEARCH_BOT_URL") == "https://nested.example.com"
    assert _lookup_in_mapping(data, "UNKNOWN_KEY") is None


def test_30_day_auth_token_and_storage_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from src.config.constants import AUTH_COOKIE_MAX_AGE_SECONDS, AUTH_COOKIE_NAME
    from src.ui import auth_storage

    assert AUTH_COOKIE_MAX_AGE_SECONDS == 30 * 24 * 3600
    assert AUTH_COOKIE_NAME == "bot_dashboard_auth_token"

    secret = "test-secret-123"
    session_only_tok = generate_admin_token(secret)
    device_bound_tok = generate_admin_token(secret, device_key_tag="a" * 32)
    assert verify_admin_token(device_bound_tok, secret, max_age_seconds=AUTH_COOKIE_MAX_AGE_SECONDS)
    assert not auth_storage.is_valid_admin_token_format(session_only_tok)
    assert not auth_storage.is_valid_admin_token_format(f"adm:1794570264:{'b' * 64}")
    assert auth_storage.is_valid_admin_token_format(device_bound_tok)
    assert not auth_storage.is_valid_admin_token_format("invalid")

    fake_component = MagicMock(return_value={"restored": device_bound_tok})
    monkeypatch.setattr(auth_storage, "_get_bridge_component", lambda: fake_component)

    assert auth_storage.mount_auth_storage_bridge("save", "invalid") == ""
    assert auth_storage.mount_auth_storage_bridge("save", session_only_tok) == ""
    fake_component.assert_not_called()

    auth_storage.mount_auth_storage_bridge("save", device_bound_tok, nonce="n1")
    assert fake_component.call_args.kwargs["data"]["action"] == "save"
    assert fake_component.call_args.kwargs["data"]["token"] == device_bound_tok
    assert fake_component.call_args.kwargs["data"]["maxAge"] == AUTH_COOKIE_MAX_AGE_SECONDS

    assert auth_storage.mount_auth_storage_bridge("restore", nonce="n2", challenge="c1") == device_bound_tok
    auth_storage.mount_auth_storage_bridge("clear", nonce="n3")
    assert fake_component.call_args.kwargs["data"]["action"] == "clear"

