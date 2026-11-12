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
from src.security.sanitization import (
    resolve_safe_sqlite_path,
    sanitize_display_url,
    validate_github_repo_slug,
    validate_user_question,
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
    admin_tok = generate_admin_token(secret)
    ro_tok = generate_readonly_token(secret)

    assert verify_admin_token(admin_tok, secret)
    assert not verify_admin_token(admin_tok, "wrong")
    assert verify_readonly_token(ro_tok, secret)

    assert resolve_credential_role(secret, secret) == AuthRole.ADMIN
    assert resolve_credential_role(admin_tok, secret) == AuthRole.ADMIN
    assert resolve_credential_role(ro_tok, secret) == AuthRole.READONLY
    assert resolve_credential_role("invalid", secret) == AuthRole.GUEST
    assert resolve_credential_role("", "") == AuthRole.ADMIN


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
    assert validate_user_question("  Wie ist der Status? ") == "Wie ist der Status?"

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


def test_30_day_auth_token_and_storage_scripts() -> None:
    from src.config.constants import AUTH_COOKIE_MAX_AGE_SECONDS, AUTH_COOKIE_NAME
    from src.ui.auth_view import (
        _render_auto_restore_script,
        _render_clear_storage_script,
        _render_save_storage_script,
    )

    assert AUTH_COOKIE_MAX_AGE_SECONDS == 30 * 24 * 3600
    assert AUTH_COOKIE_NAME == "bot_dashboard_auth_token"

    secret = "test-secret-123"
    token = generate_admin_token(secret)
    assert verify_admin_token(token, secret, max_age_seconds=AUTH_COOKIE_MAX_AGE_SECONDS)

    _render_save_storage_script(token)
    _render_save_storage_script("invalid")
    _render_clear_storage_script()
    _render_auto_restore_script()


