"""HMAC-SHA256 Token-Authentifizierung (v2 mit Kontext- & Gerätebindung)."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass

from src.config.constants import ADMIN_TOKEN_MAX_AGE_SECONDS, CLOCK_SKEW_TOLERANCE_SECONDS
from src.domain.enums import AuthRole
from src.security.device_pop import compute_fingerprint_tag
from src.security.rate_limiter import LoginRateLimiter as LoginRateLimiter

_ADMIN_SCHEME_V2 = "adm2"
_READONLY_SCHEME = "ro"
_HMAC_DOMAIN_ADMIN_V2 = "bot_dashboard_admin_v2"
_HMAC_DOMAIN_READONLY = "bot_dashboard_readonly_v1"
SESSION_DEVICE_TAG = "session"

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{24}$")
_DEVICE_TAG_RE = re.compile(r"^[0-9a-f]{32}$")
_ADMIN_V2_RE = re.compile(
    r"^adm2:(\d+):([0-9a-f]{24}):([0-9a-f]{16}):(session|[0-9a-f]{32}):([0-9a-f]{64})$"
)


@dataclass(frozen=True)
class AdminTokenParts:
    """Strukturierte Bestandteile eines v2-Admin-Session-Tokens."""

    issued_at: int
    session_id: str
    fp_tag: str
    device_key_tag: str
    signature: str

    @property
    def is_persistent_device_bound(self) -> bool:
        return self.device_key_tag != SESSION_DEVICE_TAG


def _compute_hmac_hex(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def is_valid_device_key_tag(tag: str) -> bool:
    """Prüft, ob ein Tag dem 32-Hex-Public-Key-Fingerprint entspricht."""
    return bool(tag and _DEVICE_TAG_RE.fullmatch(tag.strip()))


def parse_admin_token(token: str) -> AdminTokenParts | None:
    """Zerlegt ein v2-Admin-Token (`adm2:...`); lehnt Legacy-v1-Tokens strikt ab."""
    match = _ADMIN_V2_RE.fullmatch((token or "").strip())
    if not match:
        return None
    ts_raw, sid, fp_tag, device_tag, sig = match.groups()
    return AdminTokenParts(
        issued_at=int(ts_raw),
        session_id=sid,
        fp_tag=fp_tag,
        device_key_tag=device_tag,
        signature=sig,
    )


def generate_admin_token(
    password: str,
    issued_at: int | None = None,
    *,
    client_fingerprint: str = "",
    device_key_tag: str = SESSION_DEVICE_TAG,
    session_id: str | None = None,
) -> str:
    """Erstellt ein signiertes, kontext- und optional gerätegebundenes v2-Admin-Token."""
    clean_pw = password.strip()
    if not clean_pw:
        return ""
    clean_dev = device_key_tag.strip()
    if clean_dev != SESSION_DEVICE_TAG and not is_valid_device_key_tag(clean_dev):
        raise ValueError("Ungültiger device_key_tag für Admin-Token.")
    sid = (session_id or secrets.token_hex(12)).strip().lower()
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError("Ungültige session_id für Admin-Token.")

    timestamp = str(issued_at if issued_at is not None else int(time.time()))
    fp_tag = compute_fingerprint_tag(clean_pw, client_fingerprint)
    payload = f"{_HMAC_DOMAIN_ADMIN_V2}:{timestamp}:{sid}:{fp_tag}:{clean_dev}"
    signature = _compute_hmac_hex(clean_pw, payload)
    return f"{_ADMIN_SCHEME_V2}:{timestamp}:{sid}:{fp_tag}:{clean_dev}:{signature}"


def verify_admin_token(
    token: str,
    password: str,
    max_age_seconds: int = ADMIN_TOKEN_MAX_AGE_SECONDS,
    *,
    client_fingerprint: str = "",
    allow_session_only: bool = True,
    expected_device_key_tag: str | None = None,
) -> bool:
    """Prüft Signatur, Ablaufzeit, Client-Kontext und Gerätebindung eines v2-Admin-Tokens."""
    clean_pw = password.strip()
    parsed = parse_admin_token(token)
    if not clean_pw or parsed is None:
        return False
    if not allow_session_only and not parsed.is_persistent_device_bound:
        return False
    if expected_device_key_tag is not None and not hmac.compare_digest(
        parsed.device_key_tag, expected_device_key_tag.strip()
    ):
        return False

    now_ts = int(time.time())
    if (now_ts - parsed.issued_at) > max_age_seconds or parsed.issued_at > (
        now_ts + CLOCK_SKEW_TOLERANCE_SECONDS
    ):
        return False

    expected_fp_tag = compute_fingerprint_tag(clean_pw, client_fingerprint)
    if not hmac.compare_digest(parsed.fp_tag, expected_fp_tag):
        return False

    payload = (
        f"{_HMAC_DOMAIN_ADMIN_V2}:{parsed.issued_at}:{parsed.session_id}:"
        f"{parsed.fp_tag}:{parsed.device_key_tag}"
    )
    return hmac.compare_digest(parsed.signature, _compute_hmac_hex(clean_pw, payload))


def verify_password_secret(entered: str, expected_password: str) -> bool:
    """Prüft ausschließlich das echte Dashboard-Passwort in konstanter Zeit."""
    clean_pw = expected_password.strip()
    clean_entered = (entered or "").strip()
    if not clean_pw:
        return True
    return bool(clean_entered and hmac.compare_digest(clean_entered, clean_pw))


def generate_readonly_token(password: str, scope: str = "dashboard") -> str:
    """Erzeugt ein Lese-Token für signierte Direktlinks."""
    clean_pw = password.strip()
    if not clean_pw:
        return ""
    signature = _compute_hmac_hex(clean_pw, f"{_HMAC_DOMAIN_READONLY}:{scope}")
    return f"{_READONLY_SCHEME}:{scope}:{signature}"


def verify_readonly_token(token: str, password: str) -> bool:
    """Verifiziert ein Read-Only-Token in konstanter Zeit."""
    clean_pw = password.strip()
    if not token or not clean_pw or not token.startswith(f"{_READONLY_SCHEME}:"):
        return False
    parts = token.split(":", 2)
    if len(parts) != 3:
        return False
    expected_sig = _compute_hmac_hex(clean_pw, f"{_HMAC_DOMAIN_READONLY}:{parts[1]}")
    return hmac.compare_digest(parts[2], expected_sig)


def resolve_credential_role(
    credential: str,
    expected_password: str,
    max_age_seconds: int = ADMIN_TOKEN_MAX_AGE_SECONDS,
    *,
    client_fingerprint: str = "",
    allow_session_only: bool = True,
) -> AuthRole:
    """Ermittelt die Zugriffsrolle anhand eines Passworts oder signierten v2-Tokens."""
    clean_pw = expected_password.strip()
    if not clean_pw:
        return AuthRole.ADMIN

    candidate = credential.strip()
    if not candidate:
        return AuthRole.GUEST
    if hmac.compare_digest(candidate, clean_pw) or verify_admin_token(
        candidate,
        clean_pw,
        max_age_seconds=max_age_seconds,
        client_fingerprint=client_fingerprint,
        allow_session_only=allow_session_only,
    ):
        return AuthRole.ADMIN
    if verify_readonly_token(candidate, clean_pw):
        return AuthRole.READONLY
    return AuthRole.GUEST
