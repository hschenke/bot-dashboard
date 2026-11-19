"""Strikte App-Zugriffskontrolle für das KI-Brain Framework (Whitelist + Token-Auth)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ki_brain.exceptions import BrainAuthError

if TYPE_CHECKING:
    from ki_brain.storage import BrainRepository

ALLOWED_CLIENT_APPS: tuple[str, ...] = ("news_bot", "research_bot", "alert_bot")
INTERNAL_DASHBOARD_APP: str = "bot_dashboard"

CLIENT_APP_DISPLAY_NAMES: dict[str, str] = {
    "news_bot": "News-Bot",
    "research_bot": "Research-Bot",
    "alert_bot": "Alert-Bot",
}

_TOKEN_PREFIX = "kib_"  # noqa: S105
_HASH_PREFIX = "sha256_v1:"


class AppClientStatus(BaseModel):
    """Sicherer Zugriffsstatus einer freigegebenen App (enthält niemals Token oder Hash!)."""

    model_config = ConfigDict(frozen=True)

    app_id: str
    display_name: str
    has_token: bool = False
    enabled: bool = False
    created_at: str = ""
    last_used_at: str = ""


def normalize_client_app_id(raw_app_id: str) -> str:
    """Normalisiert und prüft eine App-ID strikt gegen die 3-App-Whitelist."""
    clean = (raw_app_id or "").strip().lower().replace("-", "_")
    if clean not in ALLOWED_CLIENT_APPS:
        allowed = ", ".join(ALLOWED_CLIENT_APPS)
        raise BrainAuthError(
            f"Zugriff verweigert: App '{raw_app_id or 'unbekannt'}' ist nicht zugelassen "
            f"(erlaubt sind ausschließlich: {allowed})."
        )
    return clean


def _compute_salted_token_hash(app_id: str, token: str, salt: bytes) -> str:
    digest = hmac.new(salt, f"{app_id}:{token}".encode(), hashlib.sha256).hexdigest()
    return f"{_HASH_PREFIX}{salt.hex()}:{digest}"


def generate_app_access_token(app_id: str) -> tuple[str, str]:
    """Erzeugt ein kryptographisches 256-Bit-App-Token und seinen gesalzenen Hash."""
    clean_app = normalize_client_app_id(app_id)
    token = f"{_TOKEN_PREFIX}{clean_app}_{secrets.token_urlsafe(32)}"
    salt = secrets.token_bytes(16)
    return token, _compute_salted_token_hash(clean_app, token, salt)


def verify_app_access_token(app_id: str, token: str, stored_hash: str) -> bool:
    """Prüft ein App-Access-Token in konstanter Zeit gegen den gespeicherten Hash."""
    try:
        clean_app = normalize_client_app_id(app_id)
    except BrainAuthError:
        return False
    clean_tok = (token or "").strip()
    raw_hash = (stored_hash or "").strip()
    if not clean_tok.startswith(f"{_TOKEN_PREFIX}{clean_app}_") or not raw_hash.startswith(_HASH_PREFIX):
        return False
    parts = raw_hash.removeprefix(_HASH_PREFIX).split(":", 1)
    if len(parts) != 2:
        return False
    try:
        salt = bytes.fromhex(parts[0])
    except ValueError:
        return False
    expected = _compute_salted_token_hash(clean_app, clean_tok, salt)
    return hmac.compare_digest(expected, raw_hash)


def authorize_client_app(repo: BrainRepository, app_id: str, app_token: str) -> str:
    """Verifiziert App-ID + Access-Token gegen die Brain-DB (fail-closed)."""
    clean_app = normalize_client_app_id(app_id)
    clean_tok = (app_token or "").strip()
    if not clean_tok:
        raise BrainAuthError(
            f"Zugriff verweigert: Kein BRAIN_APP_TOKEN für '{clean_app}' angegeben."
        )
    record = repo.get_app_client_record(clean_app)
    if not record or not record.get("token_hash"):
        raise BrainAuthError(
            f"Zugriff verweigert: Für '{clean_app}' ist kein aktives Token im Vault hinterlegt."
        )
    if not bool(record.get("enabled", False)):
        raise BrainAuthError(
            f"Zugriff verweigert: App '{clean_app}' ist im KI-Brain Vault gesperrt."
        )
    if not verify_app_access_token(clean_app, clean_tok, str(record["token_hash"])):
        raise BrainAuthError(
            f"Zugriff verweigert: Ungültiges App-Access-Token für '{clean_app}'."
        )
    repo.touch_app_client_last_used(clean_app)
    return clean_app


def build_app_client_statuses(records: dict[str, dict[str, Any]]) -> list[AppClientStatus]:
    """Erstellt die geordnete Statusliste der 3 exklusiv zugelassenen Apps."""
    statuses: list[AppClientStatus] = []
    for app_id in ALLOWED_CLIENT_APPS:
        rec = records.get(app_id) or {}
        has_tok = bool(str(rec.get("token_hash") or "").strip())
        statuses.append(
            AppClientStatus(
                app_id=app_id,
                display_name=CLIENT_APP_DISPLAY_NAMES.get(app_id, app_id),
                has_token=has_tok,
                enabled=bool(rec.get("enabled", False)) and has_tok,
                created_at=str(rec.get("created_at") or ""),
                last_used_at=str(rec.get("last_used_at") or ""),
            )
        )
    return statuses
