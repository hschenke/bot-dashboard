"""Einheitliche Flotten-Zugriffskontrolle für das AI-Brain (1 Key für alle 3 Apps)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ai_brain.exceptions import BrainAuthError

if TYPE_CHECKING:
    from ai_brain.storage import BrainRepository

ALLOWED_CLIENT_APPS: tuple[str, ...] = ("news_bot", "research_bot", "alert_bot")
FLEET_CLIENT_ID: str = "fleet"
INTERNAL_DASHBOARD_APP: str = "bot_dashboard"
FLEET_DISPLAY_NAME: str = "Bot-Flotte (News-Bot, Research-Bot & Alert-Bot)"

FLEET_SECRET_BYTES: int = 16  # 128-Bit kryptographisches Geheimnis (kompakt & sicher)
_HASH_PREFIX = "sha256_v1:"


class FleetAccessStatus(BaseModel):
    """Sicherer Zugriffsstatus des gemeinsamen Flotten-Keys (enthält niemals Key oder Hash!)."""

    model_config = ConfigDict(frozen=True)

    client_id: str = FLEET_CLIENT_ID
    display_name: str = FLEET_DISPLAY_NAME
    has_key: bool = False
    enabled: bool = False
    created_at: str = ""
    last_used_at: str = ""


def _compute_salted_hash(raw_secret: bytes, salt: bytes) -> str:
    digest = hmac.new(salt, b"fleet:" + raw_secret, hashlib.sha256).hexdigest()
    return f"{_HASH_PREFIX}{salt.hex()}:{digest}"


def generate_fleet_access_secret() -> tuple[bytes, str]:
    """Erzeugt ein 128-Bit-Flotten-Geheimnis (16 Bytes) und seinen gesalzenen HMAC-Hash."""
    raw_secret = secrets.token_bytes(FLEET_SECRET_BYTES)
    salt = secrets.token_bytes(16)
    return raw_secret, _compute_salted_hash(raw_secret, salt)


def verify_fleet_access_secret(raw_secret: bytes, stored_hash: str) -> bool:
    """Prüft das Flotten-Geheimnis in konstanter Zeit gegen den gespeicherten Hash."""
    raw_hash = (stored_hash or "").strip()
    if len(raw_secret) != FLEET_SECRET_BYTES or not raw_hash.startswith(_HASH_PREFIX):
        return False
    parts = raw_hash.removeprefix(_HASH_PREFIX).split(":", 1)
    if len(parts) != 2:
        return False
    try:
        salt = bytes.fromhex(parts[0])
    except ValueError:
        return False
    expected = _compute_salted_hash(raw_secret, salt)
    return hmac.compare_digest(expected, raw_hash)


def authorize_fleet_client(repo: BrainRepository, raw_secret: bytes) -> str:
    """Verifiziert den gemeinsamen `AI_BRAIN_KEY` gegen die Brain-DB (fail-closed)."""
    if len(raw_secret) != FLEET_SECRET_BYTES:
        raise BrainAuthError("Zugriff verweigert: Kein gültiger AI_BRAIN_KEY angegeben.")
    record = repo.get_fleet_client_record()
    if not record or not record.get("token_hash"):
        raise BrainAuthError("Zugriff verweigert: Im AI-Brain Vault ist kein aktiver AI_BRAIN_KEY hinterlegt.")
    if not bool(record.get("enabled", False)):
        raise BrainAuthError("Zugriff verweigert: AI_BRAIN_KEY ist im AI-Brain Vault gesperrt.")
    if not verify_fleet_access_secret(raw_secret, str(record["token_hash"])):
        raise BrainAuthError("Zugriff verweigert: Ungültiger oder rotierter AI_BRAIN_KEY.")
    repo.touch_fleet_client_last_used()
    return FLEET_CLIENT_ID


def build_fleet_access_status(record: dict[str, Any] | None) -> FleetAccessStatus:
    """Erstellt den sicheren UI-Status des gemeinsamen Flotten-Zugangs."""
    rec = record or {}
    has_key = bool(str(rec.get("token_hash") or "").strip())
    return FleetAccessStatus(
        client_id=FLEET_CLIENT_ID,
        display_name=FLEET_DISPLAY_NAME,
        has_key=has_key,
        enabled=bool(rec.get("enabled", False)) and has_key,
        created_at=str(rec.get("created_at") or ""),
        last_used_at=str(rec.get("last_used_at") or ""),
    )
