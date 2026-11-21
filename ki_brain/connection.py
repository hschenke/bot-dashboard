"""1-Secret Plug-&-Play Verbindungs-Bundle (`KI_BRAIN_KEY`) & Auto-Secret-Auflösung."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ki_brain.access import normalize_client_app_id
from ki_brain.exceptions import BrainAuthError

logger = logging.getLogger(__name__)

CONNECTION_KEY_ENV: str = "KI_BRAIN_KEY"
CONNECTION_KEY_PREFIX: str = "kibk1_"
_MAX_CONNECTION_KEY_LEN: int = 8192


@dataclass(frozen=True)
class BrainConnectionConfig:
    """Aufgelöste, sichere Verbindungsparameter für eine autorisierte Client-App."""

    app_id: str
    app_token: str = field(repr=False)
    turso_url: str = ""
    turso_auth_token: str = field(default="", repr=False)
    vault_passphrase: str = field(default="", repr=False)

    @property
    def effective_vault_passphrase(self) -> str:
        """Liefert das isolierte Entschlüsselungsmaterial (nie das fremde App-TURSO_KEY)."""
        return self.vault_passphrase.strip() or self.turso_auth_token.strip()


def encode_connection_key(
    app_id: str,
    app_token: str,
    *,
    turso_url: str = "",
    turso_auth_token: str = "",
    vault_passphrase: str = "",
) -> str:
    """Bündelt App-Token & Brain-DB-Zugang in einen einzelnen `KI_BRAIN_KEY`-String."""
    clean_app = normalize_client_app_id(app_id)
    clean_tok = (app_token or "").strip()
    if not clean_tok:
        raise BrainAuthError("Kein gültiges App-Access-Token zum Erstellen von KI_BRAIN_KEY angegeben.")
    payload: dict[str, Any] = {"v": 1, "a": clean_app, "t": clean_tok}
    if turso_url.strip():
        payload["u"] = turso_url.strip()
    if turso_auth_token.strip():
        payload["k"] = turso_auth_token.strip()
    if vault_passphrase.strip():
        payload["s"] = vault_passphrase.strip()
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{CONNECTION_KEY_PREFIX}{encoded}"


def decode_connection_key(raw_key: str) -> BrainConnectionConfig:
    """Dekodiert und validiert ein `KI_BRAIN_KEY`-Bundle (`kibk1_...`, fail-closed)."""
    cleaned = (raw_key or "").strip()
    if not cleaned.startswith(CONNECTION_KEY_PREFIX) or len(cleaned) > _MAX_CONNECTION_KEY_LEN:
        raise BrainAuthError("Ungültiges KI_BRAIN_KEY-Format (erwartet Präfix 'kibk1_').")
    body = cleaned.removeprefix(CONNECTION_KEY_PREFIX)
    padded = body + ("=" * ((4 - len(body) % 4) % 4))
    try:
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise BrainAuthError("KI_BRAIN_KEY konnte nicht dekodiert werden.") from exc
    if not isinstance(data, dict) or data.get("v") != 1:
        raise BrainAuthError("Nicht unterstützte KI_BRAIN_KEY-Version.")
    clean_app = normalize_client_app_id(str(data.get("a") or ""))
    clean_tok = str(data.get("t") or "").strip()
    if not clean_tok:
        raise BrainAuthError("KI_BRAIN_KEY enthält kein gültiges App-Token.")
    return BrainConnectionConfig(
        app_id=clean_app,
        app_token=clean_tok,
        turso_url=str(data.get("u") or "").strip(),
        turso_auth_token=str(data.get("k") or "").strip(),
        vault_passphrase=str(data.get("s") or "").strip(),
    )


def _lookup_in_mapping(mapping: Any, key: str) -> str | None:
    if not mapping or not hasattr(mapping, "get"):
        return None
    for cand in (key, key.lower()):
        val = mapping.get(cand)
        if val is not None and not hasattr(val, "get") and str(val).strip():
            return str(val).strip()
    parts = key.split("_")
    for idx in range(1, len(parts)):
        prefix, suffix = "_".join(parts[:idx]), "_".join(parts[idx:])
        for p_cand in (prefix, prefix.lower()):
            sec = mapping.get(p_cand)
            if hasattr(sec, "get"):
                for s_cand in (suffix, suffix.lower()):
                    s_val = sec.get(s_cand)
                    if s_val is not None and not hasattr(s_val, "get") and str(s_val).strip():
                        return str(s_val).strip()
    return None


def _read_streamlit_secret(key: str) -> str:
    if "STREAMLIT_SERVER_PORT" not in os.environ and "streamlit" not in sys.modules:
        return ""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            return _lookup_in_mapping(st.secrets, key) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Streamlit Secret '%s' nicht lesbar: %s", key, exc)
    return ""


def _read_local_secrets_toml(key: str, base_dir: Path | None = None) -> str:
    if base_dir is None and os.getenv("PYTEST_CURRENT_TEST"):
        return ""
    target = (base_dir or Path.cwd()) / ".streamlit" / "secrets.toml"
    if not target.is_file():
        return ""
    try:
        parsed = tomllib.loads(target.read_text(encoding="utf-8"))
        return _lookup_in_mapping(parsed, key) or "" if isinstance(parsed, dict) else ""
    except (OSError, tomllib.TOMLDecodeError):
        return ""


def get_runtime_secret(key: str, *, base_dir: Path | None = None) -> str:
    """Liest ein Secret automatisch aus ENV, `st.secrets` oder lokalem `.streamlit/secrets.toml`."""
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val
    st_val = _read_streamlit_secret(key)
    if st_val:
        return st_val
    return _read_local_secrets_toml(key, base_dir=base_dir)


def resolve_connection_config(
    *,
    connection_key: str = "",
    app_id: str = "",
    app_token: str = "",
    turso_url: str = "",
    turso_auth_token: str = "",
    vault_passphrase: str = "",
    base_dir: Path | None = None,
) -> BrainConnectionConfig:
    """Löst 1-Secret (`KI_BRAIN_KEY`) oder Legacy-Einzelvariablen fehlertolerant & isoliert auf."""
    raw_bundle = (
        connection_key.strip()
        or get_runtime_secret(CONNECTION_KEY_ENV, base_dir=base_dir)
        or get_runtime_secret("BRAIN_CONNECTION_KEY", base_dir=base_dir)
    )
    if raw_bundle:
        decoded = decode_connection_key(raw_bundle)
        return BrainConnectionConfig(
            app_id=app_id.strip() or decoded.app_id,
            app_token=app_token.strip() or decoded.app_token,
            turso_url=turso_url.strip() or decoded.turso_url,
            turso_auth_token=turso_auth_token.strip() or decoded.turso_auth_token,
            vault_passphrase=vault_passphrase.strip() or decoded.vault_passphrase,
        )

    resolved_app = app_id.strip() or get_runtime_secret("BRAIN_APP_ID", base_dir=base_dir)
    resolved_tok = app_token.strip() or get_runtime_secret("BRAIN_APP_TOKEN", base_dir=base_dir)
    if resolved_tok.startswith(CONNECTION_KEY_PREFIX):
        decoded = decode_connection_key(resolved_tok)
        return BrainConnectionConfig(
            app_id=resolved_app or decoded.app_id,
            app_token=decoded.app_token,
            turso_url=turso_url.strip() or decoded.turso_url,
            turso_auth_token=turso_auth_token.strip() or decoded.turso_auth_token,
            vault_passphrase=vault_passphrase.strip() or decoded.vault_passphrase,
        )

    brain_url = turso_url.strip() or get_runtime_secret("BRAIN_TURSO_URL", base_dir=base_dir)
    brain_key = turso_auth_token.strip() or get_runtime_secret("BRAIN_TURSO_KEY", base_dir=base_dir)
    resolved_url = brain_url or os.getenv("TURSO_URL", "").strip()
    resolved_key = brain_key or os.getenv("TURSO_KEY", "").strip()
    resolved_pass = vault_passphrase.strip() or get_runtime_secret("BRAIN_SECRET_KEY", base_dir=base_dir)
    return BrainConnectionConfig(
        app_id=resolved_app,
        app_token=resolved_tok,
        turso_url=resolved_url,
        turso_auth_token=resolved_key,
        vault_passphrase=resolved_pass,
    )


def is_brain_configured(
    connection_key: str = "",
    *,
    base_dir: Path | None = None,
) -> bool:
    """Prüft schnell & ohne DB-Aufruf, ob `KI_BRAIN_KEY` oder Legacy-Brain-Auth gesetzt ist."""
    raw_bundle = (
        connection_key.strip()
        or get_runtime_secret(CONNECTION_KEY_ENV, base_dir=base_dir)
        or get_runtime_secret("BRAIN_CONNECTION_KEY", base_dir=base_dir)
    )
    if raw_bundle:
        try:
            decode_connection_key(raw_bundle)
            return True
        except BrainAuthError:
            return False
    app_id = get_runtime_secret("BRAIN_APP_ID", base_dir=base_dir)
    app_tok = get_runtime_secret("BRAIN_APP_TOKEN", base_dir=base_dir)
    if not (app_id and app_tok):
        return False
    try:
        normalize_client_app_id(app_id)
        return True
    except BrainAuthError:
        return False
