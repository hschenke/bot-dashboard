"""Kompakter 1-Secret Plug-&-Play Flotten-Key (`AI_BRAIN_KEY`) & Auto-Secret-Auflösung."""

from __future__ import annotations

import base64
import logging
import os
import sys
import tomllib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_brain.access import FLEET_SECRET_BYTES
from ai_brain.exceptions import BrainAuthError

logger = logging.getLogger(__name__)

CONNECTION_KEY_ENV: str = "AI_BRAIN_KEY"
CONNECTION_KEY_PREFIX: str = "aib_"
_MAX_CONNECTION_KEY_LEN: int = 4096
_MAX_UNCOMPRESSED_BYTES: int = 4096
_STD_TURSO_JWT_HDR: str = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9"
_ED25519_SIG_BYTES: int = 64

_VERSION_MASK: int = 0xE0
_VERSION_1: int = 0x20
_FLAG_TURSO_SUFFIX: int = 0x01
_FLAG_COMPACT_JWT: int = 0x02
_FLAG_HAS_VAULT_SECRET: int = 0x04
_FLAG_DEFLATED: int = 0x08
_FLAG_HAS_TURSO: int = 0x10


@dataclass(frozen=True)
class BrainConnectionConfig:
    """Aufgelöste, sichere Verbindungsparameter aus dem kompakten `AI_BRAIN_KEY`."""

    fleet_secret: bytes = field(repr=False)
    turso_url: str = ""
    turso_auth_token: str = field(default="", repr=False)
    vault_passphrase: str = field(default="", repr=False)

    @property
    def effective_vault_passphrase(self) -> str:
        """Liefert das isolierte Entschlüsselungsmaterial (nie das fremde App-TURSO_KEY)."""
        return self.vault_passphrase.strip() or self.turso_auth_token.strip()


def _b64u_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64u_decode(text: str) -> bytes:
    padded = text + ("=" * ((4 - len(text) % 4) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _try_split_eddsa_jwt(token: str) -> tuple[bytes, bytes] | None:
    """Dekodiert Standard-Turso-EdDSA-JWT verlustfrei in `(payload_raw, sig_64B)`."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _STD_TURSO_JWT_HDR:
        return None
    try:
        payload_raw = _b64u_decode(parts[1])
        sig_raw = _b64u_decode(parts[2])
    except Exception:  # noqa: BLE001
        return None
    if len(sig_raw) != _ED25519_SIG_BYTES:
        return None
    if _b64u_encode(payload_raw) != parts[1] or _b64u_encode(sig_raw) != parts[2]:
        return None
    return payload_raw, sig_raw


def _normalize_compact_host(turso_url: str) -> tuple[bytes, bool]:
    clean = turso_url.strip().rstrip("/")
    if clean.startswith("libsql://"):
        clean = "https://" + clean.removeprefix("libsql://")
    if clean.startswith("https://"):
        host = clean.removeprefix("https://")
        if host.endswith(".turso.io") and "/" not in host:
            return host.removesuffix(".turso.io").encode("utf-8"), True
        return host.encode("utf-8"), False
    return clean.encode("utf-8"), False


def _reconstruct_turso_url(host: str, has_suffix: bool) -> str:
    if has_suffix:
        return f"https://{host}.turso.io"
    if host.startswith(("http://", "https://")):
        return host
    return f"https://{host}"


def encode_connection_key(
    fleet_secret: bytes,
    *,
    turso_url: str = "",
    turso_auth_token: str = "",
    vault_passphrase: str = "",
) -> str:
    """Kodiert Flotten-Secret & Brain-DB-Zugang kompakt als `aib_<base64url>`."""
    if len(fleet_secret) != FLEET_SECRET_BYTES:
        raise BrainAuthError("Ungültiges Flotten-Secret zum Erstellen von AI_BRAIN_KEY.")

    flags = _VERSION_1
    clean_url, clean_tok = turso_url.strip(), turso_auth_token.strip()
    clean_pass = vault_passphrase.strip()
    has_turso = bool(clean_url and clean_tok)
    if not has_turso and not clean_pass:
        return f"{CONNECTION_KEY_PREFIX}{_b64u_encode(bytes([flags]) + fleet_secret)}"

    text_parts = bytearray()
    sig_trailer = b""
    if has_turso:
        flags |= _FLAG_HAS_TURSO
        host_bytes, has_suffix = _normalize_compact_host(clean_url)
        if not host_bytes or len(host_bytes) > 255:
            raise BrainAuthError("Ungültige Turso-URL für AI_BRAIN_KEY.")
        if has_suffix:
            flags |= _FLAG_TURSO_SUFFIX
        text_parts.append(len(host_bytes))
        text_parts.extend(host_bytes)
        jwt_parts = _try_split_eddsa_jwt(clean_tok)
        if jwt_parts is not None:
            flags |= _FLAG_COMPACT_JWT
            payload_raw, sig_trailer = jwt_parts
            text_parts.extend(len(payload_raw).to_bytes(2, "big"))
            text_parts.extend(payload_raw)
        else:
            raw_tok = clean_tok.encode("utf-8")
            text_parts.extend(len(raw_tok).to_bytes(2, "big"))
            text_parts.extend(raw_tok)

    if clean_pass:
        flags |= _FLAG_HAS_VAULT_SECRET
        pass_bytes = clean_pass.encode("utf-8")
        text_parts.extend(len(pass_bytes).to_bytes(2, "big"))
        text_parts.extend(pass_bytes)

    raw_text = bytes(text_parts)
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(raw_text) + compressor.flush()
    if len(compressed) < len(raw_text):
        flags |= _FLAG_DEFLATED
        head_payload = compressed
    else:
        head_payload = raw_text

    blob = bytes([flags]) + len(head_payload).to_bytes(2, "big") + head_payload + fleet_secret + sig_trailer
    return f"{CONNECTION_KEY_PREFIX}{_b64u_encode(blob)}"


def _read_u16_slice(buf: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 2 > len(buf):
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY (Längenfeld abgeschnitten).")
    length = int.from_bytes(buf[offset : offset + 2], "big")
    end = offset + 2 + length
    if end > len(buf):
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY (Nutzdaten abgeschnitten).")
    return buf[offset + 2 : end], end


def _decompress_safe(data: bytes) -> bytes:
    decomp = zlib.decompressobj(wbits=-15)
    try:
        out = decomp.decompress(data, _MAX_UNCOMPRESSED_BYTES)
        tail = decomp.flush(_MAX_UNCOMPRESSED_BYTES - len(out) + 1)
    except zlib.error as exc:
        raise BrainAuthError("AI_BRAIN_KEY konnte nicht dekomprimiert werden.") from exc
    if decomp.unconsumed_tail or len(out) + len(tail) > _MAX_UNCOMPRESSED_BYTES:
        raise BrainAuthError("AI_BRAIN_KEY überschreitet das zulässige Größenlimit.")
    return out + tail


def decode_connection_key(raw_key: str) -> BrainConnectionConfig:
    """Dekodiert und validiert einen kompakten `AI_BRAIN_KEY` (`aib_...`, fail-closed)."""
    cleaned = (raw_key or "").strip()
    if not cleaned.startswith(CONNECTION_KEY_PREFIX) or len(cleaned) > _MAX_CONNECTION_KEY_LEN:
        raise BrainAuthError("Ungültiges AI_BRAIN_KEY-Format (erwartet Präfix 'aib_').")
    try:
        blob = _b64u_decode(cleaned.removeprefix(CONNECTION_KEY_PREFIX))
    except Exception as exc:  # noqa: BLE001
        raise BrainAuthError("AI_BRAIN_KEY konnte nicht dekodiert werden.") from exc
    if not blob:
        raise BrainAuthError("Leerer AI_BRAIN_KEY.")

    flags = blob[0]
    if (flags & _VERSION_MASK) != _VERSION_1:
        raise BrainAuthError("Nicht unterstützte AI_BRAIN_KEY-Version.")
    has_turso = bool(flags & _FLAG_HAS_TURSO)
    has_secret = bool(flags & _FLAG_HAS_VAULT_SECRET)
    compact_jwt = bool(flags & _FLAG_COMPACT_JWT)

    if not has_turso and not has_secret:
        if len(blob) != 1 + FLEET_SECRET_BYTES:
            raise BrainAuthError("Ungültige Länge für lokalen AI_BRAIN_KEY.")
        return BrainConnectionConfig(fleet_secret=blob[1:])

    if len(blob) < 3:
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY-Header.")
    head_len = int.from_bytes(blob[1:3], "big")
    trailer_len = FLEET_SECRET_BYTES + (_ED25519_SIG_BYTES if (has_turso and compact_jwt) else 0)
    if len(blob) != 3 + head_len + trailer_len:
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY (inkonsistente Länge).")

    head_raw = blob[3 : 3 + head_len]
    text_buf = _decompress_safe(head_raw) if (flags & _FLAG_DEFLATED) else head_raw
    sec_start = 3 + head_len
    fleet_secret = blob[sec_start : sec_start + FLEET_SECRET_BYTES]
    sig_bytes = blob[sec_start + FLEET_SECRET_BYTES :]

    pos = 0
    turso_url, turso_token, vault_pass = "", "", ""
    try:
        if has_turso:
            if pos >= len(text_buf):
                raise BrainAuthError("Beschädigter AI_BRAIN_KEY (Host fehlt).")
            host_len = text_buf[pos]
            pos += 1
            if pos + host_len > len(text_buf):
                raise BrainAuthError("Beschädigter AI_BRAIN_KEY (Host abgeschnitten).")
            host = text_buf[pos : pos + host_len].decode("utf-8", errors="strict")
            pos += host_len
            turso_url = _reconstruct_turso_url(host, bool(flags & _FLAG_TURSO_SUFFIX))
            tok_part, pos = _read_u16_slice(text_buf, pos)
            if compact_jwt:
                turso_token = f"{_STD_TURSO_JWT_HDR}.{_b64u_encode(tok_part)}.{_b64u_encode(sig_bytes)}"
            else:
                turso_token = tok_part.decode("utf-8", errors="strict")

        if has_secret:
            sec_part, pos = _read_u16_slice(text_buf, pos)
            vault_pass = sec_part.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY (ungültige Textkodierung).") from exc
    if pos != len(text_buf):
        raise BrainAuthError("Beschädigter AI_BRAIN_KEY (unerwartete Restdaten).")

    return BrainConnectionConfig(
        fleet_secret=fleet_secret,
        turso_url=turso_url,
        turso_auth_token=turso_token,
        vault_passphrase=vault_pass,
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
    base_dir: Path | None = None,
) -> BrainConnectionConfig:
    """Löst den 1-Secret `AI_BRAIN_KEY` automatisch aus Argument, ENV, `st.secrets` oder TOML auf."""
    raw_key = connection_key.strip() or get_runtime_secret(CONNECTION_KEY_ENV, base_dir=base_dir)
    if not raw_key:
        raise BrainAuthError("Kein AI_BRAIN_KEY in Umgebungsvariablen oder Streamlit Secrets gefunden.")
    return decode_connection_key(raw_key)


def is_brain_configured(
    connection_key: str = "",
    *,
    base_dir: Path | None = None,
) -> bool:
    """Prüft schnell & ohne DB-Aufruf, ob ein gültiger `AI_BRAIN_KEY` gesetzt ist."""
    raw_key = connection_key.strip() or get_runtime_secret(CONNECTION_KEY_ENV, base_dir=base_dir)
    if not raw_key:
        return False
    try:
        decode_connection_key(raw_key)
        return True
    except BrainAuthError:
        return False
