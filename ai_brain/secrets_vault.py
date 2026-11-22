"""Sichere Write-Only Secret-Verwaltung (wie GitHub Secrets) mit AEAD-Verschlüsselung."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

from ai_brain.exceptions import BrainValidationError

_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_ENC_V2_PREFIX = "enc_v2:"
_KDF_ITERATIONS = 120_000


def validate_secret_env_name(env_name: str) -> str:
    """Prüft den Namen eines Secret-Schlüssels gegen eine strikte Großbuchstaben-Whitelist."""
    cleaned = (env_name or "").strip().upper()
    if not _ENV_NAME_RE.match(cleaned):
        raise BrainValidationError(f"Ungültiger Secret-Name '{env_name}'.")
    return cleaned


def _hash_seed(parts: list[str]) -> bytes:
    raw_seed = "|".join(p for p in parts if p)
    return hashlib.sha256(raw_seed.encode("utf-8")).digest()


# Fester kryptographischer KDF-Domain-Separator (nicht ändern, damit bestehende Turso-Vault-Secrets lesbar bleiben).
_VAULT_KDF_DOMAIN: str = "ki-websearch-brain-vault-v2"


def _derive_master_material(passphrase: str = "") -> bytes:
    """Leitet app-übergreifendes Schlüsselmaterial ab (unabhängig vom Dashboard-APP_PASSWORD)."""
    explicit = passphrase.strip() or os.getenv("BRAIN_SECRET_KEY", "").strip()
    if explicit:
        return _hash_seed([explicit, _VAULT_KDF_DOMAIN])
    turso_seed = os.getenv("BRAIN_TURSO_KEY", "").strip() or os.getenv("TURSO_KEY", "").strip()
    if turso_seed:
        return _hash_seed([turso_seed, _VAULT_KDF_DOMAIN])
    return _hash_seed([_VAULT_KDF_DOMAIN])


def _keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0
    while sum(len(b) for b in blocks) < length:
        blocks.append(
            hmac.new(enc_key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        )
        counter += 1
    return b"".join(blocks)[:length]


def encrypt_secret_value(plaintext: str, passphrase: str = "") -> str:
    """Verschlüsselt einen Secret-Wert authentifiziert (`enc_v2`, CTR-HMAC-SHA256, 0 externe Deps)."""
    clean = (plaintext or "").strip()
    if not clean or len(clean) > 4096:
        raise BrainValidationError("Ungültiger oder leerer Secret-Wert.")
    nonce = os.urandom(16)
    master = _derive_master_material(passphrase)
    derived = hashlib.pbkdf2_hmac("sha256", master, nonce, _KDF_ITERATIONS, dklen=64)
    enc_key, mac_key = derived[:32], derived[32:]
    pt_bytes = clean.encode("utf-8")
    stream = _keystream(enc_key, nonce, len(pt_bytes))
    ciphertext = bytes(a ^ b for a, b in zip(pt_bytes, stream, strict=True))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(nonce + tag + ciphertext).decode("ascii")
    return f"{_ENC_V2_PREFIX}{token}"


def decrypt_secret_value(ciphertext_token: str, passphrase: str = "") -> str:
    """Entschlüsselt gespeicherte `enc_v2`-Tokens intern zur Laufzeit (niemals für UI!)."""
    raw = (ciphertext_token or "").strip()
    if not raw.startswith(_ENC_V2_PREFIX):
        return ""
    try:
        blob = base64.urlsafe_b64decode(raw.removeprefix(_ENC_V2_PREFIX).encode("ascii"))
    except Exception:
        return ""
    if len(blob) <= 48:
        return ""
    nonce, tag, ciphertext = blob[:16], blob[16:48], blob[48:]
    master = _derive_master_material(passphrase)
    derived = hashlib.pbkdf2_hmac("sha256", master, nonce, _KDF_ITERATIONS, dklen=64)
    enc_key, mac_key = derived[:32], derived[32:]
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        return ""
    stream = _keystream(enc_key, nonce, len(ciphertext))
    pt_bytes = bytes(a ^ b for a, b in zip(ciphertext, stream, strict=True))
    try:
        return pt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""
