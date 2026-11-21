"""Sichere Write-Only Secret-Verwaltung (wie GitHub Secrets) mit AEAD-Verschlüsselung."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

from ki_brain.exceptions import BrainValidationError

_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_ENC_V1_PREFIX = "enc_v1:"
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


def _derive_shared_master_material(passphrase: str = "") -> bytes:
    """Leitet app-übergreifendes Schlüsselmaterial ab (unabhängig vom Dashboard-APP_PASSWORD)."""
    explicit = passphrase.strip() or os.getenv("BRAIN_SECRET_KEY", "").strip()
    if explicit:
        return _hash_seed([explicit, "ki-websearch-brain-vault-v2"])
    turso_seed = os.getenv("BRAIN_TURSO_KEY", "").strip() or os.getenv("TURSO_KEY", "").strip()
    if turso_seed:
        return _hash_seed([turso_seed, "ki-websearch-brain-vault-v2"])
    return _derive_legacy_master_material(passphrase)


def _derive_legacy_master_material(passphrase: str = "") -> bytes:
    """Legacy-Ableitung (`enc_v1`) für 100 % Rückwärtskompatibilität bestehender Vault-Einträge."""
    seed_parts = [
        passphrase.strip(),
        os.getenv("BRAIN_SECRET_KEY", "").strip(),
        os.getenv("APP_PASSWORD", "").strip(),
        os.getenv("BRAIN_TURSO_KEY", "").strip(),
        os.getenv("TURSO_KEY", "").strip(),
        "ki-websearch-brain-vault-v1",
    ]
    return _hash_seed(seed_parts)


def _candidate_decryption_materials(prefix: str, passphrase: str = "") -> list[bytes]:
    shared = _derive_shared_master_material(passphrase)
    legacy = _derive_legacy_master_material(passphrase)
    legacy_env_only = _derive_legacy_master_material("")
    turso_only = _hash_seed(
        [
            os.getenv("BRAIN_TURSO_KEY", "").strip() or os.getenv("TURSO_KEY", "").strip(),
            "ki-websearch-brain-vault-v2",
        ]
    )
    ordered = (
        [shared, legacy, legacy_env_only, turso_only]
        if prefix == _ENC_V2_PREFIX
        else [legacy, legacy_env_only, shared, turso_only]
    )
    unique: list[bytes] = []
    for mat in ordered:
        if mat not in unique:
            unique.append(mat)
    return unique


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
    master = _derive_shared_master_material(passphrase)
    derived = hashlib.pbkdf2_hmac("sha256", master, nonce, _KDF_ITERATIONS, dklen=64)
    enc_key, mac_key = derived[:32], derived[32:]
    pt_bytes = clean.encode("utf-8")
    stream = _keystream(enc_key, nonce, len(pt_bytes))
    ciphertext = bytes(a ^ b for a, b in zip(pt_bytes, stream, strict=True))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(nonce + tag + ciphertext).decode("ascii")
    return f"{_ENC_V2_PREFIX}{token}"


def decrypt_secret_value(ciphertext_token: str, passphrase: str = "") -> str:
    """Entschlüsselt gespeicherte `enc_v2`/`enc_v1`-Tokens intern zur Laufzeit (niemals für UI!)."""
    raw = (ciphertext_token or "").strip()
    if raw.startswith(_ENC_V2_PREFIX):
        prefix = _ENC_V2_PREFIX
    elif raw.startswith(_ENC_V1_PREFIX):
        prefix = _ENC_V1_PREFIX
    else:
        return ""
    try:
        blob = base64.urlsafe_b64decode(raw.removeprefix(prefix).encode("ascii"))
    except Exception:
        return ""
    if len(blob) <= 48:
        return ""
    nonce, tag, ciphertext = blob[:16], blob[16:48], blob[48:]
    for master in _candidate_decryption_materials(prefix, passphrase):
        derived = hashlib.pbkdf2_hmac("sha256", master, nonce, _KDF_ITERATIONS, dklen=64)
        enc_key, mac_key = derived[:32], derived[32:]
        expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            continue
        stream = _keystream(enc_key, nonce, len(ciphertext))
        pt_bytes = bytes(a ^ b for a, b in zip(ciphertext, stream, strict=True))
        try:
            return pt_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return ""
