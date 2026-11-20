"""Browser-Kontext-Fingerprinting und WebCrypto ECDSA P-256 Proof-of-Possession."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from collections.abc import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

_FP_HMAC_DOMAIN = "bot_dashboard_fp_v2"
_P256_SIG_BYTES = 64
_MAX_B64_FIELD_LEN = 512


def _b64url_decode(raw: str) -> bytes:
    cleaned = (raw or "").strip()
    if not cleaned or len(cleaned) > _MAX_B64_FIELD_LEN:
        return b""
    padded = cleaned + "=" * (-len(cleaned) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, binascii.Error, UnicodeEncodeError):
        return b""


def _lookup_header(headers: Mapping[str, str] | None, target: str) -> str:
    if not headers:
        return ""
    direct = headers.get(target)
    if direct is not None:
        return str(direct).strip()
    lowered = target.lower()
    for key, val in headers.items():
        if str(key).lower() == lowered:
            return str(val).strip()
    return ""


def _detect_os_family(user_agent: str, sec_platform: str) -> str:
    combined = f"{sec_platform} {user_agent}".lower()
    for marker, label in (
        ("iphone", "ios"),
        ("ipad", "ios"),
        ("ipod", "ios"),
        ("ios", "ios"),
        ("android", "android"),
        ("windows", "windows"),
        ("macintosh", "macos"),
        ("mac os x", "macos"),
        ("macos", "macos"),
        ("cros", "chromeos"),
        ("linux", "linux"),
    ):
        if marker in combined:
            return label
    return "other_os" if combined.strip() else ""


def _detect_browser_family(user_agent: str) -> str:
    ua = user_agent.lower()
    for marker, label in (
        ("edg/", "edge"),
        ("edgios/", "edge"),
        ("edga/", "edge"),
        ("opr/", "opera"),
        ("opera", "opera"),
        ("samsungbrowser/", "samsung"),
        ("firefox/", "firefox"),
        ("fxios/", "firefox"),
        ("chrome/", "chrome"),
        ("crios/", "chrome"),
        ("safari/", "safari"),
        ("curl/", "curl"),
        ("python", "python"),
    ):
        if marker in ua:
            return label
    return "other_browser" if ua.strip() else ""


def compute_client_fingerprint(
    headers: Mapping[str, str] | None,
    locale: str | None = None,
) -> str:
    """Erstellt ein stabiles, browser- und OS-gebundenes Kontext-Merkmal."""
    ua = _lookup_header(headers, "user-agent")
    sec_platform = _lookup_header(headers, "sec-ch-ua-platform").strip('" ')
    sec_mobile = _lookup_header(headers, "sec-ch-ua-mobile").lower()
    loc = (locale or "").strip().lower().split(",")[0].split("-")[0]

    if not ua and not sec_platform and not loc:
        return "no_client_context"

    os_fam = _detect_os_family(ua, sec_platform)
    br_fam = _detect_browser_family(ua)
    is_mobile = (
        "1"
        if sec_mobile == "?1" or any(tok in ua.lower() for tok in ("mobile", "iphone", "android", "ipad"))
        else "0"
    )
    return f"os={os_fam}|br={br_fam}|mob={is_mobile}|loc={loc}"


def compute_fingerprint_tag(secret: str, client_fingerprint: str) -> str:
    """Erzeugt einen 16-Zeichen-HMAC-Tag für das Client-Kontext-Merkmal."""
    clean_fp = (client_fingerprint or "").strip() or "no_client_context"
    return hmac.new(
        secret.encode("utf-8"),
        f"{_FP_HMAC_DOMAIN}:{clean_fp}".encode(),
        hashlib.sha256,
    ).hexdigest()[:16]


def _load_p256_public_key(spki_b64url: str) -> ec.EllipticCurvePublicKey | None:
    der_bytes = _b64url_decode(spki_b64url)
    if not der_bytes:
        return None
    try:
        pub = serialization.load_der_public_key(der_bytes)
    except ValueError:
        return None
    if not isinstance(pub, ec.EllipticCurvePublicKey) or not isinstance(pub.curve, ec.SECP256R1):
        return None
    return pub


def compute_device_key_tag(spki_b64url: str) -> str:
    """Berechnet den 32-Hex-Fingerprint eines gültigen ECDSA-P256-Public-Keys."""
    pub = _load_p256_public_key(spki_b64url)
    if pub is None:
        return ""
    canonical_der = pub.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(canonical_der).hexdigest()[:32]


def verify_device_pop_signature(spki_b64url: str, sig_b64url: str, message: str) -> bool:
    """Prüft eine WebCrypto ECDSA-SHA256-Signatur (IEEE P1363) gegen den Geräte-Key."""
    pub = _load_p256_public_key(spki_b64url)
    raw_sig = _b64url_decode(sig_b64url)
    if pub is None or len(raw_sig) != _P256_SIG_BYTES or not message:
        return False

    r = int.from_bytes(raw_sig[:32], "big")
    s = int.from_bytes(raw_sig[32:], "big")
    der_sig = encode_dss_signature(r, s)
    try:
        pub.verify(der_sig, message.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False
