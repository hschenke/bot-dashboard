"""Eigenständiger SSRF-, DNS-Rebinding-, Pfad- und URL-Schutz für ki_brain."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import requests

from ki_brain.exceptions import BrainSSRFError, BrainValidationError

ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
ALLOWED_PORTS: frozenset[int] = frozenset({80, 443, 8080, 8443})
BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "ip6-localhost",
        "ip6-loopback",
    }
)
BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (".local", ".internal", ".localhost", ".lan", ".arpa")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{1,95}$")

DnsResolver = Callable[[str, int], list[str]]


def _default_dns_resolver(hostname: str, port: int) -> list[str]:
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BrainSSRFError(f"DNS-Auflösung für '{hostname}' fehlgeschlagen.") from exc
    resolved = [info[4][0] for info in addr_infos if info[4] and isinstance(info[4][0], str)]
    if not resolved:
        raise BrainSSRFError(f"Keine gültige IP-Adresse für '{hostname}' gefunden.")
    return resolved


def is_public_ip_address(raw_ip: str) -> bool:
    """Prüft strikt auf global öffentliche IPv4/IPv6-Adresse (blockiert Private/Loopback/Metadata)."""
    clean_ip = raw_ip.split("%", 1)[0].strip()
    try:
        ip_obj = ipaddress.ip_address(clean_ip)
    except ValueError:
        return False
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped
    return bool(
        ip_obj.is_global
        and not ip_obj.is_loopback
        and not ip_obj.is_private
        and not ip_obj.is_link_local
        and not ip_obj.is_multicast
        and not ip_obj.is_reserved
        and not ip_obj.is_unspecified
    )


def validate_safe_external_url(
    url: str,
    dns_resolver: DnsResolver = _default_dns_resolver,
    resolve_dns: bool = True,
) -> str:
    """Validiert eine externe URL gegen SSRF, unsichere Schemata und DNS-Rebinding."""
    cleaned = (url or "").strip()
    if not cleaned or len(cleaned) > 2048:
        raise BrainSSRFError("Leere oder unzulässig lange Ziel-URL.")
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise BrainSSRFError(f"Nicht erlaubtes URL-Schema '{scheme}'.")
    if parsed.username or parsed.password:
        raise BrainSSRFError("Eingebettete Zugangsdaten in URLs sind nicht erlaubt.")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise BrainSSRFError("Ziel-URL enthält keinen gültigen Hostnamen.")
    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(BLOCKED_HOST_SUFFIXES):
        raise BrainSSRFError(f"Interner Hostname '{hostname}' ist blockiert.")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise BrainSSRFError("Ungültige Portangabe in Ziel-URL.") from exc
    if port not in ALLOWED_PORTS:
        raise BrainSSRFError(f"Nicht erlaubter Ziel-Port {port}.")

    try:
        ipaddress.ip_address(hostname)
        if not is_public_ip_address(hostname):
            raise BrainSSRFError(f"Zugriff auf nicht-öffentliche IP '{hostname}' verweigert.")
        return cleaned
    except ValueError:
        pass

    if resolve_dns:
        for ip_str in dns_resolver(hostname, port):
            if not is_public_ip_address(ip_str):
                raise BrainSSRFError(f"Host '{hostname}' löst auf nicht-öffentliche IP '{ip_str}' auf.")
    return cleaned


def sanitize_display_url(url: str) -> str:
    """Erlaubt ausschließlich sichere http/https-Links für Ausgaben und Zitate."""
    cleaned = (url or "").strip()
    if not cleaned:
        return "#"
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.netloc:
        return "#"
    return cleaned


def validate_identifier(raw_id: str, label: str = "ID") -> str:
    """Validiert eine Modell- oder Tool-Kennung gegen eine strikte Zeichen-Whitelist."""
    cleaned = (raw_id or "").strip().lower()
    if not _IDENTIFIER_RE.match(cleaned) or ".." in cleaned:
        raise BrainValidationError(f"Ungültige {label}: '{raw_id}'.")
    return cleaned


def resolve_safe_sqlite_path(base_dir: Path, raw_path: str) -> Path:
    """Löst einen lokalen SQLite-Dateipfad sicher gegen Path-Traversal auf."""
    cleaned = (raw_path or "").strip()
    if not cleaned:
        raise BrainValidationError("Leerer SQLite-Dateipfad.")
    anchor = base_dir.resolve()
    candidate = (base_dir / cleaned).resolve()
    if not candidate.is_relative_to(anchor):
        raise BrainValidationError(f"Pfad '{raw_path}' liegt außerhalb des erlaubten Verzeichnisses.")
    return candidate


def read_bounded_response(response: requests.Response, max_bytes: int = 1_500_000) -> bytes:
    """Liest HTTP-Response-Body speicherbegrenzt gegen DoS-Payloads."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=16_384):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            chunks.append(chunk[: max(0, max_bytes - (total - len(chunk)))])
            break
        chunks.append(chunk)
    return b"".join(chunks)
