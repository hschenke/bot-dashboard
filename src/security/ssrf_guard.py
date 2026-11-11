"""SSRF- & DNS-Rebinding-Schutz für alle ausgehenden HTTP/HTTPS-Anfragen."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

from src.config.constants import ALLOWED_HTTP_PORTS, ALLOWED_URL_SCHEMES
from src.domain.exceptions import SSRFBlockedError

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

DnsResolver = Callable[[str, int], list[str]]


def _default_dns_resolver(hostname: str, port: int) -> list[str]:
    """Löst einen Hostnamen über das Betriebssystem in alle Ziel-IP-Adressen auf."""
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFBlockedError(f"DNS-Auflösung für '{hostname}' fehlgeschlagen.") from exc

    resolved_ips = [
        info[4][0]
        for info in addr_infos
        if info[4] and isinstance(info[4][0], str)
    ]
    if not resolved_ips:
        raise SSRFBlockedError(f"Keine gültige IP-Adresse für '{hostname}' gefunden.")
    return resolved_ips


def is_public_ip_address(raw_ip: str) -> bool:
    """
    Prüft strikt, ob eine IPv4/IPv6-Adresse global öffentlich erreichbar ist.
    Blockiert Loopback (127.0.0.1, ::1), RFC-1918, Link-Local/Cloud-Metadata (169.254.169.254) & CGNAT.
    """
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


def _validate_url_structure(url: str) -> tuple[str, str, int]:
    """Validiert Schema, eingebettete Credentials, Hostnamen und Ziel-Port."""
    cleaned = url.strip()
    if not cleaned or len(cleaned) > 2048:
        raise SSRFBlockedError("Leere oder unzulässig lange Ziel-URL.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise SSRFBlockedError(f"Nicht erlaubtes URL-Schema '{scheme}'.")

    if parsed.username or parsed.password:
        raise SSRFBlockedError("Eingebettete Zugangsdaten in URLs sind nicht erlaubt.")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise SSRFBlockedError("Ziel-URL enthält keinen gültigen Hostnamen.")

    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(BLOCKED_HOST_SUFFIXES):
        raise SSRFBlockedError(f"Interner Hostname '{hostname}' ist blockiert.")

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise SSRFBlockedError("Ungültige Portangabe in Ziel-URL.") from exc

    if port not in ALLOWED_HTTP_PORTS:
        raise SSRFBlockedError(f"Nicht erlaubter Ziel-Port {port}.")

    return cleaned, hostname, port


def validate_safe_external_url(
    url: str,
    dns_resolver: DnsResolver = _default_dns_resolver,
) -> str:
    """Validiert eine externe URL gegen SSRF und DNS-Rebinding."""
    cleaned_url, hostname, port = _validate_url_structure(url)

    try:
        ipaddress.ip_address(hostname)
        if not is_public_ip_address(hostname):
            raise SSRFBlockedError(f"Zugriff auf nicht-öffentliche IP '{hostname}' verweigert.")
        return cleaned_url
    except ValueError:
        pass

    for ip_str in dns_resolver(hostname, port):
        if not is_public_ip_address(ip_str):
            raise SSRFBlockedError(
                f"Host '{hostname}' löst auf nicht-öffentliche IP '{ip_str}' auf."
            )
    return cleaned_url
