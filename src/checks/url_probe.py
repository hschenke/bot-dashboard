"""SSRF-geschützter HTTP- & Streamlit-Health-Check für gehostete Web-Apps."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from urllib.parse import urljoin

import requests

from src.config.constants import (
    HTTP_TIMEOUT_TUPLE,
    MAX_HTTP_REDIRECTS,
    MAX_RESPONSE_BYTES,
)
from src.domain.enums import HealthStatus
from src.domain.exceptions import SSRFBlockedError
from src.domain.models import ManagedAppSpec, UrlCheckResult
from src.security.ssrf_guard import DnsResolver, validate_safe_external_url

_SLEEP_MARKERS: tuple[str, ...] = (
    "get this app back up",
    "this app has gone to sleep",
    "waking up",
    "streamlit.io/cloud",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_streamlit_health_url(app_url: str) -> str:
    """Leitet aus der App-Basis-URL den offiziellen `/_stcore/health`-Endpunkt ab."""
    base = app_url.strip()
    if not base.endswith("/"):
        base += "/"
    return urljoin(base, "_stcore/health")


class SafeUrlProbe:
    """Prüft Erreichbarkeit, Antwortzeit und Schlafzustand von Streamlit Cloud Apps."""

    def __init__(
        self,
        session: requests.Session | None = None,
        dns_resolver: DnsResolver | None = None,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "BotDashboard-HealthProbe/0.1",
                "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
            }
        )
        self._dns_resolver = dns_resolver

    def check_app_url(self, spec: ManagedAppSpec) -> UrlCheckResult:
        """Prüft Haupt-URL und `/_stcore/health` einer überwachten Streamlit-App."""
        checked_at = _utc_now_iso()
        raw_url = spec.streamlit_url.strip()
        if not raw_url:
            return UrlCheckResult(
                app_id=spec.app_id,
                target_url="",
                health_endpoint_url="",
                status=HealthStatus.UNKNOWN,
                http_status_code=0,
                latency_ms=0,
                is_streamlit_awake=False,
                health_endpoint_ok=False,
                message="Keine Streamlit-URL konfiguriert.",
                checked_at=checked_at,
            )

        health_url = build_streamlit_health_url(raw_url)
        start_ts = time.perf_counter()
        try:
            status_code, body_text = self._fetch_redirect_safe(raw_url)
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
        except SSRFBlockedError as exc:
            return UrlCheckResult(
                app_id=spec.app_id,
                target_url=raw_url,
                health_endpoint_url=health_url,
                status=HealthStatus.OFFLINE,
                http_status_code=0,
                latency_ms=0,
                is_streamlit_awake=False,
                health_endpoint_ok=False,
                message=f"SSRF-Schutz hat URL blockiert: {exc}",
                checked_at=checked_at,
            )
        except requests.RequestException as exc:
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
            return UrlCheckResult(
                app_id=spec.app_id,
                target_url=raw_url,
                health_endpoint_url=health_url,
                status=HealthStatus.OFFLINE,
                http_status_code=0,
                latency_ms=latency_ms,
                is_streamlit_awake=False,
                health_endpoint_ok=False,
                message=f"Nicht erreichbar: {exc.__class__.__name__}",
                checked_at=checked_at,
            )

        health_ok = self._probe_health_endpoint(health_url)
        lower_body = body_text.lower()
        is_sleeping = any(marker in lower_body for marker in _SLEEP_MARKERS)

        if 200 <= status_code < 400 and not is_sleeping:
            return UrlCheckResult(
                app_id=spec.app_id,
                target_url=raw_url,
                health_endpoint_url=health_url,
                status=HealthStatus.ONLINE,
                http_status_code=status_code,
                latency_ms=latency_ms,
                is_streamlit_awake=True,
                health_endpoint_ok=health_ok,
                message=f"Online & wach (HTTP {status_code}, {latency_ms} ms)",
                checked_at=checked_at,
            )

        if is_sleeping or status_code == 303:
            return UrlCheckResult(
                app_id=spec.app_id,
                target_url=raw_url,
                health_endpoint_url=health_url,
                status=HealthStatus.ASLEEP,
                http_status_code=status_code,
                latency_ms=latency_ms,
                is_streamlit_awake=False,
                health_endpoint_ok=health_ok,
                message=f"App im Ruhezustand / Wakeup nötig (HTTP {status_code})",
                checked_at=checked_at,
            )

        return UrlCheckResult(
            app_id=spec.app_id,
            target_url=raw_url,
            health_endpoint_url=health_url,
            status=HealthStatus.DEGRADED if status_code < 500 else HealthStatus.OFFLINE,
            http_status_code=status_code,
            latency_ms=latency_ms,
            is_streamlit_awake=False,
            health_endpoint_ok=health_ok,
            message=f"HTTP-Status {status_code} ({latency_ms} ms)",
            checked_at=checked_at,
        )

    def _probe_health_endpoint(self, health_url: str) -> bool:
        try:
            code, text = self._fetch_redirect_safe(health_url)
            return code == 200 and ("ok" in text.lower() or len(text.strip()) > 0)
        except (SSRFBlockedError, requests.RequestException):
            return False

    def _validate_url(self, url: str) -> str:
        if self._dns_resolver is not None:
            return validate_safe_external_url(url, dns_resolver=self._dns_resolver)
        return validate_safe_external_url(url)

    def _fetch_redirect_safe(self, url: str) -> tuple[int, str]:
        current_url = self._validate_url(url)
        for _ in range(MAX_HTTP_REDIRECTS + 1):
            resp = self._session.get(
                current_url,
                timeout=HTTP_TIMEOUT_TUPLE,
                allow_redirects=False,
                stream=True,
            )
            try:
                if resp.is_redirect or resp.status_code in {301, 302, 303, 307, 308}:
                    location = resp.headers.get("Location", "").strip()
                    if not location:
                        return resp.status_code, ""
                    next_url = urljoin(current_url, location)
                    current_url = self._validate_url(next_url)
                    continue

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=16_384):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        chunks.append(chunk[: MAX_RESPONSE_BYTES - (total - len(chunk))])
                        break
                    chunks.append(chunk)
                body = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
                return resp.status_code, body
            finally:
                resp.close()

        raise SSRFBlockedError("Maximale Anzahl an HTTP-Redirects überschritten.")
