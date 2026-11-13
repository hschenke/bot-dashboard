"""SSRF-geschützter HTTP- & Streamlit-Health-Check für gehostete Web-Apps."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from urllib.parse import urljoin

import requests

from src.checks.streamlit_cloud import (
    build_streamlit_cloud_health_url,
    build_streamlit_cloud_status_url,
    is_streamlit_cloud_url,
    is_valid_streamlit_health_body,
    parse_streamlit_cloud_status,
    wake_streamlit_cloud_app,
)
from src.config.constants import HTTP_TIMEOUT_TUPLE, MAX_HTTP_REDIRECTS, MAX_RESPONSE_BYTES
from src.domain.enums import AppId, HealthStatus
from src.domain.exceptions import SSRFBlockedError
from src.domain.models import ManagedAppSpec, UrlCheckResult
from src.security.ssrf_guard import DnsResolver, validate_safe_external_url

_SLEEP_MARKERS: tuple[str, ...] = (
    "get this app back up",
    "this app has gone to sleep",
    "waking up",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_streamlit_health_url(app_url: str) -> str:
    """Leitet aus der App-Basis-URL den offiziellen `/_stcore/health`-Endpunkt ab."""
    base = app_url.strip()
    if not base.endswith("/"):
        base += "/"
    return urljoin(base, "_stcore/health")


def _classify_url_check(
    *,
    app_id: AppId | str,
    raw_url: str,
    health_url: str,
    status_code: int,
    latency_ms: int,
    is_sleeping_html: bool,
    health_ok: bool,
    is_cloud: bool,
    cloud_state: str | None,
    cloud_detail: str,
    checked_at: str,
) -> UrlCheckResult:
    """Klassifiziert HTTP-, `/_stcore/health`- und Streamlit-Cloud-Signale präzise."""
    status = HealthStatus.OFFLINE
    awake = False
    if is_sleeping_html or status_code == 303 or cloud_state == "asleep":
        status = HealthStatus.ASLEEP
        msg = cloud_detail or f"App im Ruhezustand / Wakeup nötig (HTTP {status_code})"
    elif is_cloud and not health_ok:
        if cloud_state == "error":
            status = HealthStatus.OFFLINE
            msg = cloud_detail or f"Streamlit Cloud Fehlerstatus (HTTP {status_code})"
        elif cloud_state == "running":
            status = HealthStatus.DEGRADED
            msg = "Container antwortet nicht auf /_stcore/health (Cloud-Status: RUNNING)"
        else:
            status = HealthStatus.ASLEEP
            msg = f"App im Ruhezustand oder /_stcore/health inaktiv (HTTP {status_code})"
    elif 200 <= status_code < 400 and health_ok:
        status, awake = HealthStatus.ONLINE, True
        msg = f"Online & wach (HTTP {status_code}, {latency_ms} ms)"
    elif 200 <= status_code < 400:
        status = HealthStatus.DEGRADED
        msg = f"HTTP {status_code}, aber /_stcore/health nicht bestätigt ({latency_ms} ms)"
    else:
        status = HealthStatus.DEGRADED if status_code < 500 else HealthStatus.OFFLINE
        msg = f"HTTP-Status {status_code} ({latency_ms} ms)"

    return UrlCheckResult(
        app_id=app_id,
        target_url=raw_url,
        health_endpoint_url=health_url,
        status=status,
        http_status_code=status_code,
        latency_ms=latency_ms,
        is_streamlit_awake=awake,
        health_endpoint_ok=health_ok,
        message=msg,
        checked_at=checked_at,
    )


class SafeUrlProbe:
    """Prüft Erreichbarkeit, Antwortzeit, Schlafzustand & Wakeup von Streamlit Cloud Apps."""

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
        """Prüft Haupt-URL, Cloud-Status und `/_stcore/health` einer überwachten App."""
        return self.check_url(spec.streamlit_url, app_id=spec.app_id)

    def check_url(self, target_url: str, app_id: AppId | str = "bot_dashboard") -> UrlCheckResult:
        """Prüft eine beliebige Streamlit-App-URL inkl. Cloud-Schlafzustandserkennung."""
        checked_at = _utc_now_iso()
        raw_url = target_url.strip()
        if not raw_url:
            return UrlCheckResult(
                app_id=app_id,
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

        is_cloud = is_streamlit_cloud_url(raw_url)
        default_health = (
            build_streamlit_cloud_health_url(raw_url) if is_cloud else build_streamlit_health_url(raw_url)
        )
        start_ts = time.perf_counter()
        try:
            status_code, body_text = self._fetch_redirect_safe(raw_url)
            latency_ms = int((time.perf_counter() - start_ts) * 1000)
        except (SSRFBlockedError, requests.RequestException) as exc:
            latency_ms = 0 if isinstance(exc, SSRFBlockedError) else int((time.perf_counter() - start_ts) * 1000)
            msg = (
                f"SSRF-Schutz hat URL blockiert: {exc}"
                if isinstance(exc, SSRFBlockedError)
                else f"Nicht erreichbar: {exc.__class__.__name__}"
            )
            return UrlCheckResult(
                app_id=app_id,
                target_url=raw_url,
                health_endpoint_url=default_health,
                status=HealthStatus.OFFLINE,
                http_status_code=0,
                latency_ms=latency_ms,
                is_streamlit_awake=False,
                health_endpoint_ok=False,
                message=msg,
                checked_at=checked_at,
            )

        health_ok, effective_health = self._probe_health_endpoint_for_app(raw_url, is_cloud)
        cloud_state, cloud_detail = self._probe_cloud_status(raw_url) if is_cloud else (None, "")
        is_sleeping_html = any(marker in body_text.lower() for marker in _SLEEP_MARKERS)
        return _classify_url_check(
            app_id=app_id,
            raw_url=raw_url,
            health_url=effective_health,
            status_code=status_code,
            latency_ms=latency_ms,
            is_sleeping_html=is_sleeping_html,
            health_ok=health_ok,
            is_cloud=is_cloud,
            cloud_state=cloud_state,
            cloud_detail=cloud_detail,
            checked_at=checked_at,
        )

    def wake_app(self, target_url: str) -> tuple[bool, str]:
        """Weckt eine schlafende Streamlit Community Cloud App per Resume-API auf."""
        return wake_streamlit_cloud_app(target_url, self._session, self._validate_url)

    def _probe_health_endpoint_for_app(self, raw_url: str, is_cloud: bool) -> tuple[bool, str]:
        if is_cloud:
            cloud_health = build_streamlit_cloud_health_url(raw_url)
            if self._probe_health_endpoint(cloud_health):
                return True, cloud_health
            std_health = build_streamlit_health_url(raw_url)
            if self._probe_health_endpoint(std_health):
                return True, std_health
            return False, cloud_health

        std_health = build_streamlit_health_url(raw_url)
        return self._probe_health_endpoint(std_health), std_health

    def _probe_health_endpoint(self, health_url: str) -> bool:
        try:
            code, text = self._fetch_redirect_safe(health_url)
            return is_valid_streamlit_health_body(code, text)
        except (SSRFBlockedError, requests.RequestException):
            return False

    def _probe_cloud_status(self, raw_url: str) -> tuple[str | None, str]:
        try:
            code, text = self._fetch_redirect_safe(build_streamlit_cloud_status_url(raw_url))
            return parse_streamlit_cloud_status(code, text)
        except (SSRFBlockedError, requests.RequestException):
            return None, ""

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
                    current_url = self._validate_url(urljoin(current_url, location))
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
                return resp.status_code, b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
            finally:
                resp.close()

        raise SSRFBlockedError("Maximale Anzahl an HTTP-Redirects überschritten.")
