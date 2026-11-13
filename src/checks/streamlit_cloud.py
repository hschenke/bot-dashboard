"""Hilfsfunktionen für Streamlit Community Cloud Status-Abfragen und Wakeup-Signale."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import requests

from src.config.constants import HTTP_TIMEOUT_TUPLE
from src.domain.exceptions import SSRFBlockedError


def is_streamlit_cloud_url(url: str) -> bool:
    """Prüft, ob die URL auf Streamlit Community Cloud (`*.streamlit.app`) gehostet ist."""
    host = (urlparse(url.strip()).hostname or "").strip().lower().rstrip(".")
    return host.endswith(".streamlit.app") or host in {"streamlit.app", "share.streamlit.io"}


def _ensure_trailing_slash(app_url: str) -> str:
    base = app_url.strip()
    return base if base.endswith("/") else f"{base}/"


def build_streamlit_cloud_health_url(app_url: str) -> str:
    """Leitet den Container-Health-Endpunkt `/~/+/_stcore/health` für Streamlit Cloud ab."""
    return urljoin(_ensure_trailing_slash(app_url), "~/+/_stcore/health")


def build_streamlit_cloud_status_url(app_url: str) -> str:
    """Leitet die Streamlit Cloud App-Status-API (`/api/v2/app/status`) ab."""
    return urljoin(_ensure_trailing_slash(app_url), "api/v2/app/status")


def build_streamlit_cloud_resume_url(app_url: str) -> str:
    """Leitet den Streamlit Cloud Wakeup-Endpunkt (`/api/v2/app/resume`) ab."""
    return urljoin(_ensure_trailing_slash(app_url), "api/v2/app/resume")


def is_valid_streamlit_health_body(status_code: int, body_text: str) -> bool:
    """Bestätigt `/_stcore/health` nur bei echtem Plain-Text `ok` (niemals bei HTML-Wrappern)."""
    if status_code != 200:
        return False
    cleaned = body_text.strip().lower()
    if not cleaned or len(cleaned) > 128:
        return False
    if any(tag in cleaned for tag in ("<html", "<!doctype", "<script", "<meta", "<head")):
        return False
    if cleaned == "ok":
        return True
    try:
        parsed = json.loads(cleaned)
        return isinstance(parsed, dict) and str(parsed.get("status", "")).lower() == "ok"
    except ValueError:
        return False


def parse_streamlit_cloud_status(status_code: int, body_text: str) -> tuple[str | None, str]:
    """Interpretiert die JSON-Antwort von `/api/v2/app/status` (`running`, `asleep`, `error`, `None`)."""
    if status_code != 200:
        return None, ""
    try:
        payload = json.loads(body_text)
    except ValueError:
        return None, ""
    if not isinstance(payload, dict):
        return None, ""

    raw_status = payload.get("status", payload.get("state"))
    if raw_status is None:
        return None, ""

    upper_val = str(raw_status).strip().upper()
    if raw_status == 5 or upper_val == "RUNNING":
        return "running", f"Cloud-Status: RUNNING ({raw_status})"
    if upper_val in {"ERROR", "FAILED", "CRASHED"}:
        return "error", f"Cloud-Status: Fehler ({raw_status})"
    return "asleep", f"Cloud-Status: Ruhezustand/Inaktiv (status={raw_status})"


def wake_streamlit_cloud_app(
    app_url: str,
    session: requests.Session,
    validate_url: Callable[[str], str],
) -> tuple[bool, str]:
    """Sendet über `/api/v2/app/status` + `/api/v2/app/resume` ein Wakeup-Signal an Streamlit Cloud."""
    raw = app_url.strip()
    if not raw:
        return False, "Keine Streamlit-URL konfiguriert."
    if not is_streamlit_cloud_url(raw):
        return False, "API-Wakeup wird nur für Streamlit Community Cloud (*.streamlit.app) unterstützt."

    try:
        validated_base = validate_url(raw)
        parsed = urlparse(validated_base)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        status_url = validate_url(build_streamlit_cloud_status_url(validated_base))
        resume_url = validate_url(build_streamlit_cloud_resume_url(validated_base))

        status_resp = session.get(status_url, timeout=HTTP_TIMEOUT_TUPLE, allow_redirects=False)
        try:
            csrf_token = str(status_resp.headers.get("x-csrf-token", "")).strip()
            cookies = getattr(status_resp, "cookies", None)
        finally:
            status_resp.close()

        headers = {
            "x-streamlit-machine-id": str(uuid.uuid4()),
            "origin": origin,
            "referer": f"{origin}/",
        }
        if csrf_token:
            headers["x-csrf-token"] = csrf_token

        post_kwargs: dict[str, object] = {
            "headers": headers,
            "timeout": HTTP_TIMEOUT_TUPLE,
            "allow_redirects": False,
        }
        if cookies is not None:
            post_kwargs["cookies"] = cookies

        resume_resp = session.post(resume_url, **post_kwargs)  # type: ignore[arg-type]
        try:
            if 200 <= resume_resp.status_code < 300:
                return True, f"Wakeup-Signal gesendet (HTTP {resume_resp.status_code}). App fährt hoch."
            return False, f"Wakeup abgelehnt (HTTP {resume_resp.status_code})."
        finally:
            resume_resp.close()
    except (SSRFBlockedError, requests.RequestException) as exc:
        return False, f"Wakeup fehlgeschlagen: {exc}"
