"""Automatische Herkunfts-Erkennung (Wer/Wo) & datensparsame Zugriffs-Telemetrie für `ai_brain`."""

from __future__ import annotations

import datetime
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

from ai_brain.models import BrainAccessLogEntry

_KNOWN_CALLER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("alert-bot", "Alert-Bot"),
    ("alert_bot", "Alert-Bot"),
    ("news-bot", "News-Bot"),
    ("news_bot", "News-Bot"),
    ("news-aggregator", "News-Bot"),
    ("research-bot", "Research-Bot"),
    ("research_bot", "Research-Bot"),
    ("bot-dashboard", "Dashboard"),
    ("bot_dashboard", "Dashboard"),
)
_SECRET_SNIPPET_RE = re.compile(
    r"(?:(?:aib_|kib_)[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{12,}|gsk_[A-Za-z0-9_\-]{12,}|tvly-[A-Za-z0-9_\-]{10,}|Bearer\s+\S+)",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
MAX_LOG_SUMMARY_CHARS: int = 160


def normalize_caller_label(raw: str) -> str:
    """Ordnet einen rohen App-/Repo-/URL-Namen einem sauberen Anzeigenamen zu."""
    cleaned = _WS_RE.sub(" ", (raw or "").strip())
    if not cleaned:
        return ""
    lower = cleaned.lower()
    for needle, label in _KNOWN_CALLER_PATTERNS:
        if needle in lower:
            return label
    if "/" in cleaned and not cleaned.startswith(("http://", "https://")):
        cleaned = cleaned.rsplit("/", 1)[-1].strip()
    return cleaned[:40]


def detect_caller_app(*, explicit_caller: str = "", is_admin: bool = False) -> str:
    """Erkennt automatisch, welche App (`Alert-Bot`, `News-Bot`, `Research-Bot`, `Dashboard`) zugreift."""
    if is_admin:
        return "Dashboard"
    for raw_explicit in (explicit_caller, os.getenv("BRAIN_CALLER", "")):
        label = normalize_caller_label(raw_explicit)
        if label:
            return label

    candidates = (
        os.getenv("GITHUB_REPOSITORY", ""),
        os.getenv("GITHUB_REPO", ""),
        os.getenv("STREAMLIT_APP_URL", ""),
        str(Path.cwd()),
    )
    for cand in candidates:
        lower = (cand or "").lower()
        for needle, label in _KNOWN_CALLER_PATTERNS:
            if needle in lower:
                return label

    gh_repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if gh_repo:
        return normalize_caller_label(gh_repo) or "Fleet-App"
    return "Fleet-App"


def detect_runtime_environment() -> str:
    """Erkennt die Laufzeitumgebung (`GitHub Actions`, `Streamlit Cloud`, `Streamlit (Local)`, `Local / CLI`)."""
    if os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true":
        wf = _WS_RE.sub(" ", os.getenv("GITHUB_WORKFLOW", "").strip())
        if wf and len(wf) <= 24:
            return f"GitHub Actions · {wf}"
        return "GitHub Actions"

    cwd_norm = str(Path.cwd()).replace("\\", "/").lower()
    if os.getenv("STREAMLIT_SHARING_MODE") or "/mount/src/" in cwd_norm:
        return "Streamlit Cloud"
    if "STREAMLIT_SERVER_PORT" in os.environ or "streamlit" in sys.modules:
        return "Streamlit (Local)"
    return "Local / CLI"


def sanitize_log_summary(raw_text: str, *, max_len: int = MAX_LOG_SUMMARY_CHARS) -> str:
    """Bereinigt Log-Details von Steuerzeichen und maskiert versehentliche Secret-Muster."""
    compact = _WS_RE.sub(" ", (raw_text or "").strip())
    if not compact:
        return ""
    masked = _SECRET_SNIPPET_RE.sub("[REDACTED]", compact)
    if len(masked) > max_len:
        return masked[: max_len - 1].rstrip() + "…"
    return masked


def sanitize_url_for_log(raw_url: str) -> str:
    """Kürzt eine URL auf Host + Pfad (entfernt Query-Strings mit möglichen Tokens)."""
    try:
        parsed = urlparse((raw_url or "").strip())
    except ValueError:
        return sanitize_log_summary(raw_url, max_len=110)
    if parsed.netloc:
        path = parsed.path if parsed.path and parsed.path != "/" else ""
        return sanitize_log_summary(f"{parsed.netloc}{path}", max_len=110)
    return sanitize_log_summary(raw_url, max_len=110)


def build_access_log_entry(
    *,
    caller_app: str,
    environment: str,
    action: str,
    status: str,
    tier: str = "",
    allow_paid: bool = False,
    latency_ms: int = 0,
    model_used: str = "",
    tools_used: list[str] | tuple[str, ...] = (),
    result_count: int = 0,
    summary: str = "",
) -> BrainAccessLogEntry:
    """Erstellt einen normalisierten, unveränderlichen Zugriffs-Log-Eintrag."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    clean_tools = [str(t).strip() for t in tools_used if str(t).strip()][:8]
    return BrainAccessLogEntry(
        id=f"{int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)}_{uuid.uuid4().hex[:8]}",
        created_at=now_iso,
        caller_app=caller_app[:40] or "Fleet-App",
        environment=environment[:48] or "Local / CLI",
        action=action[:32] or "unknown",
        tier=tier[:16],
        budget_mode="paid" if allow_paid else "free",
        status="Success" if status == "Success" else "Failed",
        latency_ms=max(0, int(latency_ms)),
        model_used=model_used[:80],
        tools_used=clean_tools,
        result_count=max(0, int(result_count)),
        summary=sanitize_log_summary(summary),
    )
