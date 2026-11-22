"""Zentrale, benannte Konstanten ohne Magic Numbers oder Magic Strings."""

from __future__ import annotations

# HTTP & Netzwerk-Timeouts (DoS-Prävention)
HTTP_CONNECT_TIMEOUT_SECONDS: float = 4.0
HTTP_READ_TIMEOUT_SECONDS: float = 10.0
HTTP_TIMEOUT_TUPLE: tuple[float, float] = (
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_READ_TIMEOUT_SECONDS,
)
MAX_HTTP_REDIRECTS: int = 5
MAX_RESPONSE_BYTES: int = 512 * 1024  # 512 KB reichen für Health- & Startseiten-Checks

# Turso & Datenbank-Grenzwerte
TURSO_HTTP_TIMEOUT_SECONDS: float = 10.0
TURSO_BATCH_SIZE: int = 20
DEFAULT_HTTP_RETRIES: int = 2
RETRY_BACKOFF_BASE_SECONDS: float = 0.6
SQLITE_TIMEOUT_SECONDS: float = 10.0

# GitHub API Grenzwerte
GITHUB_API_TIMEOUT_SECONDS: float = 10.0

# SSRF-Whitelist
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})
ALLOWED_HTTP_PORTS: frozenset[int] = frozenset({80, 443, 8080, 8443})

# Whitelist erlaubter Tabellennamen für dynamische Zählabfragen (SQL-Injection-Schutz)
ALLOWED_INSPECTION_TABLES: frozenset[str] = frozenset(
    {
        "articles",
        "archived_articles",
        "briefings",
        "offline_articles",
        "settings_backups",
        "metadata",
        "research_jobs",
        "research_reports",
        "app_domain_config",
        "alert_rules",
        "alert_events",
        "alert_check_logs",
        "alert_seen_items",
    }
)

# Persistenz & Grenzwerte
DEFAULT_SQLITE_REL_PATH: str = "data/bot_dashboard.db"
DEFAULT_BRAIN_SQLITE_REL_PATH: str = "data/ai_brain.db"
MAX_RECENT_ITEMS_PER_APP: int = 8
MAX_CHECK_LOG_HISTORY: int = 60

# Auth, 30-Tage-Cookie & Brute-Force-Schutz
AUTH_COOKIE_NAME: str = "bot_dashboard_auth_token"
AUTH_COOKIE_MAX_AGE_SECONDS: int = 30 * 24 * 3600  # 30 Tage
ADMIN_TOKEN_MAX_AGE_SECONDS: int = 30 * 24 * 3600  # 30 Tage
AUTH_SESSION_CACHE_TTL_SECONDS: float = 10.0
CLOCK_SKEW_TOLERANCE_SECONDS: int = 120
MAX_FAILED_LOGIN_ATTEMPTS: int = 5
LOGIN_LOCKOUT_SECONDS: int = 300
