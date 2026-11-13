"""Unveränderliche Datenmodelle (`@dataclass(frozen=True)`) für alle Dashboard-Checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.enums import AppId, HealthStatus, WorkflowRunState


@dataclass(frozen=True)
class ManagedAppSpec:
    """Deklarative Definition und Verbindungsdaten einer überwachten Streamlit-App."""

    app_id: AppId
    title: str
    icon: str
    description: str
    streamlit_url: str
    github_repo: str
    primary_workflow: str
    secondary_workflows: tuple[str, ...] = ()
    turso_url: str = ""
    turso_auth_token: str = field(default="", repr=False)
    local_sqlite_path: str = ""
    expected_tables: tuple[str, ...] = ()

    @property
    def has_turso_credentials(self) -> bool:
        return bool(self.turso_url.strip() and self.turso_auth_token.strip())

    @property
    def github_repo_url(self) -> str:
        clean = self.github_repo.strip().strip("/")
        return f"https://github.com/{clean}" if clean else ""

    @property
    def github_actions_url(self) -> str:
        base = self.github_repo_url
        return f"{base}/actions" if base else ""


@dataclass(frozen=True)
class UrlCheckResult:
    """Ergebnis eines HTTP- & Streamlit-Health-Checks für eine Web-App."""

    app_id: AppId | str
    target_url: str
    health_endpoint_url: str
    status: HealthStatus
    http_status_code: int
    latency_ms: int
    is_streamlit_awake: bool
    health_endpoint_ok: bool
    message: str
    checked_at: str


@dataclass(frozen=True)
class WorkflowRunSummary:
    """Zusammenfassung des jüngsten GitHub Actions Laufs eines Bots."""

    app_id: AppId
    workflow_file: str
    state: WorkflowRunState
    status_label: str
    conclusion: str
    run_url: str
    branch: str
    started_at: str
    updated_at: str
    message: str


@dataclass(frozen=True)
class DatabaseTableMetric:
    """Tabellenname und Zeilenanzahl einer einzelnen Bot-Datenbanktabelle."""

    table_name: str
    row_count: int


@dataclass(frozen=True)
class DatabaseRecentItem:
    """Kompakter Vorschaueintrag aus einer Bot-Datenbank (Artikel, Job, Regel oder Event)."""

    category: str
    title: str
    subtitle: str
    status: str
    timestamp: str
    url: str = ""


@dataclass(frozen=True)
class DatabaseHealthSummary:
    """Ergebnis der Datenbank-Inspektion (Turso Cloud oder lokales SQLite) eines Bots."""

    app_id: AppId
    backend_label: str
    status: HealthStatus
    latency_ms: int
    table_metrics: tuple[DatabaseTableMetric, ...] = ()
    recent_items: tuple[DatabaseRecentItem, ...] = ()
    active_count: int = 0
    error_count: int = 0
    last_activity_at: str = ""
    message: str = ""
    checked_at: str = ""

    @property
    def total_rows(self) -> int:
        return sum(metric.row_count for metric in self.table_metrics)


@dataclass(frozen=True)
class AppFleetSnapshot:
    """Gesamtbild einer einzelnen überwachten App (URL, Worker & Datenbank)."""

    spec: ManagedAppSpec
    overall_status: HealthStatus
    url_check: UrlCheckResult
    workflow_check: WorkflowRunSummary
    db_check: DatabaseHealthSummary
    checked_at: str


@dataclass(frozen=True)
class FleetCheckLog:
    """Persistierter Historieneintrag eines Flotten-Prüflaufs im Dashboard-Speicher."""

    id: str
    app_id: str
    overall_status: str
    url_status: str
    url_latency_ms: int
    workflow_state: str
    db_status: str
    db_total_rows: int
    summary_text: str
    created_at: str
