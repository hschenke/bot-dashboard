"""Koordinator für parallele Flotten-Prüfungen aller überwachten Streamlit-Applikationen."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from src.checks.db_probe import BotDatabaseInspector
from src.checks.github_probe import GitHubWorkflowProbe
from src.checks.url_probe import SafeUrlProbe
from src.config.settings import AppSettings
from src.domain.enums import HealthStatus, WorkflowRunState
from src.domain.models import (
    AppFleetSnapshot,
    DatabaseHealthSummary,
    ManagedAppSpec,
    UrlCheckResult,
    WorkflowRunSummary,
)
from src.storage.repository import SqlDashboardRepository


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def derive_overall_health(
    url_check: UrlCheckResult,
    workflow_check: WorkflowRunSummary,
    db_check: DatabaseHealthSummary,
) -> HealthStatus:
    """Aggregiert URL-, Worker- und DB-Status zu einem Gesamtzustand der App."""
    if url_check.status == HealthStatus.OFFLINE and db_check.status == HealthStatus.OFFLINE:
        return HealthStatus.OFFLINE
    if url_check.status == HealthStatus.ASLEEP:
        return HealthStatus.ASLEEP
    if (
        url_check.status in {HealthStatus.OFFLINE, HealthStatus.DEGRADED}
        or db_check.status in {HealthStatus.OFFLINE, HealthStatus.DEGRADED}
        or workflow_check.state == WorkflowRunState.FAILURE
    ):
        return HealthStatus.DEGRADED
    if url_check.status == HealthStatus.ONLINE or db_check.status == HealthStatus.ONLINE:
        return HealthStatus.ONLINE
    return HealthStatus.UNKNOWN


class FleetCheckCoordinator:
    """Führt URL-, GitHub-Worker- und Datenbank-Prüfungen für alle konfigurierten Bots durch."""

    def __init__(
        self,
        settings: AppSettings,
        url_probe: SafeUrlProbe | None = None,
        github_probe: GitHubWorkflowProbe | None = None,
        db_inspector: BotDatabaseInspector | None = None,
    ) -> None:
        self._settings = settings
        self._url_probe = url_probe or SafeUrlProbe()
        self._github_probe = github_probe or GitHubWorkflowProbe(settings.github_token)
        self._db_inspector = db_inspector or BotDatabaseInspector()

    @property
    def github_probe(self) -> GitHubWorkflowProbe:
        return self._github_probe

    @property
    def url_probe(self) -> SafeUrlProbe:
        return self._url_probe

    def check_dashboard_url(self) -> UrlCheckResult | None:
        """Prüft die eigene Streamlit-URL des Dashboards (`STREAMLIT_APP_URL`)."""
        raw = self._settings.streamlit_app_url.strip()
        if not raw:
            return None
        return self._url_probe.check_url(raw, app_id="bot_dashboard")

    def wake_sleeping_apps(self, snapshots: list[AppFleetSnapshot]) -> list[tuple[str, bool, str]]:
        """Sendet Wakeup-Signale an alle aktuell nicht wachen überwachten Streamlit Cloud Apps."""
        results: list[tuple[str, bool, str]] = []
        for snap in snapshots:
            raw_url = snap.spec.streamlit_url.strip()
            if not raw_url or snap.url_check.is_streamlit_awake:
                continue
            if snap.url_check.status in {HealthStatus.ASLEEP, HealthStatus.DEGRADED}:
                ok, msg = self._url_probe.wake_app(raw_url)
                results.append((snap.spec.title, ok, msg))
        return results

    def check_single_app(
        self,
        spec: ManagedAppSpec,
        repository: SqlDashboardRepository | None = None,
    ) -> AppFleetSnapshot:
        """Prüft URL, GitHub Actions Worker und Datenbank einer einzelnen App."""
        checked_at = _utc_now_iso()
        url_res = self._url_probe.check_app_url(spec)
        wf_res = self._github_probe.check_primary_workflow(spec)
        db_res = self._db_inspector.inspect_bot_database(spec)
        overall = derive_overall_health(url_res, wf_res, db_res)

        snapshot = AppFleetSnapshot(
            spec=spec,
            overall_status=overall,
            url_check=url_res,
            workflow_check=wf_res,
            db_check=db_res,
            checked_at=checked_at,
        )
        if repository is not None:
            repository.record_snapshot(snapshot)
        return snapshot

    def check_all_apps(
        self,
        repository: SqlDashboardRepository | None = None,
    ) -> list[AppFleetSnapshot]:
        """Prüft alle 3 überwachten Apps parallel und speichert optional das Protokoll."""
        specs = self._settings.managed_apps
        if not specs:
            return []

        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = [pool.submit(self.check_single_app, spec, None) for spec in specs]
            snapshots = [fut.result() for fut in futures]

        if repository is not None:
            for snap in snapshots:
                repository.record_snapshot(snap)
        return snapshots
