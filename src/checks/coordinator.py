"""Koordinator für parallele Flotten-Prüfungen aller überwachten Streamlit-Applikationen."""

from __future__ import annotations

import time
from collections.abc import Callable
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


def _wake_targets(snapshots: list[AppFleetSnapshot]) -> list[AppFleetSnapshot]:
    return [
        s
        for s in snapshots
        if s.spec.streamlit_url.strip()
        and not s.url_check.is_streamlit_awake
        and s.url_check.status in {HealthStatus.ASLEEP, HealthStatus.DEGRADED}
    ]


def derive_overall_health(
    url_check: UrlCheckResult, workflow_check: WorkflowRunSummary, db_check: DatabaseHealthSummary
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
        return self._url_probe.check_url(raw, app_id="bot_dashboard") if raw else None

    def wake_sleeping_apps(self, snapshots: list[AppFleetSnapshot]) -> list[tuple[str, bool, str]]:
        """Sendet Wakeup-Signale parallel an alle nicht wachen überwachten Streamlit Cloud Apps."""
        targets = _wake_targets(snapshots)
        if not targets:
            return []
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            return list(pool.map(lambda s: (s.spec.title, *self._url_probe.wake_app(s.spec.streamlit_url.strip())), targets))

    def _wait_until_specs_awake(
        self,
        specs: list[ManagedAppSpec],
        wait_timeout_seconds: float,
        poll_interval_seconds: float,
        sleep_fn: Callable[[float], None],
    ) -> None:
        if not specs or wait_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            return
        pending, elapsed = list(specs), 0.0
        while pending and elapsed < wait_timeout_seconds:
            step = min(poll_interval_seconds, wait_timeout_seconds - elapsed)
            sleep_fn(step)
            elapsed += step
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                url_results = list(pool.map(self._url_probe.check_app_url, pending))
            pending = [s for s, r in zip(pending, url_results, strict=True) if not r.is_streamlit_awake]

    def wake_and_refresh_apps(
        self,
        snapshots: list[AppFleetSnapshot],
        repository: SqlDashboardRepository | None = None,
        wait_timeout_seconds: float = 9.0,
        poll_interval_seconds: float = 3.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> tuple[list[tuple[str, bool, str]], list[AppFleetSnapshot]]:
        """Weckt schlafende Apps, wartet kurz auf den Start und aktualisiert alle Snapshots."""
        targets = _wake_targets(snapshots)
        wake_results = self.wake_sleeping_apps(snapshots)
        ok_titles = {title for title, ok, _ in wake_results if ok}
        poll_specs = [snap.spec for snap in targets if snap.spec.title in ok_titles]
        self._wait_until_specs_awake(poll_specs, wait_timeout_seconds, poll_interval_seconds, sleep_fn)
        return wake_results, self.check_all_apps(repository=repository)

    def wake_and_refresh_single_app(
        self,
        spec: ManagedAppSpec,
        repository: SqlDashboardRepository | None = None,
        wait_timeout_seconds: float = 9.0,
        poll_interval_seconds: float = 3.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> tuple[bool, str, AppFleetSnapshot]:
        """Weckt eine einzelne App auf, wartet kurz auf den Start und aktualisiert ihren Snapshot."""
        ok, msg = self._url_probe.wake_app(spec.streamlit_url.strip())
        if ok:
            self._wait_until_specs_awake([spec], wait_timeout_seconds, poll_interval_seconds, sleep_fn)
        return ok, msg, self.check_single_app(spec, repository=repository)

    def trigger_and_refresh_worker(
        self,
        snapshot: AppFleetSnapshot,
        repository: SqlDashboardRepository | None = None,
        wait_timeout_seconds: float = 6.0,
        poll_interval_seconds: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> tuple[str, AppFleetSnapshot]:
        """Startet den Worker, wartet kurz auf den neuen Run und aktualisiert den Snapshot."""
        wf = snapshot.workflow_check
        msg = self._github_probe.trigger_workflow_dispatch(snapshot.spec)
        self._github_probe.wait_for_triggered_run(
            snapshot.spec, wf.run_url, wf.started_at, wait_timeout_seconds, poll_interval_seconds, sleep_fn
        )
        return msg, self.check_single_app(snapshot.spec, repository=repository)

    def check_single_app(
        self,
        spec: ManagedAppSpec,
        repository: SqlDashboardRepository | None = None,
        *,
        cached_workflow: WorkflowRunSummary | None = None,
    ) -> AppFleetSnapshot:
        """Prüft URL, GitHub Actions Worker (optional gecacht) und DB einer einzelnen App."""
        url_res = self._url_probe.check_app_url(spec)
        wf_res = cached_workflow or self._github_probe.check_primary_workflow(spec)
        db_res = self._db_inspector.inspect_bot_database(spec)
        snapshot = AppFleetSnapshot(
            spec=spec,
            overall_status=derive_overall_health(url_res, wf_res, db_res),
            url_check=url_res,
            workflow_check=wf_res,
            db_check=db_res,
            checked_at=_utc_now_iso(),
        )
        if repository is not None:
            repository.record_snapshot(snapshot)
        return snapshot

    def refresh_quiet_all(self, current: list[AppFleetSnapshot]) -> list[AppFleetSnapshot]:
        """Hintergrund-Refresh: prüft URLs/DBs (und Workflows falls GitHub-Token aktiv ist)."""
        if bool(getattr(self._github_probe, "has_token", True)) or not current:
            return self.check_all_apps(repository=None)
        with ThreadPoolExecutor(max_workers=len(current)) as pool:
            return list(pool.map(lambda s: self.check_single_app(s.spec, None, cached_workflow=s.workflow_check), current))

    def check_all_apps(self, repository: SqlDashboardRepository | None = None) -> list[AppFleetSnapshot]:
        """Prüft alle überwachten Apps parallel und speichert optional das Protokoll."""
        specs = self._settings.managed_apps
        if not specs:
            return []
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            snapshots = list(pool.map(lambda spec: self.check_single_app(spec, None), specs))
        if repository is not None:
            for snap in snapshots:
                repository.record_snapshot(snap)
        return snapshots
