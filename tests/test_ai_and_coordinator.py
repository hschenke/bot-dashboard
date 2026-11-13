"""Tests für Flotten-Koordinator, Dashboard-Repository und KI-Diagnose-Gateway."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.ai.llm_gateway import DashboardAIGateway, build_heuristic_fleet_analysis
from src.checks.coordinator import FleetCheckCoordinator, derive_overall_health
from src.config.settings import AppSettings, load_app_settings
from src.domain.enums import AppId, HealthStatus, WorkflowRunState
from src.domain.models import (
    AppFleetSnapshot,
    DatabaseHealthSummary,
    DatabaseTableMetric,
    ManagedAppSpec,
    UrlCheckResult,
    WorkflowRunSummary,
)
from src.storage.repository import SqlDashboardRepository
from src.storage.sqlite_executor import SqliteExecutor


def _make_snapshot(
    app_id: AppId = AppId.NEWS_BOT,
    url_status: HealthStatus = HealthStatus.ONLINE,
    wf_state: WorkflowRunState = WorkflowRunState.SUCCESS,
    db_status: HealthStatus = HealthStatus.ONLINE,
) -> AppFleetSnapshot:
    spec = ManagedAppSpec(
        app_id=app_id,
        title=f"Title-{app_id.value}",
        icon="📰",
        description="Desc",
        streamlit_url="https://holli-news-bot.streamlit.app/",
        github_repo="hschenke/news-bot",
        primary_workflow="daily_digest.yaml",
    )
    url_res = UrlCheckResult(
        app_id=app_id,
        target_url=spec.streamlit_url,
        health_endpoint_url=f"{spec.streamlit_url}_stcore/health",
        status=url_status,
        http_status_code=200 if url_status == HealthStatus.ONLINE else 503,
        latency_ms=120,
        is_streamlit_awake=(url_status == HealthStatus.ONLINE),
        health_endpoint_ok=(url_status == HealthStatus.ONLINE),
        message="OK",
        checked_at="2026-11-11T18:00:00+00:00",
    )
    wf_res = WorkflowRunSummary(
        app_id=app_id,
        workflow_file="daily_digest.yaml",
        state=wf_state,
        status_label="Erfolgreich",
        conclusion="success",
        run_url="https://github.com/hschenke/news-bot/actions",
        branch="main",
        started_at="2026-11-11T07:00:00Z",
        updated_at="2026-11-11T07:02:00Z",
        message="OK",
    )
    db_res = DatabaseHealthSummary(
        app_id=app_id,
        backend_label="Turso Cloud",
        status=db_status,
        latency_ms=45,
        table_metrics=(DatabaseTableMetric("articles", 42),),
        message="Verbunden",
        checked_at="2026-11-11T18:00:00+00:00",
    )
    overall = derive_overall_health(url_res, wf_res, db_res)
    return AppFleetSnapshot(
        spec=spec,
        overall_status=overall,
        url_check=url_res,
        workflow_check=wf_res,
        db_check=db_res,
        checked_at="2026-11-11T18:00:00+00:00",
    )


def test_dashboard_repository_records_and_lists_snapshots() -> None:
    repo = SqlDashboardRepository(SqliteExecutor(":memory:"))
    snap = _make_snapshot(AppId.NEWS_BOT)
    saved = repo.record_snapshot(snap)
    assert saved.app_id == "news_bot"
    assert saved.db_total_rows == 42

    logs = repo.list_recent_logs(app_id="news_bot")
    assert len(logs) == 1
    assert logs[0].overall_status == "online"
    repo.close()


def test_heuristic_and_gemini_ai_analysis() -> None:
    snapshots = [
        _make_snapshot(AppId.NEWS_BOT, url_status=HealthStatus.ONLINE),
        _make_snapshot(AppId.RESEARCH_BOT, url_status=HealthStatus.ASLEEP),
        _make_snapshot(AppId.ALERT_BOT, wf_state=WorkflowRunState.FAILURE),
    ]
    heuristic = build_heuristic_fleet_analysis(snapshots)
    assert heuristic.fleet_health_score < 100
    assert heuristic.headline.startswith("System-Status:")
    assert "Flotten" not in heuristic.headline
    assert len(heuristic.app_insights) == 3
    assert len(heuristic.anomalies_or_warnings) >= 2

    # Mock Gemini Interactions API call
    mock_client = MagicMock()
    mock_interaction = MagicMock()
    mock_interaction.output_text = (
        '{"fleet_health_score": 92, "headline": "System stabil", '
        '"executive_summary": "Alle Systeme geprüft.", "app_insights": [], '
        '"anomalies_or_warnings": [], "next_steps": ["Weiter beobachten"]}'
    )
    mock_client.interactions.create.return_value = mock_interaction

    settings = AppSettings(gemini_api_key="AIzaSyRealKeyForTest123")
    gateway = DashboardAIGateway(settings, client=mock_client)
    ai_res = gateway.analyze_fleet(snapshots)
    assert ai_res.fleet_health_score == 92
    assert ai_res.headline == "System stabil"

    mock_interaction.output_text = "News-Bot hat 42 Artikel gespeichert."
    answer = gateway.answer_fleet_question(snapshots, "Wie viele Artikel hat der News-Bot?")
    assert "42 Artikel" in answer


def test_coordinator_checks_all_three_apps() -> None:
    settings = load_app_settings(allow_sibling_env_fallback=False)
    mock_url = MagicMock()
    mock_gh = MagicMock()
    mock_db = MagicMock()

    snap = _make_snapshot()
    mock_url.check_app_url.return_value = snap.url_check
    mock_gh.check_primary_workflow.return_value = snap.workflow_check
    mock_db.inspect_bot_database.return_value = snap.db_check

    repo = SqlDashboardRepository(SqliteExecutor(":memory:"))
    coord = FleetCheckCoordinator(
        settings,
        url_probe=mock_url,
        github_probe=mock_gh,
        db_inspector=mock_db,
    )
    results = coord.check_all_apps(repository=repo)
    assert len(results) == 3
    assert len(repo.list_recent_logs()) == 3

    sleeping_snap = _make_snapshot(AppId.ALERT_BOT, url_status=HealthStatus.ASLEEP)
    awake_snap = _make_snapshot(AppId.NEWS_BOT, url_status=HealthStatus.ONLINE)
    mock_url.wake_app.return_value = (True, "Wakeup-Signal gesendet (HTTP 204).")
    wake_results = coord.wake_sleeping_apps([awake_snap, sleeping_snap])
    assert len(wake_results) == 1
    assert wake_results[0][0] == sleeping_snap.spec.title
    assert wake_results[0][1] is True
    repo.close()


def test_cli_check_wakes_sleeping_apps_and_dashboard_url(monkeypatch: object) -> None:
    from unittest.mock import patch

    from src.worker.cli_check import run_cli_fleet_check

    sleeping = _make_snapshot(AppId.NEWS_BOT, url_status=HealthStatus.ASLEEP)
    dash_sleeping = sleeping.url_check
    mock_coord = MagicMock()
    mock_coord.check_all_apps.return_value = [sleeping]
    mock_coord.wake_sleeping_apps.return_value = [("News-Bot", True, "HTTP 204")]
    mock_coord.check_dashboard_url.return_value = dash_sleeping
    mock_coord.url_probe.wake_app.return_value = (True, "HTTP 204")

    mock_repo = MagicMock()
    with (
        patch("src.worker.cli_check.load_app_settings", return_value=AppSettings(streamlit_app_url="https://holli-bot-dashboard.streamlit.app/")),
        patch("src.worker.cli_check.create_dashboard_repository", return_value=mock_repo),
        patch("src.worker.cli_check.FleetCheckCoordinator", return_value=mock_coord),
    ):
        exit_code = run_cli_fleet_check(wake_sleeping=True)

    assert exit_code == 0
    mock_coord.wake_sleeping_apps.assert_called_once()
    mock_coord.check_dashboard_url.assert_called_once()
    mock_coord.url_probe.wake_app.assert_called_once_with(dash_sleeping.target_url)
    mock_repo.close.assert_called_once()


def test_wake_and_refresh_apps_stops_early_and_refreshes_snapshots() -> None:
    settings = load_app_settings(allow_sibling_env_fallback=False)
    mock_url = MagicMock()
    mock_gh = MagicMock()
    mock_db = MagicMock()

    online_snap = _make_snapshot(AppId.NEWS_BOT, url_status=HealthStatus.ONLINE)
    asleep_snap = _make_snapshot(AppId.ALERT_BOT, url_status=HealthStatus.ASLEEP)
    mock_gh.check_primary_workflow.return_value = online_snap.workflow_check
    mock_db.inspect_bot_database.return_value = online_snap.db_check
    mock_url.wake_app.return_value = (True, "Wakeup-Signal gesendet (HTTP 204).")

    poll_calls = 0

    def _check_app_url(spec: ManagedAppSpec) -> UrlCheckResult:
        nonlocal poll_calls
        if spec.app_id == AppId.ALERT_BOT:
            poll_calls += 1
            return asleep_snap.url_check if poll_calls == 1 else online_snap.url_check
        return online_snap.url_check

    mock_url.check_app_url.side_effect = _check_app_url
    sleep_calls: list[float] = []

    coord = FleetCheckCoordinator(
        settings,
        url_probe=mock_url,
        github_probe=mock_gh,
        db_inspector=mock_db,
    )
    wake_results, updated = coord.wake_and_refresh_apps(
        [online_snap, asleep_snap],
        wait_timeout_seconds=9.0,
        poll_interval_seconds=3.0,
        sleep_fn=sleep_calls.append,
    )

    assert len(wake_results) == 1
    assert wake_results[0][0] == asleep_snap.spec.title
    assert sleep_calls == [3.0, 3.0]
    assert len(updated) == 3
    assert all(s.url_check.is_streamlit_awake for s in updated)


def test_wake_and_refresh_apps_respects_timeout_and_single_app_wake() -> None:
    settings = load_app_settings(allow_sibling_env_fallback=False)
    mock_url = MagicMock()
    mock_gh = MagicMock()
    mock_db = MagicMock()

    asleep_snap = _make_snapshot(AppId.RESEARCH_BOT, url_status=HealthStatus.ASLEEP)
    online_snap = _make_snapshot(AppId.RESEARCH_BOT, url_status=HealthStatus.ONLINE)
    mock_gh.check_primary_workflow.return_value = asleep_snap.workflow_check
    mock_db.inspect_bot_database.return_value = asleep_snap.db_check
    mock_url.wake_app.return_value = (True, "Wakeup-Signal gesendet (HTTP 204).")
    mock_url.check_app_url.return_value = asleep_snap.url_check

    sleep_calls: list[float] = []
    coord = FleetCheckCoordinator(
        settings,
        url_probe=mock_url,
        github_probe=mock_gh,
        db_inspector=mock_db,
    )
    _, updated = coord.wake_and_refresh_apps(
        [asleep_snap],
        wait_timeout_seconds=7.0,
        poll_interval_seconds=3.0,
        sleep_fn=sleep_calls.append,
    )
    assert sleep_calls == [3.0, 3.0, 1.0]
    assert all(not s.url_check.is_streamlit_awake for s in updated)

    # Single app wake succeeds and stops immediately once awake
    single_sleep: list[float] = []
    mock_url.check_app_url.return_value = online_snap.url_check
    ok, msg, single_updated = coord.wake_and_refresh_single_app(
        online_snap.spec,
        wait_timeout_seconds=9.0,
        poll_interval_seconds=3.0,
        sleep_fn=single_sleep.append,
    )
    assert ok is True
    assert "204" in msg
    assert single_sleep == [3.0]
    assert single_updated.url_check.is_streamlit_awake is True


def test_trigger_and_refresh_worker_and_quiet_refresh() -> None:
    from src.checks.github_probe import GitHubWorkflowProbe

    settings = load_app_settings(allow_sibling_env_fallback=False)
    mock_url = MagicMock()
    mock_gh = MagicMock(spec=GitHubWorkflowProbe)
    mock_db = MagicMock()

    snap = _make_snapshot(AppId.ALERT_BOT, url_status=HealthStatus.ONLINE)
    queued_wf = WorkflowRunSummary(
        app_id=AppId.ALERT_BOT,
        workflow_file="daily_digest.yaml",
        state=WorkflowRunState.QUEUED,
        status_label="Queued",
        conclusion="",
        run_url="https://github.com/hschenke/alert-bot/actions/runs/888",
        branch="main",
        started_at="2026-11-13T13:05:00Z",
        updated_at="2026-11-13T13:05:00Z",
        message="daily_digest.yaml: Queued",
    )
    mock_url.check_app_url.return_value = snap.url_check
    mock_db.inspect_bot_database.return_value = snap.db_check
    mock_gh.trigger_workflow_dispatch.return_value = "Workflow gestartet."
    mock_gh.wait_for_triggered_run.return_value = queued_wf
    mock_gh.check_primary_workflow.return_value = queued_wf

    coord = FleetCheckCoordinator(
        settings,
        url_probe=mock_url,
        github_probe=mock_gh,
        db_inspector=mock_db,
    )
    sleep_calls: list[float] = []
    msg, refreshed = coord.trigger_and_refresh_worker(
        snap, wait_timeout_seconds=4.0, poll_interval_seconds=2.0, sleep_fn=sleep_calls.append
    )
    assert msg == "Workflow gestartet."
    assert refreshed.workflow_check.state == WorkflowRunState.QUEUED
    mock_gh.trigger_workflow_dispatch.assert_called_once_with(snap.spec)
    mock_gh.wait_for_triggered_run.assert_called_once()

    # Ohne Token überspringt refresh_quiet_all anonyme GitHub API Calls
    mock_gh.has_token = False
    mock_gh.check_primary_workflow.reset_mock()
    quiet_res = coord.refresh_quiet_all([snap])
    assert len(quiet_res) == 1
    assert quiet_res[0].workflow_check == snap.workflow_check
    mock_gh.check_primary_workflow.assert_not_called()



