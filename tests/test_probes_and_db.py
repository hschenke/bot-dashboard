"""Tests für URL-Probe, GitHub-Probe, DB-Inspektor und Dashboard-Repository."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.checks.db_probe import BotDatabaseInspector
from src.checks.github_probe import GitHubWorkflowProbe
from src.checks.url_probe import SafeUrlProbe, build_streamlit_health_url
from src.domain.enums import AppId, HealthStatus, WorkflowRunState
from src.domain.models import ManagedAppSpec
from src.storage.sqlite_executor import SqliteExecutor


def _sample_spec(app_id: AppId = AppId.ALERT_BOT) -> ManagedAppSpec:
    tables_map = {
        AppId.NEWS_BOT: ("articles", "briefings"),
        AppId.RESEARCH_BOT: ("research_jobs", "research_reports"),
        AppId.ALERT_BOT: ("alert_rules", "alert_events", "alert_check_logs"),
    }
    return ManagedAppSpec(
        app_id=app_id,
        title="Test-Bot",
        icon="🔔",
        description="Test",
        streamlit_url="https://holli-alert-bot.streamlit.app/",
        github_repo="hschenke/alert-bot",
        primary_workflow="check_alerts.yaml",
        expected_tables=tables_map[app_id],
    )


def test_url_probe_detects_online_and_sleeping_apps() -> None:
    assert (
        build_streamlit_health_url("https://holli-alert-bot.streamlit.app")
        == "https://holli-alert-bot.streamlit.app/_stcore/health"
    )

    mock_session = MagicMock()

    def _fake_get(url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.is_redirect = False
        resp.status_code = 200
        resp.encoding = "utf-8"
        body = b"ok" if "_stcore/health" in url else b"<html>Streamlit App Running</html>"
        resp.iter_content.return_value = [body]
        return resp

    mock_session.get.side_effect = _fake_get
    probe = SafeUrlProbe(
        session=mock_session,
        dns_resolver=lambda host, port: ["35.190.10.10"],
    )
    res = probe.check_app_url(_sample_spec())
    assert res.status == HealthStatus.ONLINE
    assert res.is_streamlit_awake is True
    assert res.health_endpoint_ok is True

    # Sleeping Streamlit app simulation
    def _sleeping_get(url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.is_redirect = False
        resp.status_code = 200
        resp.encoding = "utf-8"
        resp.iter_content.return_value = [b"This app has gone to sleep. Yes, get this app back up!"]
        return resp

    mock_session.get.side_effect = _sleeping_get
    res_sleep = probe.check_app_url(_sample_spec())
    assert res_sleep.status == HealthStatus.ASLEEP
    assert res_sleep.is_streamlit_awake is False


def test_github_workflow_probe_parses_runs_and_triggers_dispatch() -> None:
    mock_session = MagicMock()
    run_resp = MagicMock()
    run_resp.status_code = 200
    run_resp.json.return_value = {
        "workflow_runs": [
            {
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/hschenke/alert-bot/actions/runs/123",
                "head_branch": "main",
                "run_started_at": "2026-11-11T17:00:00Z",
                "updated_at": "2026-11-11T17:01:00Z",
            }
        ]
    }
    mock_session.get.return_value = run_resp

    dispatch_resp = MagicMock()
    dispatch_resp.status_code = 204
    mock_session.post.return_value = dispatch_resp

    probe = GitHubWorkflowProbe(github_token="ghp_test_token", session=mock_session)
    spec = _sample_spec()
    summary = probe.check_primary_workflow(spec)
    assert summary.state == WorkflowRunState.SUCCESS
    assert summary.conclusion == "success"

    msg = probe.trigger_workflow_dispatch(spec)
    assert "check_alerts.yaml" in msg


def test_bot_database_inspector_reads_alert_research_and_news_schemas() -> None:
    inspector = BotDatabaseInspector()

    # 1. Alert-Bot DB
    alert_exec = SqliteExecutor(":memory:")
    alert_exec.execute(
        """
        CREATE TABLE alert_rules (
            id TEXT PRIMARY KEY, title TEXT, detector_type TEXT, status TEXT,
            last_value_text TEXT, last_checked_at TEXT, target_url TEXT,
            consecutive_failures INTEGER, updated_at TEXT
        )
        """
    )
    alert_exec.execute(
        """
        CREATE TABLE alert_events (
            id TEXT PRIMARY KEY, rule_title TEXT, headline TEXT,
            measured_display TEXT, created_at TEXT, source_url TEXT
        )
        """
    )
    alert_exec.execute("CREATE TABLE alert_check_logs (id TEXT PRIMARY KEY)")
    alert_exec.execute(
        """
        INSERT INTO alert_rules VALUES (
            'r1', 'Freiburg Temperatur', 'weather_threshold', 'active',
            '14.2 °C', '2026-11-11T17:00:00Z', 'https://open-meteo.com', 0, '2026-11-11T17:00:00Z'
        )
        """
    )
    summary_alert = inspector.inspect_bot_database(
        _sample_spec(AppId.ALERT_BOT),
        executor_override=alert_exec,
    )
    assert summary_alert.status == HealthStatus.ONLINE
    assert summary_alert.total_rows == 1
    assert summary_alert.active_count == 1
    assert len(summary_alert.recent_items) == 1
    alert_exec.close()

    # 2. Research-Bot DB
    res_exec = SqliteExecutor(":memory:")
    res_exec.execute(
        """
        CREATE TABLE research_jobs (
            id TEXT PRIMARY KEY, topic TEXT, status TEXT, phase TEXT,
            progress_percent INTEGER, created_at TEXT, completed_at TEXT
        )
        """
    )
    res_exec.execute("CREATE TABLE research_reports (job_id TEXT PRIMARY KEY)")
    res_exec.execute(
        "INSERT INTO research_jobs VALUES ('j1', 'Quantencomputing', 'running', 'search', 60, '2026-11-11T16:00:00Z', '')"
    )
    summary_res = inspector.inspect_bot_database(
        _sample_spec(AppId.RESEARCH_BOT),
        executor_override=res_exec,
    )
    assert summary_res.status == HealthStatus.ONLINE
    assert summary_res.active_count == 1
    res_exec.close()

    # 3. News-Bot DB
    news_exec = SqliteExecutor(":memory:")
    news_exec.execute(
        """
        CREATE TABLE articles (
            url TEXT PRIMARY KEY, title TEXT, source TEXT,
            category TEXT, timestamp REAL, created_at TEXT
        )
        """
    )
    news_exec.execute(
        "CREATE TABLE briefings (id INTEGER PRIMARY KEY, briefing_date TEXT, model_used TEXT, created_at TEXT)"
    )
    news_exec.execute(
        "INSERT INTO articles VALUES ('https://heise.de/1', 'KI-News', 'Heise', 'Tech', 100.0, '2026-11-11')"
    )
    summary_news = inspector.inspect_bot_database(
        _sample_spec(AppId.NEWS_BOT),
        executor_override=news_exec,
    )
    assert summary_news.status == HealthStatus.ONLINE
    assert summary_news.total_rows == 1
    news_exec.close()
