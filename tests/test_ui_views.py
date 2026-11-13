"""Tests für die konsolidierte Dashboard-Übersicht und Views."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.checks.coordinator import FleetCheckCoordinator
from src.domain.enums import AppId, AuthRole, HealthStatus, WorkflowRunState
from src.domain.models import (
    AppFleetSnapshot,
    DatabaseHealthSummary,
    DatabaseTableMetric,
    ManagedAppSpec,
    UrlCheckResult,
    WorkflowRunSummary,
)
from src.ui.views.overview_view import render_overview_view


def _create_mock_snapshot() -> AppFleetSnapshot:
    spec = ManagedAppSpec(
        app_id=AppId.NEWS_BOT,
        title="News-Bot",
        icon="📰",
        description="Daily News Digest Bot",
        streamlit_url="https://holli-news-bot.streamlit.app/",
        github_repo="hschenke/news-bot",
        primary_workflow="daily_digest.yaml",
        secondary_workflows=("run_extra.yaml",),
    )
    url_res = UrlCheckResult(
        app_id=AppId.NEWS_BOT,
        target_url=spec.streamlit_url,
        health_endpoint_url=f"{spec.streamlit_url}_stcore/health",
        status=HealthStatus.ONLINE,
        http_status_code=200,
        latency_ms=85,
        is_streamlit_awake=True,
        health_endpoint_ok=True,
        message="OK",
        checked_at="2026-11-12T10:00:00+00:00",
    )
    wf_res = WorkflowRunSummary(
        app_id=AppId.NEWS_BOT,
        workflow_file="daily_digest.yaml",
        state=WorkflowRunState.SUCCESS,
        status_label="Erfolgreich",
        conclusion="success",
        run_url="https://github.com/hschenke/news-bot/actions/runs/123",
        branch="main",
        started_at="2026-11-12T07:00:00Z",
        updated_at="2026-11-12T07:05:00Z",
        message="OK",
    )
    db_res = DatabaseHealthSummary(
        app_id=AppId.NEWS_BOT,
        backend_label="Turso Cloud",
        status=HealthStatus.ONLINE,
        latency_ms=40,
        table_metrics=(DatabaseTableMetric("articles", 120),),
        message="Verbunden",
        checked_at="2026-11-12T10:00:00+00:00",
    )
    return AppFleetSnapshot(
        spec=spec,
        overall_status=HealthStatus.ONLINE,
        url_check=url_res,
        workflow_check=wf_res,
        db_check=db_res,
        checked_at="2026-11-12T10:00:00+00:00",
    )


class _DummyContext(MagicMock):
    def __enter__(self) -> _DummyContext:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return False


def _fake_columns(spec: Any, **kwargs: Any) -> list[_DummyContext]:
    count = spec if isinstance(spec, int) else len(spec)
    return [_DummyContext() for _ in range(count)]


@patch("streamlit.columns", side_effect=_fake_columns)
@patch("streamlit.container", return_value=_DummyContext())
@patch("streamlit.expander", return_value=_DummyContext())
@patch("streamlit.title")
@patch("streamlit.subheader")
@patch("streamlit.caption")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.table")
@patch("streamlit.link_button")
@patch("streamlit.button", return_value=False)
def test_render_overview_view_renders_successfully(
    mock_btn: MagicMock,
    mock_link: MagicMock,
    mock_table: MagicMock,
    mock_div: MagicMock,
    mock_md: MagicMock,
    mock_cap: MagicMock,
    mock_sub: MagicMock,
    mock_title: MagicMock,
    mock_expander: MagicMock,
    mock_container: MagicMock,
    mock_cols: MagicMock,
) -> None:
    snapshot = _create_mock_snapshot()
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    repository = MagicMock()

    render_overview_view([snapshot], coordinator, repository, role=AuthRole.ADMIN)

    mock_title.assert_called_once_with("Holli Dashboard")
    mock_table.assert_called_once()
    table_rows = mock_table.call_args[0][0]
    assert table_rows[0]["App"] == "📰 [News-Bot](https://holli-news-bot.streamlit.app/)"
    assert "Streamlit URL" not in table_rows[0]
    assert "GitHub Repo" not in table_rows[0]
    assert "Details" not in table_rows[0]
    sub_titles = [call.args[0] for call in mock_sub.call_args_list]
    assert sub_titles == ["🌐 Health", "⚙️ Action"]
    assert mock_expander.call_args.kwargs["expanded"] is False
    btn_labels = [call.args[0] for call in mock_btn.call_args_list]
    assert "⚡ Alle prüfen" in btn_labels


def test_format_iso_timestamp_format() -> None:
    from src.ui.formatters import build_workflow_file_url, format_iso_timestamp

    assert format_iso_timestamp("2026-11-12T07:05:00Z") == "12.11.2026 - 07:05"
    assert format_iso_timestamp("2026-11-12T10:30:00+00:00") == "12.11.2026 - 10:30"
    assert format_iso_timestamp("") == "–"
    assert format_iso_timestamp(None) == "–"

    assert (
        build_workflow_file_url("https://github.com/hschenke/news-bot", "daily_digest.yaml", "main")
        == "https://github.com/hschenke/news-bot/blob/main/.github/workflows/daily_digest.yaml"
    )
    assert build_workflow_file_url("https://github.com/hschenke/news-bot", "../bad.yaml") == "#"
    assert build_workflow_file_url("javascript:alert(1)", "daily_digest.yaml") == "#"


@patch("streamlit.columns", side_effect=_fake_columns)
@patch("streamlit.container", return_value=_DummyContext())
@patch("streamlit.expander", return_value=_DummyContext())
@patch("streamlit.title")
@patch("streamlit.subheader")
@patch("streamlit.caption")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.table")
@patch("streamlit.link_button")
@patch("streamlit.button", return_value=False)
def test_render_overview_view_shows_wakeup_controls_for_sleeping_app(
    mock_btn: MagicMock,
    mock_link: MagicMock,
    mock_table: MagicMock,
    mock_div: MagicMock,
    mock_md: MagicMock,
    mock_cap: MagicMock,
    mock_sub: MagicMock,
    mock_title: MagicMock,
    mock_expander: MagicMock,
    mock_container: MagicMock,
    mock_cols: MagicMock,
) -> None:
    base = _create_mock_snapshot()
    sleeping_url = UrlCheckResult(
        app_id=AppId.NEWS_BOT,
        target_url=base.spec.streamlit_url,
        health_endpoint_url=f"{base.spec.streamlit_url}~/+/_stcore/health",
        status=HealthStatus.ASLEEP,
        http_status_code=200,
        latency_ms=110,
        is_streamlit_awake=False,
        health_endpoint_ok=False,
        message="Cloud-Status: Ruhezustand/Inaktiv (status=7)",
        checked_at="2026-11-13T10:00:00+00:00",
    )
    sleeping_snap = AppFleetSnapshot(
        spec=base.spec,
        overall_status=HealthStatus.ASLEEP,
        url_check=sleeping_url,
        workflow_check=base.workflow_check,
        db_check=base.db_check,
        checked_at=base.checked_at,
    )
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    repository = MagicMock()

    render_overview_view([sleeping_snap], coordinator, repository, role=AuthRole.ADMIN)

    table_rows = mock_table.call_args[0][0]
    assert table_rows[0]["Awake"] == "🌙 Sleeping"
    assert "Details" not in table_rows[0]
    btn_labels = [call.args[0] for call in mock_btn.call_args_list]
    assert "⚡ Alle prüfen" in btn_labels
    assert "☀️ Alle aufwecken" in btn_labels
    assert "☀️ App aufwecken" in btn_labels


def test_queued_worker_renders_english_badge_status_link_and_active_end() -> None:
    from src.ui.formatters import format_worker_status, format_workflow_badge

    assert format_workflow_badge(WorkflowRunState.QUEUED) == "⏳ Queued"
    assert format_worker_status(WorkflowRunState.QUEUED, "Queued") == "⏳ Queued"
    assert format_worker_status(WorkflowRunState.FAILURE, "Timed out") == "🔴 Failed (Timed out)"

    base = _create_mock_snapshot()
    queued_wf = WorkflowRunSummary(
        app_id=AppId.ALERT_BOT,
        workflow_file="check_alerts.yaml",
        state=WorkflowRunState.QUEUED,
        status_label="Queued",
        conclusion="",
        run_url="https://github.com/hschenke/alert-bot/actions/runs/999",
        branch="main",
        started_at="2026-11-13T13:00:00Z",
        updated_at="2026-11-13T13:00:00Z",
        message="check_alerts.yaml: Queued",
    )
    queued_snap = AppFleetSnapshot(
        spec=base.spec,
        overall_status=HealthStatus.ONLINE,
        url_check=base.url_check,
        workflow_check=queued_wf,
        db_check=base.db_check,
        checked_at=base.checked_at,
    )
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    repository = MagicMock()

    with (
        patch("streamlit.columns", side_effect=_fake_columns),
        patch("streamlit.expander", return_value=_DummyContext()) as mock_exp,
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.caption") as mock_cap,
        patch("streamlit.markdown") as mock_md,
        patch("streamlit.divider"),
        patch("streamlit.table"),
        patch("streamlit.link_button"),
        patch("streamlit.button", return_value=False),
    ):
        render_overview_view([queued_snap], coordinator, repository, role=AuthRole.ADMIN)

    expander_title = mock_exp.call_args.args[0]
    assert "⏳ Queued" in expander_title
    assert "Warteschlange" not in expander_title
    assert mock_exp.call_args.kwargs["expanded"] is False

    md_calls = [call.args[0] for call in mock_md.call_args_list]
    cap_calls = [call.args[0] for call in mock_cap.call_args_list]
    assert any("[⏳ Queued](https://github.com/hschenke/alert-bot/actions/runs/999)" in m for m in md_calls)
    assert any("**Finished:** ⏳ Active" in m for m in md_calls)
    assert any(
        "[`check_alerts.yaml`](https://github.com/hschenke/news-bot/blob/main/.github/workflows/check_alerts.yaml)"
        in c
        for c in cap_calls
    )


class _RerunCalled(Exception):
    pass


def test_check_all_button_runs_spinner_outside_header_columns() -> None:
    awake_snap = _create_mock_snapshot()
    session_state: dict[str, Any] = {"fleet_snapshots": [awake_snap]}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.check_all_apps.return_value = [awake_snap]
    repository = MagicMock()

    col_depth = {"current": 0, "max": 0}
    spinner_depths: list[int] = []

    class _TrackedCol:
        def __enter__(self) -> _TrackedCol:
            col_depth["current"] += 1
            col_depth["max"] = max(col_depth["max"], col_depth["current"])
            return self

        def __exit__(self, *args: object) -> bool:
            col_depth["current"] -= 1
            return False

    class _TrackedSpinner:
        def __enter__(self) -> _TrackedSpinner:
            spinner_depths.append(col_depth["current"])
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def _tracked_cols(spec: Any, **kwargs: Any) -> list[_TrackedCol]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_TrackedCol() for _ in range(count)]

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=_tracked_cols),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.title"),
        patch("streamlit.button", side_effect=lambda label, **kwargs: label == "⚡ Alle prüfen"),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_overview_view([awake_snap], coordinator, repository, role=AuthRole.ADMIN)

    coordinator.check_all_apps.assert_called_once_with(repository=repository)
    assert col_depth == {"current": 0, "max": 1}
    assert spinner_depths == [0]


def test_wake_all_action_and_single_wake_and_worker_feedback() -> None:
    from src.ui.views.overview_feedback import (
        run_wake_all_action,
        run_wake_single_action,
        run_worker_trigger_action,
        update_single_snapshot_state,
    )

    awake_news = _create_mock_snapshot()
    asleep_spec = ManagedAppSpec(
        app_id=AppId.RESEARCH_BOT,
        title="Research-Bot",
        icon="🔬",
        description="Research",
        streamlit_url="https://holli-research-bot.streamlit.app/",
        github_repo="hschenke/research-bot",
        primary_workflow="run.yaml",
    )
    asleep_url = UrlCheckResult(
        app_id=AppId.RESEARCH_BOT,
        target_url=asleep_spec.streamlit_url,
        health_endpoint_url=f"{asleep_spec.streamlit_url}~/+/_stcore/health",
        status=HealthStatus.ASLEEP,
        http_status_code=200,
        latency_ms=100,
        is_streamlit_awake=False,
        health_endpoint_ok=False,
        message="Cloud-Status: status=12",
        checked_at="2026-11-13T10:00:00+00:00",
    )
    asleep_research = AppFleetSnapshot(
        spec=asleep_spec,
        overall_status=HealthStatus.ASLEEP,
        url_check=asleep_url,
        workflow_check=awake_news.workflow_check,
        db_check=awake_news.db_check,
        checked_at=awake_news.checked_at,
    )

    session_state: dict[str, Any] = {"fleet_snapshots": [awake_news]}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.wake_and_refresh_apps.return_value = (
        [("News-Bot", True, "OK"), ("Research-Bot", True, "HTTP 204")],
        [awake_news, asleep_research],
    )
    coordinator.wake_and_refresh_single_app.return_value = (True, "HTTP 204", awake_news)
    repository = MagicMock()

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_wake_all_action([asleep_research], coordinator, repository)

    feedback = session_state["overview_feedback"]
    assert feedback["wake_titles"] == ["Research-Bot"]
    assert "Online & Awake: News-Bot" in feedback["message"]
    assert "Research-Bot startet" in feedback["message"]

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_wake_single_action(awake_news, 0, coordinator, repository)
    assert session_state["overview_feedback"]["message"] == "Online & Awake: News-Bot."

    # Worker trigger aktualisiert Status-Link & markiert Live-Update wenn Queued
    queued_wf = WorkflowRunSummary(
        app_id=AppId.NEWS_BOT,
        workflow_file="daily_digest.yaml",
        state=WorkflowRunState.QUEUED,
        status_label="Queued",
        conclusion="",
        run_url="https://github.com/hschenke/news-bot/actions/runs/555",
        branch="main",
        started_at="2026-11-13T13:05:00Z",
        updated_at="2026-11-13T13:05:00Z",
        message="daily_digest.yaml: Queued",
    )
    queued_snap = AppFleetSnapshot(
        spec=awake_news.spec,
        overall_status=HealthStatus.ONLINE,
        url_check=awake_news.url_check,
        workflow_check=queued_wf,
        db_check=awake_news.db_check,
        checked_at=awake_news.checked_at,
    )
    coordinator.trigger_and_refresh_worker.return_value = ("Workflow gestartet.", queued_snap)
    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_worker_trigger_action(awake_news, 0, coordinator, repository)

    fb = session_state["overview_feedback"]
    assert fb["worker_titles"] == ["News-Bot"]
    assert "[⏳ Queued](https://github.com/hschenke/news-bot/actions/runs/555)" in fb["message"]
    assert "Live-Update aktiv" in fb["message"]

    # Wenn neuer Run noch nicht in der GitHub-API sichtbar ist, zeigt Pending-Feedback Queued statt Alt-Failure
    from src.ui.views.overview_feedback import build_worker_feedback

    failed_prev_wf = WorkflowRunSummary(
        app_id=AppId.NEWS_BOT,
        workflow_file="daily_digest.yaml",
        state=WorkflowRunState.FAILURE,
        status_label="Failed",
        conclusion="failure",
        run_url="https://github.com/hschenke/news-bot/actions/runs/111",
        branch="main",
        started_at="2026-11-12T07:00:00Z",
        updated_at="2026-11-12T07:05:00Z",
        message="Failed",
    )
    failed_prev_snap = AppFleetSnapshot(
        spec=awake_news.spec,
        overall_status=HealthStatus.DEGRADED,
        url_check=awake_news.url_check,
        workflow_check=failed_prev_wf,
        db_check=awake_news.db_check,
        checked_at=awake_news.checked_at,
    )
    delayed_fb = build_worker_feedback(failed_prev_snap, pending=True)
    assert delayed_fb["level"] == "success"
    assert delayed_fb["worker_titles"] == ["News-Bot"]
    assert f"[⏳ Queued]({awake_news.spec.github_actions_url})" in delayed_fb["message"]

    # Offenen Wakeup-Status bei Einzelprüfung auflösen
    session_state["overview_feedback"] = {
        "level": "success",
        "message": "Wakeup gesendet",
        "wake_titles": ["News-Bot", "Alert-Bot"],
        "worker_titles": [],
    }
    with patch("streamlit.session_state", session_state):
        update_single_snapshot_state(0, awake_news)
    assert session_state["overview_feedback"]["wake_titles"] == ["Alert-Bot"]


def test_poll_live_updates_detects_queued_worker_completion_and_steady_refresh() -> None:
    from src.ui.views.live_poller import poll_live_updates

    done_snap = _create_mock_snapshot()
    queued_wf = WorkflowRunSummary(
        app_id=AppId.NEWS_BOT,
        workflow_file="daily_digest.yaml",
        state=WorkflowRunState.QUEUED,
        status_label="Queued",
        conclusion="",
        run_url="https://github.com/hschenke/news-bot/actions/runs/777",
        branch="main",
        started_at="2026-11-13T13:00:00Z",
        updated_at="2026-11-13T13:00:00Z",
        message="daily_digest.yaml: Queued",
    )
    queued_snap = AppFleetSnapshot(
        spec=done_snap.spec,
        overall_status=HealthStatus.ONLINE,
        url_check=done_snap.url_check,
        workflow_check=queued_wf,
        db_check=done_snap.db_check,
        checked_at=done_snap.checked_at,
    )

    session_state: dict[str, Any] = {
        "fleet_snapshots": [queued_snap],
        "overview_feedback": {
            "level": "success",
            "message": "News-Bot: Worker [⏳ Queued](...) · Live-Update aktiv.",
            "wake_titles": [],
            "worker_titles": ["News-Bot"],
        },
    }
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.check_single_app.return_value = queued_snap

    # Keine Änderung -> kein Rerun nötig
    with patch("streamlit.session_state", session_state):
        assert poll_live_updates(coordinator, fast_only=True) is False

    # Worker wird fertig (Queued -> Success) -> State & Banner aktualisieren sich automatisch
    coordinator.check_single_app.return_value = done_snap
    with patch("streamlit.session_state", session_state):
        assert poll_live_updates(coordinator, fast_only=True) is True

    assert session_state["fleet_snapshots"] == [done_snap]
    assert session_state["_persist_snapshot_indices"] == [0]
    assert session_state["overview_feedback"]["worker_titles"] == []
    assert "[🟢 Success](" in session_state["overview_feedback"]["message"]

    # Steady 60s-Polling ohne Änderung -> False; mit DB-Zuwachs -> True
    coordinator.refresh_quiet_all.return_value = [done_snap]
    with patch("streamlit.session_state", session_state):
        assert poll_live_updates(coordinator, fast_only=False) is False

    more_rows_db = DatabaseHealthSummary(
        app_id=AppId.NEWS_BOT,
        backend_label="Turso Cloud",
        status=HealthStatus.ONLINE,
        latency_ms=40,
        table_metrics=(DatabaseTableMetric("articles", 125),),
        message="Verbunden",
        checked_at="2026-11-12T10:05:00+00:00",
    )
    updated_db_snap = AppFleetSnapshot(
        spec=done_snap.spec,
        overall_status=HealthStatus.ONLINE,
        url_check=done_snap.url_check,
        workflow_check=done_snap.workflow_check,
        db_check=more_rows_db,
        checked_at="2026-11-12T10:05:00+00:00",
    )
    coordinator.refresh_quiet_all.return_value = [updated_db_snap]
    with patch("streamlit.session_state", session_state):
        assert poll_live_updates(coordinator, fast_only=False) is True
    assert session_state["fleet_snapshots"] == [updated_db_snap]


def test_alert_bot_icon_updated_and_no_yellow_bell_remains() -> None:
    from pathlib import Path

    from src.config.settings import load_app_settings

    settings = load_app_settings(allow_sibling_env_fallback=False)
    alert_spec = settings.get_app_spec(AppId.ALERT_BOT)
    assert alert_spec.icon == ":material/notifications_active:"

    root = Path(__file__).resolve().parent.parent
    assert "🔔" not in (root / "config" / "settings.yaml").read_text(encoding="utf-8")
    for py_file in (root / "src").rglob("*.py"):
        assert "🔔" not in py_file.read_text(encoding="utf-8"), f"Veraltetes 🔔 in {py_file}"


def test_apptest_overview_db_and_settings_views_render_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEWS_BOT_URL", "https://holli-news-bot.streamlit.app/")
    monkeypatch.setenv("RESEARCH_BOT_URL", "https://holli-research-bot.streamlit.app/")
    monkeypatch.setenv("ALERT_BOT_URL", "https://holli-alert-bot.streamlit.app/")
    monkeypatch.setenv("NEWS_BOT_REPO", "hschenke/news-bot")
    monkeypatch.setenv("RESEARCH_BOT_REPO", "hschenke/research-bot")
    monkeypatch.setenv("ALERT_BOT_REPO", "hschenke/alert-bot")

    def _app() -> None:
        from unittest.mock import MagicMock

        import streamlit as st
        from src.checks.coordinator import FleetCheckCoordinator
        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole, HealthStatus, WorkflowRunState
        from src.domain.models import (
            AppFleetSnapshot,
            DatabaseHealthSummary,
            DatabaseTableMetric,
            UrlCheckResult,
            WorkflowRunSummary,
        )
        from src.ui.views.db_view import render_db_view
        from src.ui.views.overview_view import render_overview_view
        from src.ui.views.settings_view import render_settings_view

        settings = load_app_settings(allow_sibling_env_fallback=False)
        snapshots = [
            AppFleetSnapshot(
                spec=spec,
                overall_status=HealthStatus.ONLINE,
                url_check=UrlCheckResult(
                    app_id=spec.app_id,
                    target_url=spec.streamlit_url,
                    health_endpoint_url=f"{spec.streamlit_url}_stcore/health",
                    status=HealthStatus.ONLINE,
                    http_status_code=200,
                    latency_ms=80,
                    is_streamlit_awake=True,
                    health_endpoint_ok=True,
                    message="OK",
                    checked_at="2026-11-13T12:00:00+00:00",
                ),
                workflow_check=WorkflowRunSummary(
                    app_id=spec.app_id,
                    workflow_file=spec.primary_workflow,
                    state=WorkflowRunState.SUCCESS,
                    status_label="Success",
                    conclusion="success",
                    run_url=f"https://github.com/{spec.github_repo}/actions/runs/42",
                    branch="main",
                    started_at="2026-11-13T11:00:00Z",
                    updated_at="2026-11-13T11:02:00Z",
                    message="OK",
                ),
                db_check=DatabaseHealthSummary(
                    app_id=spec.app_id,
                    backend_label="Turso Cloud",
                    status=HealthStatus.ONLINE,
                    latency_ms=30,
                    table_metrics=(DatabaseTableMetric("articles", 10),),
                    message="OK",
                    checked_at="2026-11-13T12:00:00+00:00",
                ),
                checked_at="2026-11-13T12:00:00+00:00",
            )
            for spec in settings.managed_apps
        ]
        st.session_state["fleet_snapshots"] = snapshots
        coord = MagicMock(spec=FleetCheckCoordinator)
        repo = MagicMock()
        repo.backend_name = "SQLite"
        repo.list_recent_logs.return_value = []

        render_overview_view(snapshots, coord, repo, role=AuthRole.ADMIN)
        render_db_view(snapshots)
        render_settings_view(settings, repo)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    # KPI-Reihe: DB-Einträge entfernt, nur 3 Overview-Metriken + 3x4 DB-Tab-Metriken
    metric_labels = [m.label for m in at.metric]
    assert metric_labels[:3] == ["Apps Online", "Apps Awake", "DBs Online"]
    assert "DB-Einträge" not in metric_labels

    # Abschnitts-Titel: Health & Action
    subheaders = [s.value for s in at.subheader]
    assert "🌐 Health" in subheaders
    assert "⚙️ Action" in subheaders

    # Alle 3 App-Detail-Boxen standardmäßig eingeklappt
    assert len(at.expander) == 3
    assert all(exp.proto.expanded is False for exp in at.expander)
    assert any(":material/notifications_active: Alert-Bot" in exp.label for exp in at.expander)

    # Health-Tabelle enthält alle 3 App-Links + neues Alert-Bot-Icon
    assert len(at.table) >= 3
    health_df = at.table[0].value
    app_cells = list(health_df["App"])
    assert "📰 [News-Bot](https://holli-news-bot.streamlit.app/)" in app_cells
    assert "🔬 [Research-Bot](https://holli-research-bot.streamlit.app/)" in app_cells
    assert ":material/notifications_active: [Alert-Bot](https://holli-alert-bot.streamlit.app/)" in app_cells

    # Workflow-Datei-Link & Worker-Status-Link unterscheiden sich
    captions = [c.value for c in at.caption]
    markdowns = [m.value for m in at.markdown]
    assert any(
        "https://github.com/hschenke/alert-bot/blob/main/.github/workflows/check_alerts.yaml" in c
        for c in captions
    )
    assert any("https://github.com/hschenke/alert-bot/actions/runs/42" in m for m in markdowns)

