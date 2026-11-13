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
@patch("streamlit.dataframe")
@patch("streamlit.link_button")
@patch("streamlit.button", return_value=False)
def test_render_overview_view_renders_successfully(
    mock_btn: MagicMock,
    mock_link: MagicMock,
    mock_df: MagicMock,
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
    mock_df.assert_called_once()
    df_rows = mock_df.call_args[0][0]
    assert "Streamlit URL" not in df_rows[0]
    assert "GitHub Repo" not in df_rows[0]
    assert "Details" not in df_rows[0]
    assert mock_sub.call_count == 2
    btn_labels = [call.args[0] for call in mock_btn.call_args_list]
    assert "⚡ Alle prüfen" in btn_labels


def test_format_iso_timestamp_format() -> None:
    from src.ui.formatters import format_iso_timestamp

    assert format_iso_timestamp("2026-11-12T07:05:00Z") == "12.11.2026 - 07:05"
    assert format_iso_timestamp("2026-11-12T10:30:00+00:00") == "12.11.2026 - 10:30"
    assert format_iso_timestamp("") == "–"
    assert format_iso_timestamp(None) == "–"


@patch("streamlit.columns", side_effect=_fake_columns)
@patch("streamlit.container", return_value=_DummyContext())
@patch("streamlit.expander", return_value=_DummyContext())
@patch("streamlit.title")
@patch("streamlit.subheader")
@patch("streamlit.caption")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.dataframe")
@patch("streamlit.link_button")
@patch("streamlit.button", return_value=False)
def test_render_overview_view_shows_wakeup_controls_for_sleeping_app(
    mock_btn: MagicMock,
    mock_link: MagicMock,
    mock_df: MagicMock,
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

    df_rows = mock_df.call_args[0][0]
    assert df_rows[0]["Wach (Awake)"] == "🌙 Nein (Schläft)"
    assert "Details" not in df_rows[0]
    btn_labels = [call.args[0] for call in mock_btn.call_args_list]
    assert "⚡ Alle prüfen" in btn_labels
    assert "☀️ Alle aufwecken" in btn_labels
    assert "☀️ App aufwecken" in btn_labels


class _RerunCalled(Exception):
    pass


def test_compact_feedback_banner_and_refresh_button_updates_table() -> None:
    awake_snap = _create_mock_snapshot()
    session_state: dict[str, Any] = {
        "fleet_snapshots": [awake_snap],
        "overview_feedback": {
            "level": "success",
            "message": "Wakeup gesendet (HTTP 204): News-Bot fährt noch hoch.",
            "wake_titles": ["News-Bot"],
            "needs_refresh": True,
        },
    }
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.check_all_apps.return_value = [awake_snap]
    repository = MagicMock()

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=_fake_columns),
        patch("streamlit.expander", return_value=_DummyContext()),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.title"),
        patch("streamlit.subheader"),
        patch("streamlit.caption"),
        patch("streamlit.markdown"),
        patch("streamlit.divider"),
        patch("streamlit.dataframe"),
        patch("streamlit.link_button"),
        patch("streamlit.success") as mock_success,
        patch("streamlit.button", return_value=False) as mock_btn,
    ):
        render_overview_view([awake_snap], coordinator, repository, role=AuthRole.ADMIN)
        mock_success.assert_called_once_with(
            "Wakeup gesendet (HTTP 204): News-Bot fährt noch hoch."
        )
        assert "🔄 Aktualisieren" in [call.args[0] for call in mock_btn.call_args_list]

    # Klick auf "🔄 Aktualisieren" prüft alle Apps und aktualisiert Banner + Snapshots
    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=_fake_columns),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.title"),
        patch("streamlit.success"),
        patch(
            "streamlit.button",
            side_effect=lambda *args, **kwargs: kwargs.get("key") == "refresh_after_wake",
        ),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_overview_view([awake_snap], coordinator, repository, role=AuthRole.ADMIN)

    coordinator.check_all_apps.assert_called_once_with(repository=repository)
    assert session_state["fleet_snapshots"] == [awake_snap]
    assert session_state["overview_feedback"]["needs_refresh"] is False
    assert session_state["overview_feedback"]["message"] == "Aktiv & wach: News-Bot."


def test_wake_all_action_builds_single_compact_feedback() -> None:
    from src.ui.views.overview_feedback import run_wake_all_action

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
        message="Cloud-Status: Ruhezustand/Inaktiv (status=12)",
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

    session_state: dict[str, Any] = {}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.wake_and_refresh_apps.return_value = (
        [("News-Bot", True, "OK"), ("Research-Bot", True, "HTTP 204")],
        [awake_news, asleep_research],
    )
    repository = MagicMock()

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_wake_all_action([asleep_research], coordinator, repository)

    feedback = session_state["overview_feedback"]
    assert feedback["needs_refresh"] is True
    assert feedback["wake_titles"] == ["Research-Bot"]
    assert "Aktiv & wach: News-Bot" in feedback["message"]
    assert "Research-Bot fährt noch hoch" in feedback["message"]


def test_single_wake_and_worker_actions_set_compact_feedback() -> None:
    from src.ui.views.overview_feedback import (
        run_wake_single_action,
        run_worker_trigger_action,
        update_single_snapshot_state,
    )

    awake_news = _create_mock_snapshot()
    session_state: dict[str, Any] = {"fleet_snapshots": [awake_news]}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.wake_and_refresh_single_app.return_value = (True, "HTTP 204", awake_news)
    coordinator.github_probe.trigger_workflow_dispatch.return_value = "Workflow gestartet."
    repository = MagicMock()

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_wake_single_action(awake_news, 0, coordinator, repository)

    assert session_state["overview_feedback"]["needs_refresh"] is False
    assert session_state["overview_feedback"]["message"] == "Aktiv & wach: News-Bot."

    # Offenen Wakeup-Status bei Einzelprüfung auflösen
    session_state["overview_feedback"] = {
        "level": "success",
        "message": "Wakeup gesendet",
        "wake_titles": ["News-Bot", "Alert-Bot"],
        "needs_refresh": True,
    }
    with patch("streamlit.session_state", session_state):
        update_single_snapshot_state(0, awake_news)
    assert session_state["overview_feedback"]["wake_titles"] == ["Alert-Bot"]
    assert "Alert-Bot fährt noch hoch" in session_state["overview_feedback"]["message"]

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_worker_trigger_action(awake_news, coordinator)
    assert session_state["overview_feedback"]["message"] == "News-Bot: Workflow gestartet."



