"""Tests für die konsolidierte Dashboard-Übersicht und Views."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

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
from src.ui.views.urls_running_view import render_urls_running_view


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
    mock_container: MagicMock,
    mock_cols: MagicMock,
) -> None:
    snapshot = _create_mock_snapshot()
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    repository = MagicMock()

    render_overview_view([snapshot], coordinator, repository, role=AuthRole.ADMIN)

    mock_title.assert_called_once_with("Holli Dashboard")
    mock_df.assert_called_once()
    assert mock_sub.call_count == 2


@patch("streamlit.columns", side_effect=_fake_columns)
@patch("streamlit.container", return_value=_DummyContext())
@patch("streamlit.title")
@patch("streamlit.subheader")
@patch("streamlit.caption")
@patch("streamlit.markdown")
@patch("streamlit.divider")
@patch("streamlit.dataframe")
@patch("streamlit.link_button")
@patch("streamlit.button", return_value=False)
def test_render_urls_running_view_compatibility_shim(
    mock_btn: MagicMock,
    mock_link: MagicMock,
    mock_df: MagicMock,
    mock_div: MagicMock,
    mock_md: MagicMock,
    mock_cap: MagicMock,
    mock_sub: MagicMock,
    mock_title: MagicMock,
    mock_container: MagicMock,
    mock_cols: MagicMock,
) -> None:
    snapshot = _create_mock_snapshot()
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    repository = MagicMock()

    render_urls_running_view([snapshot], coordinator, role=AuthRole.READONLY, repository=repository)
    mock_title.assert_called_once_with("Holli Dashboard")
