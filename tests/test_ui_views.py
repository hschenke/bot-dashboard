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
    assert "⚡ Check All" in btn_labels
    assert "⚡ Alle prüfen" not in btn_labels
    check_all_call = next(c for c in mock_btn.call_args_list if c.args[0] == "⚡ Check All")
    assert check_all_call.kwargs.get("type", "secondary") == "secondary"


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
    assert "⚡ Check All" in btn_labels
    assert "☀️ Wake All" in btn_labels
    assert "☀️ App aufwecken" in btn_labels
    assert "⚡ Alle prüfen" not in btn_labels
    assert "☀️ Alle aufwecken" not in btn_labels


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


@pytest.mark.parametrize("trigger_label", ["⚡ Check All", "☀️ Wake All"])
def test_check_all_and_wake_all_buttons_run_task_overlay_outside_header_columns(
    trigger_label: str,
) -> None:
    from src.ui.task_overlay import INSTANT_OVERLAY_HTML

    assert "st-key-overview_top_btn_" in INSTANT_OVERLAY_HTML
    assert "st-key-ki_top_btn_" in INSTANT_OVERLAY_HTML

    awake_snap = _create_mock_snapshot()
    sleeping_url = UrlCheckResult(
        app_id=AppId.NEWS_BOT,
        target_url=awake_snap.spec.streamlit_url,
        health_endpoint_url=f"{awake_snap.spec.streamlit_url}~/+/_stcore/health",
        status=HealthStatus.ASLEEP,
        http_status_code=200,
        latency_ms=110,
        is_streamlit_awake=False,
        health_endpoint_ok=False,
        message="Sleeping",
        checked_at="2026-11-13T10:00:00+00:00",
    )
    sleeping_snap = AppFleetSnapshot(
        spec=awake_snap.spec,
        overall_status=HealthStatus.ASLEEP,
        url_check=sleeping_url,
        workflow_check=awake_snap.workflow_check,
        db_check=awake_snap.db_check,
        checked_at=awake_snap.checked_at,
    )

    session_state: dict[str, Any] = {"fleet_snapshots": [sleeping_snap]}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.check_all_apps.return_value = [awake_snap]
    coordinator.wake_and_refresh_apps.return_value = ([("News-Bot", True, "HTTP 204")], [awake_snap])
    repository = MagicMock()

    col_depth = {"current": 0, "max": 0}
    spinner_depths: list[int] = []
    overlay_keys: list[str] = []
    md_calls: list[str] = []
    empty_clears = {"count": 0}

    class _TrackedCol:
        def __enter__(self) -> _TrackedCol:
            col_depth["current"] += 1
            col_depth["max"] = max(col_depth["max"], col_depth["current"])
            return self

        def __exit__(self, *args: object) -> bool:
            col_depth["current"] -= 1
            return False

        def metric(self, *_a: object, **_kw: object) -> None:
            return None

    class _TrackedSpinner:
        def __enter__(self) -> _TrackedSpinner:
            spinner_depths.append(col_depth["current"])
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    class _TrackedEmpty:
        def markdown(self, text: str, **_kw: object) -> None:
            md_calls.append(str(text))

        def info(self, *_a: object, **_kw: object) -> None:
            return None

        def container(self, **kwargs: Any) -> _DummyContext:
            if "key" in kwargs:
                overlay_keys.append(str(kwargs["key"]))
            return _DummyContext()

        def empty(self) -> None:
            empty_clears["count"] += 1

    def _tracked_cols(spec: Any, **kwargs: Any) -> list[_TrackedCol]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_TrackedCol() for _ in range(count)]

    def _tracked_container(**kwargs: Any) -> _DummyContext:
        if "key" in kwargs:
            overlay_keys.append(str(kwargs["key"]))
        return _DummyContext()

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=_tracked_cols),
        patch("streamlit.container", side_effect=_tracked_container),
        patch("streamlit.empty", side_effect=lambda: _TrackedEmpty()),
        patch("streamlit.markdown", side_effect=lambda text, **kw: md_calls.append(str(text))),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.title"),
        patch("streamlit.button", side_effect=lambda label, **kwargs: label == trigger_label),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_overview_view([sleeping_snap], coordinator, repository, role=AuthRole.ADMIN)

    if trigger_label == "⚡ Check All":
        coordinator.check_all_apps.assert_called_once_with(repository=repository)
    else:
        coordinator.wake_and_refresh_apps.assert_called_once()
        assert coordinator.wake_and_refresh_apps.call_args.args == ([sleeping_snap],)
        assert coordinator.wake_and_refresh_apps.call_args.kwargs["repository"] is repository
        assert callable(coordinator.wake_and_refresh_apps.call_args.kwargs["on_progress"])
    assert col_depth == {"current": 0, "max": 1}
    assert spinner_depths == [0]
    assert "overview_kpi_row" in overlay_keys
    assert "overview_action_row" in overlay_keys
    assert "ki_task_overlay" in overlay_keys
    assert any("ki-task-backdrop" in m for m in md_calls)
    assert empty_clears["count"] >= 2


def test_wake_all_releases_background_overlay_on_error_never_locks_permanently() -> None:
    from src.ui.task_overlay import INSTANT_OVERLAY_HTML
    from src.ui.views.overview_actions import run_wake_all_action

    # JS-Brücke erkennt Fehler-Widget und Fail-Safe-Timeout
    assert 'data-testid="stException"' in INSTANT_OVERLAY_HTML
    assert "releaseBackdrop" in INSTANT_OVERLAY_HTML

    snap = _create_mock_snapshot()
    session_state: dict[str, Any] = {"fleet_snapshots": [snap]}
    coordinator = MagicMock(spec=FleetCheckCoordinator)
    coordinator.wake_and_refresh_apps.side_effect = RuntimeError("Streamlit Cloud timeout")
    repository = MagicMock()
    cleared_slots: list[str] = []

    class _Slot:
        def __init__(self, idx: int) -> None:
            self._idx = idx

        def markdown(self, *_a: object, **_kw: object) -> None:
            return None

        def info(self, *_a: object, **_kw: object) -> None:
            return None

        def container(self, **_kw: object) -> _DummyContext:
            return _DummyContext()

        def empty(self) -> None:
            cleared_slots.append(f"slot_{self._idx}")

    slot_counter = {"n": 0}

    def _make_slot() -> _Slot:
        slot_counter["n"] += 1
        return _Slot(slot_counter["n"])

    with (
        patch("streamlit.session_state", session_state),
        patch("streamlit.empty", side_effect=_make_slot),
        patch("streamlit.spinner", return_value=_DummyContext()),
        patch("streamlit.progress", return_value=MagicMock()),
        patch("streamlit.markdown"),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        run_wake_all_action([snap], coordinator, repository)

    # Backdrop und Overlay-Slot wurden im finally-Block garantiert geleert & Fehlerbanner gesetzt
    assert len(cleared_slots) >= 2
    assert session_state["overview_feedback"]["level"] == "error"
    assert "Streamlit Cloud timeout" in session_state["overview_feedback"]["message"]


def test_wake_all_action_and_single_wake_and_worker_feedback() -> None:
    from src.ui.views.overview_actions import (
        run_wake_all_action,
        run_wake_single_action,
        run_worker_trigger_action,
    )
    from src.ui.views.overview_feedback import update_single_snapshot_state

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


def test_apptest_overview_and_db_views_render_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui.styles import _STATIC_CSS
    from streamlit.testing.v1 import AppTest

    assert ".st-key-overview_kpi_row" in _STATIC_CSS
    assert ".st-key-overview_action_row" in _STATIC_CSS
    assert "flex-wrap: nowrap !important" in _STATIC_CSS

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

    at = AppTest.from_function(_app).run()
    assert not at.exception

    # KPI-Reihe: 3 kompakte Overview-Metriken + 3x4 DB-Tab-Metriken
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
    assert len(at.table) >= 2
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

    check_all_btn = next(b for b in at.button if b.label == "⚡ Check All")
    assert check_all_btn.proto.type == "secondary"
    assert "overview_top_btn_check_all" in check_all_btn.proto.id


def test_apptest_overview_wake_all_secondary_button_and_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEWS_BOT_URL", "https://holli-news-bot.streamlit.app/")
    monkeypatch.setenv("NEWS_BOT_REPO", "hschenke/news-bot")

    def _app() -> None:
        from unittest.mock import MagicMock

        from src.checks.coordinator import FleetCheckCoordinator
        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole, HealthStatus, WorkflowRunState
        from src.domain.models import (
            AppFleetSnapshot,
            DatabaseHealthSummary,
            UrlCheckResult,
            WorkflowRunSummary,
        )
        from src.ui.views.overview_view import render_overview_view

        spec = load_app_settings(allow_sibling_env_fallback=False).managed_apps[0]
        sleeping_snap = AppFleetSnapshot(
            spec=spec,
            overall_status=HealthStatus.ASLEEP,
            url_check=UrlCheckResult(
                app_id=spec.app_id,
                target_url=spec.streamlit_url,
                health_endpoint_url=f"{spec.streamlit_url}_stcore/health",
                status=HealthStatus.ASLEEP,
                http_status_code=200,
                latency_ms=90,
                is_streamlit_awake=False,
                health_endpoint_ok=False,
                message="Sleeping",
                checked_at="2026-11-18T12:00:00+00:00",
            ),
            workflow_check=WorkflowRunSummary(
                app_id=spec.app_id,
                workflow_file=spec.primary_workflow,
                state=WorkflowRunState.SUCCESS,
                status_label="Success",
                conclusion="success",
                run_url=f"https://github.com/{spec.github_repo}/actions/runs/42",
                branch="main",
                started_at="2026-11-18T11:00:00Z",
                updated_at="2026-11-18T11:02:00Z",
                message="OK",
            ),
            db_check=DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label="Turso Cloud",
                status=HealthStatus.ONLINE,
                latency_ms=30,
                table_metrics=(),
                message="OK",
                checked_at="2026-11-18T12:00:00+00:00",
            ),
            checked_at="2026-11-18T12:00:00+00:00",
        )
        coord = MagicMock(spec=FleetCheckCoordinator)
        repo = MagicMock()
        render_overview_view([sleeping_snap], coord, repo, role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception
    check_btn = next(b for b in at.button if b.label == "⚡ Check All")
    wake_btn = next(b for b in at.button if b.label == "☀️ Wake All")
    assert check_btn.proto.type == "secondary"
    assert wake_btn.proto.type == "secondary"
    assert wake_btn.disabled is False
    assert "overview_top_btn_check_all" in check_btn.proto.id
    assert "overview_top_btn_wake_all" in wake_btn.proto.id

    def _app_awake() -> None:
        from unittest.mock import MagicMock

        from src.checks.coordinator import FleetCheckCoordinator
        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole, HealthStatus, WorkflowRunState
        from src.domain.models import (
            AppFleetSnapshot,
            DatabaseHealthSummary,
            UrlCheckResult,
            WorkflowRunSummary,
        )
        from src.ui.views.overview_view import render_overview_view

        spec = load_app_settings(allow_sibling_env_fallback=False).managed_apps[0]
        awake_snap = AppFleetSnapshot(
            spec=spec,
            overall_status=HealthStatus.ONLINE,
            url_check=UrlCheckResult(
                app_id=spec.app_id,
                target_url=spec.streamlit_url,
                health_endpoint_url=f"{spec.streamlit_url}_stcore/health",
                status=HealthStatus.ONLINE,
                http_status_code=200,
                latency_ms=45,
                is_streamlit_awake=True,
                health_endpoint_ok=True,
                message="Online",
                checked_at="2026-11-18T12:00:00+00:00",
            ),
            workflow_check=WorkflowRunSummary(
                app_id=spec.app_id,
                workflow_file=spec.primary_workflow,
                state=WorkflowRunState.SUCCESS,
                status_label="Success",
                conclusion="success",
                run_url=f"https://github.com/{spec.github_repo}/actions/runs/42",
                branch="main",
                started_at="2026-11-18T11:00:00Z",
                updated_at="2026-11-18T11:02:00Z",
                message="OK",
            ),
            db_check=DatabaseHealthSummary(
                app_id=spec.app_id,
                backend_label="Turso Cloud",
                status=HealthStatus.ONLINE,
                latency_ms=30,
                table_metrics=(),
                message="OK",
                checked_at="2026-11-18T12:00:00+00:00",
            ),
            checked_at="2026-11-18T12:00:00+00:00",
        )
        render_overview_view([awake_snap], MagicMock(spec=FleetCheckCoordinator), MagicMock(), role=AuthRole.ADMIN)

    at_awake = AppTest.from_function(_app_awake).run()
    assert not at_awake.exception
    wake_disabled_btn = next(b for b in at_awake.button if b.label == "☀️ Wake All")
    assert wake_disabled_btn.disabled is True
