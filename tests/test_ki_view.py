"""UI- & AppTest-Tests für das neue KI & Websearch Brain (`KI-View`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ki_brain import KIWebsearchBrain, create_brain_repository
from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_view import render_ki_view


def test_create_dashboard_brain_bridge_and_apptest_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "ui_brain.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    settings = AppSettings(brain_sqlite_path=str(db_file))
    brain = create_dashboard_brain(settings)
    assert len(brain.list_models()) >= 8
    assert len(brain.list_tools()) >= 15
    brain.close()

    def _app() -> None:
        import os

        from src.config.settings import AppSettings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        app_settings = AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"])
        render_ki_view(app_settings, role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    headers = [h.value for h in at.header]
    assert "🧠 KI & Websearch Brain" in headers

    metric_labels = [m.label for m in at.metric]
    assert metric_labels == ["KI-Modelle", "Websearch & Tools", "Tier-Profile", "Letzter Testlauf"]

    btn_labels = [b.label for b in at.button]
    assert "🚀 Testlauf starten & Sequenzen optimieren" in btn_labels
    assert "🔄 Auto-Sync (Modelle & Tools)" in btn_labels
    assert "🧹 Bereinigen (Error & Quota 429)" in btn_labels
    assert "🔒 Secret sicher speichern / komplett ersetzen" in btn_labels

    # Read-Only Nutzer sieht deaktivierte Buttons
    def _app_ro() -> None:
        import os

        from src.config.settings import AppSettings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        app_settings = AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"])
        render_ki_view(app_settings, role=AuthRole.READONLY)

    at_ro = AppTest.from_function(_app_ro).run()
    assert not at_ro.exception
    assert any(b.disabled for b in at_ro.button)


class _RerunCalled(Exception):
    pass


def test_ki_view_benchmark_button_runs_spinner_outside_columns(tmp_path: Path) -> None:
    repo = create_brain_repository(sqlite_path=tmp_path / "bench_ui.db")
    brain = KIWebsearchBrain(repo)
    fake_report = MagicMock()
    fake_report.run_id = "run123"
    fake_report.healthy_tools_count = 10
    fake_report.healthy_models_count = 3
    fake_report.sequences = {"fast": MagicMock(search_tool_sequence=["yahoo_search", "wikipedia_api"])}

    brain.run_benchmark = MagicMock(return_value=fake_report)  # type: ignore[method-assign]

    col_depth = {"current": 0}
    spinner_depths: list[int] = []
    session_state: dict[str, Any] = {}

    class _TrackedCol:
        def __enter__(self) -> _TrackedCol:
            col_depth["current"] += 1
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

    settings = AppSettings(brain_sqlite_path=str(tmp_path / "bench_ui.db"))
    with (
        patch("src.ui.views.ki_view.create_dashboard_brain", return_value=brain),
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=lambda spec, **kw: [_TrackedCol() for _ in range(spec if isinstance(spec, int) else len(spec))]),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.button", side_effect=lambda label, **kw: "Testlauf starten" in label),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_ki_view(settings, role=AuthRole.ADMIN)

    brain.run_benchmark.assert_called_once()
    assert spinner_depths == [0]
    assert session_state["ki_brain_feedback"]["level"] == "success"
    assert "run123" in session_state["ki_brain_feedback"]["message"]


def test_ki_view_apptest_sequence_edit_and_settings_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "ui_seq_edit.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")

    def _app() -> None:
        from unittest.mock import MagicMock

        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view
        from src.ui.views.settings_view import render_settings_view

        settings = load_app_settings(allow_sibling_env_fallback=False)
        render_ki_view(settings, role=AuthRole.ADMIN)
        repo = MagicMock()
        repo.backend_name = "SQLite"
        repo.list_recent_logs.return_value = []
        render_settings_view(settings, repo)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    # Sequenz für Fast im Editor speichern (Form-Submit 0 in ki_sequences_tab)
    fast_input = next(t for t in at.text_input if t.label == "Such-Tool-Reihenfolge (kommagetrennt)")
    fast_input.set_value("wikipedia_api, yahoo_search")
    seq_submit = next(b for b in at.button if "Sequenz speichern" in b.label)
    at = seq_submit.click().run()
    assert not at.exception

    reopened = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    assert reopened.get_sequence("fast").search_tool_sequence == ["wikipedia_api", "yahoo_search"]
    reopened.close()

    # Settings-View zeigt KI & Websearch Brain mit Groq statt gelöschter Gemini-Felder
    settings_df = at.table[0].value
    assert "KI & Websearch Brain" in list(settings_df["Komponente"])
    assert any("Groq" in str(val) for val in settings_df["Details"])

