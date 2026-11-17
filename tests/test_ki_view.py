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
    captions = [c.value for c in at.caption]
    assert not any("Research-Bot, Alert-Bot & News-Bot" in c for c in captions)

    metric_labels = [m.label for m in at.metric]
    assert metric_labels == ["KI-Modelle", "Websearch & Tools", "Tier-Profile", "Letzter Testlauf"]

    btn_labels = [b.label for b in at.button]
    assert "🔎 Discover" in btn_labels
    assert "🚀 Benchmark & Optimize" in btn_labels
    assert "🧹 Prune Failing" in btn_labels
    assert "🔓 Unblock Tools" in btn_labels
    assert "🧹 Clean Previews (0)" in btn_labels
    assert "🔒 Secret sicher speichern / komplett ersetzen" in btn_labels

    clean_prev_btn = next(b for b in at.tabs[1].button if "Clean Previews" in b.label)
    assert clean_prev_btn.disabled is True

    top_order = ["🔎 Discover", "🚀 Benchmark & Optimize", "🧹 Prune Failing", "🔓 Unblock Tools"]
    assert [btn_labels.index(lbl) for lbl in top_order] == sorted(btn_labels.index(lbl) for lbl in top_order)
    for top_lbl in top_order:
        top_btn = next(b for b in at.button if b.label == top_lbl)
        assert top_btn.proto.type == "secondary"

    # Herkunft- & Aktiv-Spalte entfernt in KI-Modelle & Websearch/Tools + Brave/Tavily/Serper Missing Keys + SearXNG sichtbar
    models_df = at.tabs[1].dataframe[0].value
    tools_df = at.tabs[2].dataframe[0].value
    assert "Herkunft" not in models_df.columns
    assert "Herkunft" not in tools_df.columns
    assert "Aktiv" not in models_df.columns
    assert "Aktiv" not in tools_df.columns
    assert "searxng_public" in list(tools_df["ID"])
    assert "🔑 Missing Key: BRAVE_SEARCH_API_KEY" in btn_labels
    assert "🔑 Missing Key: TAVILY_API_KEY" in btn_labels
    assert "🔑 Missing Key: SERPER_API_KEY" in btn_labels
    assert "🔑 Missing Key: REDDIT_CLIENT_ID" not in btn_labels

    # Tabs nutzen stabile client-seitige IDs ohne on_change='rerun' (kein Server-Reload beim Tab-Wechsel)
    ki_tc = at.get("tab_container")[0]
    assert ki_tc.proto.tab_container.id == ""
    assert "ki_view_tabs" in ki_tc.proto.id

    # Bild 2: Manuelles Modell-/Tool-Formular standardmäßig eingeklappt (expanded=False)
    model_exp = next(e for e in at.tabs[1].expander if "Neues Modell" in e.label)
    tool_exp = next(e for e in at.tabs[2].expander if "Neues Tool" in e.label)
    assert model_exp.proto.expanded is False
    assert tool_exp.proto.expanded is False
    assert any(t.label == "Probe Query Rule (optional)" for t in at.tabs[2].text_input)
    assert any(t.label == "JSON Snippet-Feld (optional)" for t in at.tabs[2].text_input)

    # Bild 3: Export, Import & Standardkatalog liegt im Live-Test-Tab (tabs[3]), nicht im Tools-Tab (tabs[2])
    assert not any("Standardkatalog" in b.label for b in at.tabs[2].button)
    reset_btn = next(b for b in at.tabs[3].button if "Standardkatalog" in b.label)
    assert "Research-Bot & Alert-Bot" in (reset_btn.help or "")
    assert any("Snapshot importieren" in b.label for b in at.tabs[3].button)

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
        patch("streamlit.button", side_effect=lambda label, **kw: "Benchmark & Optimize" in label),
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


def test_ki_view_click_missing_key_dialog_saves_and_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain import ProbeStatus
    from ki_brain.models import BrainFetchedPage
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "ui_missing_key_dlg.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.setattr(
        "ki_brain.benchmark.execute_extractor_tool",
        lambda tool, url, secrets=None: BrainFetchedPage(
            url=url, title="Python", text_content="Python Inhalt " * 12, extractor_used=tool.id, latency_ms=80
        ),
    )

    def _app() -> None:
        import os

        from src.config.settings import AppSettings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        app_settings = AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"])
        render_ki_view(app_settings, role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    missing_btn = next(b for b in at.button if b.label == "🔑 Missing Key: BROWSERBASE_API_KEY")
    at = missing_btn.click().run()
    assert not at.exception

    dlg_input = next(t for t in at.text_input if "BROWSERBASE_API_KEY" in t.label)
    dlg_input.set_value("bb_live_dialog_key_777")
    save_and_test_btn = next(b for b in at.button if b.label == "🔒 Speichern & testen")
    at = save_and_test_btn.click().run()
    assert not at.exception
    assert any("BROWSERBASE_API_KEY" in s.value and "OK" in s.value for s in at.success)

    reopened = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    assert reopened._secrets.resolve_key("BROWSERBASE_API_KEY") == "bb_live_dialog_key_777"  # noqa: SLF001
    bb_tool = reopened.get_tool("browserbase_fetch")
    assert bb_tool is not None and bb_tool.last_status == ProbeStatus.OK
    reopened.close()


def test_app_sidebar_button_navigation_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(tmp_path / "nav_brain.db"))
    monkeypatch.setenv("DASHBOARD_SQLITE_PATH", str(tmp_path / "nav_dash.db"))
    monkeypatch.delenv("DASHBOARD_APP_PASSWORD", raising=False)

    def _app() -> None:
        from unittest.mock import patch

        from src.ui.app import run_streamlit_app

        with patch("src.checks.coordinator.FleetCheckCoordinator.check_all_apps", return_value=[]):
            run_streamlit_app()

    at = AppTest.from_function(_app).run()
    assert not at.exception

    nav_buttons = [b for b in at.sidebar.button if b.label in {"Dashboard", "DB-Inspektor", "KI-View", "System & Historie"}]
    assert [b.label for b in nav_buttons] == ["Dashboard", "DB-Inspektor", "KI-View", "System & Historie"]
    assert nav_buttons[0].proto.type == "primary"
    assert [b.proto.icon for b in nav_buttons] == ["📡", "🗄️", "🤖", "⚙️"]

    ki_nav_btn = next(b for b in nav_buttons if b.label == "KI-View")
    at = ki_nav_btn.click().run()
    assert not at.exception
    assert "🧠 KI & Websearch Brain" in [h.value for h in at.header]
    active_ki_btn = next(b for b in at.sidebar.button if b.label == "KI-View")
    assert active_ki_btn.proto.type == "primary"


def test_ki_view_unblock_button_runs_spinner_outside_columns(tmp_path: Path) -> None:
    from ki_brain import UnblockSummary

    repo = create_brain_repository(sqlite_path=tmp_path / "unblock_ui.db")
    brain = KIWebsearchBrain(repo)
    fake_unblock = UnblockSummary(
        checked_blocked=["duckduckgo_instant"],
        rules_applied=["duckduckgo_instant: Probe-Query-Regel 'Python' hinterlegt"],
        unblocked_ids=["duckduckgo_instant"],
        activated_alternatives=["searxng_public (https://searx.example/search)"],
    )
    brain.resolve_blocked_tools = MagicMock(return_value=fake_unblock)  # type: ignore[method-assign]

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

    settings = AppSettings(brain_sqlite_path=str(tmp_path / "unblock_ui.db"))
    with (
        patch("src.ui.views.ki_view.create_dashboard_brain", return_value=brain),
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=lambda spec, **kw: [_TrackedCol() for _ in range(spec if isinstance(spec, int) else len(spec))]),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.button", side_effect=lambda label, **kw: "Unblock Tools" in label),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_ki_view(settings, role=AuthRole.ADMIN)

    brain.resolve_blocked_tools.assert_called_once()
    assert spinner_depths == [0]
    assert session_state["ki_brain_feedback"]["level"] == "success"
    assert "duckduckgo_instant" in session_state["ki_brain_feedback"]["message"]
    assert "searxng_public" in session_state["ki_brain_feedback"]["message"]


def test_ki_view_missing_key_preserves_tools_tab_and_renders_preview_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain import AIModelSpec, BrainTier, ModelProvider, is_preview_or_experimental_model
    from streamlit.testing.v1 import AppTest

    assert is_preview_or_experimental_model("custom-pro", "Gemini Pro Preview 03-25") is True
    assert is_preview_or_experimental_model("deep-research-pro-preview-12-2025", "Deep Research") is False

    db_file = tmp_path / "ui_tab_lock.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    brain = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    brain.add_or_update_model(
        AIModelSpec(
            id="gemini-legacy-exp",
            name="Gemini Legacy Preview",
            provider=ModelProvider.GOOGLE_GEMINI,
            model_id="gemini-legacy-preview",
            tier_hint=BrainTier.STANDARD,
        )
    )
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
    tab_id_before = at.get("tab_container")[0].proto.id

    # 1. Clean Previews (1) in Tab 1 (🧩 KI-Modelle) entfernt Preview-Modell & bleibt als Clean Previews (0) sichtbar
    purge_btn = next(b for b in at.tabs[1].button if b.label == "🧹 Clean Previews (1)")
    assert purge_btn.disabled is False
    at = purge_btn.click().run()
    assert not at.exception
    assert "gemini-legacy-exp" not in list(at.tabs[1].dataframe[0].value["ID"])
    clean_zero_btn = next(b for b in at.tabs[1].button if b.label == "🧹 Clean Previews (0)")
    assert clean_zero_btn.disabled is True

    # 2. Missing Key Button im Tools-Tab öffnet Modal sofort und lässt Tab-Block-ID stabil (Client behält Tab)
    tavily_tool_btn = next(b for b in at.tabs[2].button if b.label == "🔑 Missing Key: TAVILY_API_KEY")
    at = tavily_tool_btn.click().run()
    assert not at.exception
    tab_container_after = at.get("tab_container")[0]
    assert tab_container_after.proto.id == tab_id_before
    assert tab_container_after.proto.tab_container.id == ""
    assert at.session_state["ki_active_dialog_env"] == "TAVILY_API_KEY"
    assert any("TAVILY_API_KEY" in t.label for t in at.text_input)


def test_ki_view_ddg_202_unblock_feedback_and_prune_failing_overlay(tmp_path: Path) -> None:
    from ki_brain import ProbeStatus, UnblockSummary
    from src.ui.views.ki_sequences_tab import format_probe_badge
    from src.ui.views.ki_task_overlay import format_unblock_feedback

    assert format_probe_badge(ProbeStatus.OK, enabled=False) == "⛔ Disabled"
    assert format_probe_badge(ProbeStatus.OK, enabled=True) == "🟢 OK"

    # 1. Formatierung bei DuckDuckGo HTTP 202 bleibt vollständig (ohne Abschneiden mitten in '.com') + liefert Handlungsempfehlung
    sum_202 = UnblockSummary(
        checked_blocked=["duckduckgo_html", "duckduckgo_instant", "duckduckgo_lite"],
        rules_applied=["duckduckgo_instant: Probe-Query-Regel 'Python' hinterlegt"],
        unblocked_ids=[],
        still_blocked={
            "duckduckgo_html": "HTTP 202 (Anti-Bot-Challenge / Leerantwort auf html.duckduckgo.com)",
            "duckduckgo_instant": "HTTP 202 (Anti-Bot-Challenge / Leerantwort auf api.duckduckgo.com)",
            "duckduckgo_lite": "HTTP 202 (Anti-Bot-Challenge / Leerantwort auf lite.duckduckgo.com)",
        },
        activated_alternatives=[],
    )
    level, msg = format_unblock_feedback(sum_202)
    assert level == "warning"
    assert "html.duckduckgo.com)" in msg
    assert "TAVILY_API_KEY" in msg and "searxng_public" in msg

    # 2. Prune Failing läuft mit Foreground-Overlay außerhalb von st.columns und entfernt Error, 429 & HTTP 202
    repo = create_brain_repository(sqlite_path=tmp_path / "clean_ui.db")
    brain = KIWebsearchBrain(repo)
    gem = brain.get_model("gemini-3.6-flash")
    meteo = brain.get_tool("open_meteo")
    ddg_202 = brain.get_tool("duckduckgo_lite")
    ddg = brain.get_tool("duckduckgo_html")
    assert gem and meteo and ddg_202 and ddg
    brain.add_or_update_model(gem.model_copy(update={"last_status": ProbeStatus.ERROR}))
    brain.add_or_update_tool(meteo.model_copy(update={"last_status": ProbeStatus.ERROR}))
    brain.add_or_update_tool(
        ddg_202.model_copy(
            update={"last_status": ProbeStatus.BLOCKED, "last_error": "HTTP 202 (Anti-Bot-Challenge / Leerantwort auf lite.duckduckgo.com)"}
        )
    )
    brain.add_or_update_tool(ddg.model_copy(update={"last_status": ProbeStatus.BLOCKED, "last_error": "0 Treffer (kein 202)"}))

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

    settings = AppSettings(brain_sqlite_path=str(tmp_path / "clean_ui.db"))
    with (
        patch("src.ui.views.ki_view.create_dashboard_brain", return_value=brain),
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=lambda spec, **kw: [_TrackedCol() for _ in range(spec if isinstance(spec, int) else len(spec))]),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.button", side_effect=lambda label, **kw: "Prune Failing" in label),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_ki_view(settings, role=AuthRole.ADMIN)

    assert spinner_depths == [0]
    fb_msg = session_state["ki_brain_feedback"]["message"]
    assert "Prune Failing:" in fb_msg
    assert "gemini-3.6-flash [Error]" in fb_msg
    assert "open_meteo [Error]" in fb_msg
    assert "duckduckgo_lite [HTTP 202]" in fb_msg

    reopened = KIWebsearchBrain(create_brain_repository(sqlite_path=tmp_path / "clean_ui.db"))
    try:
        assert reopened.get_tool("duckduckgo_html") is not None
        assert reopened.get_tool("duckduckgo_lite") is None
        assert reopened.get_model("gemini-3.6-flash") is None
        assert reopened.get_tool("open_meteo") is None
    finally:
        reopened.close()


def test_ki_ui_cache_prevents_redundant_loads_and_invalidates_on_write(tmp_path: Path) -> None:
    from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache, load_cached_ui_state

    repo = create_brain_repository(sqlite_path=tmp_path / "cache_ui.db")
    brain = KIWebsearchBrain(repo)
    orig_load = brain.load_ui_state
    mock_load = MagicMock(side_effect=orig_load)
    brain.load_ui_state = mock_load  # type: ignore[method-assign]

    session_state: dict[str, Any] = {}
    with patch("streamlit.session_state", session_state):
        load_cached_ui_state(brain)
        load_cached_ui_state(brain)
        assert mock_load.call_count == 1

        invalidate_ki_ui_cache()
        load_cached_ui_state(brain)
        assert mock_load.call_count == 2
    brain.close()




