"""UI- & AppTest-Tests für das KI-Brain (Sequencing, KI, Tools, Vault, Check & Navigation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ki_brain import KIWebsearchBrain, PipelineSummary, create_brain_repository
from src.ai.brain_bridge import create_dashboard_brain
from src.config.settings import AppSettings
from src.domain.enums import AuthRole
from src.ui.views.ki_view import KI_TAB_LABELS, render_ki_view


class _RerunCalled(Exception):
    pass


def test_create_dashboard_brain_bridge_and_apptest_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    assert KI_TAB_LABELS == ("🎯 Sequencing", "🧩 KI", "🔧 Tools", "🔐 Vault", "🧪 Check")

    db_file = tmp_path / "ui_brain.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    settings = AppSettings(brain_sqlite_path=str(db_file))
    brain = create_dashboard_brain(settings)
    assert len(brain.list_models()) >= 7
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
    assert "🧠 KI-Brain" in headers

    metric_labels = [m.label for m in at.metric]
    assert metric_labels == ["KI-Modelle", "Tools", "Letzter Testlauf"]
    assert not any("Speicher-Backend" in c.value for c in at.caption)

    # 1 vereinter Ablauf-Button statt 4 Einzelbuttons; Clean Previews & Retry Excluded entfernt
    btn_labels = [b.label for b in at.button]
    assert "🚀 Sync & Optimize" in btn_labels
    assert "🔎 Discover" not in btn_labels
    assert "🚀 Benchmark & Optimize" not in btn_labels
    assert "🧹 Prune Failing" not in btn_labels
    assert "🔓 Unblock Tools" not in btn_labels
    assert not any("Clean Previews" in lbl for lbl in btn_labels)
    assert not any("Retry Excluded" in lbl for lbl in btn_labels)

    pipeline_btn = next(b for b in at.button if b.label == "🚀 Sync & Optimize")
    assert "ki_top_btn_pipeline" in pipeline_btn.proto.id
    assert pipeline_btn.proto.type == "secondary"

    # Ablauf-Optionen im eingeklappten Expander: Preview-Checkbox entfernt, Kontingent- & Paid-Option standardmäßig AUS
    options_exp = next(e for e in at.expander if "Ablauf-Optionen" in e.label)
    assert options_exp.proto.expanded is False
    exp_cb_labels = [c.label for c in options_exp.checkbox]
    assert not any("Preview" in lbl for lbl in exp_cb_labels)
    assert "🔎 Discover" in exp_cb_labels
    assert "🚀 Benchmark" in exp_cb_labels
    assert "🔓 Unblock" in exp_cb_labels
    assert "🧹 Prune Failing" in exp_cb_labels
    quota_cb = next(c for c in options_exp.checkbox if c.key == "ki_opt_include_quota_tools")
    assert quota_cb.value is False
    paid_cb = next(c for c in options_exp.checkbox if c.key == "ki_opt_include_paid")
    assert paid_cb.value is False
    assert not any("Ausgeschlossene Modelle erneut prüfen" in c.label for c in at.checkbox)

    # 5 Tabs: Sequencing (0), KI (1), Tools (2), Vault (3), Check (4)
    assert len(at.tabs) == 5
    ki_tc = at.get("tab_container")[0]
    assert ki_tc.proto.tab_container.id == ""
    assert "ki_brain_tabs" in ki_tc.proto.id

    # Tab 0 (Sequencing): 3 Sequence-Boxen untereinander, standardmäßig zugeklappt & ohne '0% Kosten' im Titel
    seq_exps = [e for e in at.tabs[0].expander if any(t in e.label for t in ("⚡ Fast", "⚖️ Standard", "🔬 Deep"))]
    assert len(seq_exps) == 3
    for exp in seq_exps:
        assert exp.proto.expanded is False
        assert "0% Kosten" not in exp.label

    # Tab 1 (KI) & Tab 2 (Tools): Tabellen intakt, Einzel-Test-Button vorhanden & KEINE API-Key-Buttons außerhalb von Vault
    assert not any("0%-Standard" in c.value for c in at.tabs[1].caption)
    models_df = at.tabs[1].dataframe[0].value
    tools_df = at.tabs[2].dataframe[0].value
    assert "Latenz (ms)" in models_df.columns
    assert "Kosten (~$/1M)" in models_df.columns
    assert "Budget" in models_df.columns
    assert any("Free Tier" in str(val) for val in models_df["Budget"])
    assert "Kontingent / Kosten" in tools_df.columns
    assert "searxng_public" in list(tools_df["ID"])
    assert "gemini_google_maps" in list(tools_df["ID"])
    assert "brave_search" not in list(tools_df["ID"])
    assert any("5000/5000/Mo" in str(val) for val in tools_df["Kontingent / Kosten"])
    assert any(b.key == "ki_tool_probe_btn" and b.label == "🧪 Testen" for b in at.tabs[2].button)

    for non_vault_idx in (0, 1, 2, 4):
        assert not any("Missing Key" in b.label for b in at.tabs[non_vault_idx].button)

    # Free- & Paid-Tier API-Key-Felder im manuellen Modell-Formular (Tab 1) vorbereitet
    model_inputs = [t.label for t in at.tabs[1].text_input]
    assert "Free-Tier API-Key Env" in model_inputs
    assert "Paid-Tier API-Key Env (optional)" in model_inputs

    # Tab 3 (Vault): enthält Free-Tier-, Paid-Tier-, Tool-Keys (ohne Brave!) & 1 gemeinsamen Flotten-Key
    vault_btn_keys = [b.key for b in at.tabs[3].button]
    assert "ki_vault_row_btn_GEMINI_FREE_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_GEMINI_PAID_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_GEMINI_API_KEY" not in vault_btn_keys
    assert "ki_vault_row_btn_GROQ_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_GROQ_PAID_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_TAVILY_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_BRAVE_SEARCH_API_KEY" not in vault_btn_keys
    assert "ki_vault_row_btn_SERPER_API_KEY" in vault_btn_keys
    assert "ki_vault_row_btn_BROWSERBASE_API_KEY" in vault_btn_keys
    assert "ki_fleet_issue_btn" in vault_btn_keys
    assert "ki_app_issue_btn_news_bot" not in vault_btn_keys
    assert any(b.label == "🔒 Secret sicher speichern / ersetzen" for b in at.tabs[3].button)

    # Tab 4 (Check): Live-Test, Benchmark-Report & Snapshot (ohne Bot-API-Einbindungsblock)
    assert any("Letzter Benchmark-Testlauf" in s.value for s in at.tabs[4].subheader)
    assert not any("News-Bot, Research-Bot & Alert-Bot" in s.value for s in at.tabs[4].subheader)
    assert len(at.tabs[4].code) == 0
    assert any(c.label.startswith("💳 Kostenpflichtige Modelle/Tools") and c.value is False for c in at.tabs[4].checkbox)
    maps_sel = next(s for s in at.tabs[4].selectbox if s.label == "🗺️ Google Maps Grounding")
    assert maps_sel.value == "Auto (Geo-Smart)"
    assert any("Standardkatalog" in b.label for b in at.tabs[4].button)
    assert any("Snapshot importieren" in b.label for b in at.tabs[4].button)

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


def test_ki_brain_unified_pipeline_button_runs_overlay_outside_columns(tmp_path: Path) -> None:
    from ki_brain import BenchmarkReport, DiscoverySummary, UnblockSummary

    repo = create_brain_repository(sqlite_path=tmp_path / "pipe_ui.db")
    brain = KIWebsearchBrain.for_dashboard(repo)
    fake_bench = BenchmarkReport(
        run_id="run999",
        started_at="2026-11-19T10:00:00+00:00",
        finished_at="2026-11-19T10:00:03+00:00",
        test_query="Python",
        test_url="https://example.org",
        healthy_tools_count=10,
        healthy_models_count=3,
    )
    brain.run_full_pipeline = MagicMock(  # type: ignore[method-assign]
        return_value=PipelineSummary(
            executed_steps=("discover", "benchmark", "unblock", "prune"),
            discovery=DiscoverySummary(added_models=["m1"], checked_sources=["builtins"]),
            benchmark=fake_bench,
            unblock=UnblockSummary(checked_blocked=["ddg"], unblocked_ids=["ddg"]),
        )
    )

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

    settings = AppSettings(brain_sqlite_path=str(tmp_path / "pipe_ui.db"))
    with (
        patch("src.ui.views.ki_view.create_dashboard_brain", return_value=brain),
        patch("streamlit.session_state", session_state),
        patch("streamlit.columns", side_effect=lambda spec, **kw: [_TrackedCol() for _ in range(spec if isinstance(spec, int) else len(spec))]),
        patch("streamlit.spinner", return_value=_TrackedSpinner()),
        patch("streamlit.button", side_effect=lambda label, **kw: label == "🚀 Sync & Optimize"),
        patch("streamlit.rerun", side_effect=_RerunCalled),
        pytest.raises(_RerunCalled),
    ):
        render_ki_view(settings, role=AuthRole.ADMIN)

    brain.run_full_pipeline.assert_called_once()
    assert spinner_depths == [0]
    assert session_state["ki_brain_feedback"]["level"] == "success"
    assert "run999" in session_state["ki_brain_feedback"]["message"]


def test_ki_brain_sequence_editor_saves_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "ui_seq_edit.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))

    def _app() -> None:
        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        render_ki_view(load_app_settings(allow_sibling_env_fallback=False), role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    fast_input = next(t for t in at.tabs[0].text_input if t.label == "Such-Tool-Reihenfolge (kommagetrennt)")
    fast_input.set_value("wikipedia_api, yahoo_search")
    seq_submit = next(b for b in at.tabs[0].button if "Sequenz speichern" in b.label)
    at = seq_submit.click().run()
    assert not at.exception

    reopened = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    assert reopened.get_sequence("fast").search_tool_sequence == ["wikipedia_api", "yahoo_search"]
    reopened.close()


def test_ki_vault_tab_secret_dialog_and_three_app_token_management(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain import BrainAuthError, ProbeStatus
    from ki_brain.models import BrainFetchedPage
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "ui_vault_flow.db"
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

        render_ki_view(AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"]), role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception
    tab_id_before = at.get("tab_container")[0].proto.id

    # 1. Im Vault-Tab (tabs[3]) Missing-Key anklicken -> Dialog öffnet sich, Tab-ID bleibt stabil
    bb_btn = next(b for b in at.tabs[3].button if b.key == "ki_vault_row_btn_BROWSERBASE_API_KEY")
    at = bb_btn.click().run()
    assert not at.exception
    assert at.get("tab_container")[0].proto.id == tab_id_before
    assert at.session_state["ki_active_dialog_env"] == "BROWSERBASE_API_KEY"

    dlg_input = next(t for t in at.text_input if "BROWSERBASE_API_KEY" in t.label)
    dlg_input.set_value("bb_live_dialog_key_777")
    save_and_test_btn = next(b for b in at.button if b.label == "🔒 Speichern & testen")
    at = save_and_test_btn.click().run()
    assert not at.exception
    assert "ki_active_dialog_env" not in at.session_state
    assert any("BROWSERBASE_API_KEY" in s.value and "OK" in s.value for s in at.success)

    reopened = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    assert reopened._secrets.resolve_key("BROWSERBASE_API_KEY") == "bb_live_dialog_key_777"  # noqa: SLF001
    bb_tool = reopened.get_tool("browserbase_fetch")
    assert bb_tool is not None and bb_tool.last_status == ProbeStatus.OK
    reopened.close()

    # 2. Im Vault-Tab 1 gemeinsamen kompakten KI_BRAIN_KEY erzeugen -> einmalig anzeigen & funktionsfähig
    import ki_brain
    from ki_brain import FLEET_CLIENT_ID

    issue_btn = next(b for b in at.tabs[3].button if b.key == "ki_fleet_issue_btn")
    at = issue_btn.click().run()
    assert not at.exception
    issued_info = at.session_state["ki_vault_issued_fleet_key"]
    issued_conn_key = str(issued_info["connection_key"])
    assert issued_conn_key.startswith("kib_")
    assert len(issued_conn_key) <= 32
    assert any(f'KI_BRAIN_KEY = "{issued_conn_key}"' in c.value for c in at.tabs[3].code)

    with ki_brain.connect(issued_conn_key, sqlite_path=db_file) as bot:
        assert bot.client_app_id == FLEET_CLIENT_ID

    # 3. Sperren über Vault-Button entzieht sofort den Zugriff für alle 3 Apps
    revoke_btn = next(b for b in at.tabs[3].button if b.key == "ki_fleet_revoke_btn")
    at = revoke_btn.click().run()
    assert not at.exception
    with pytest.raises(BrainAuthError):
        ki_brain.connect(issued_conn_key, sqlite_path=db_file)


def test_app_sidebar_has_three_pages_ki_brain_and_divider_before_logout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(tmp_path / "nav_brain.db"))
    monkeypatch.setenv("SQLITE_PATH", ":memory:")
    monkeypatch.setenv("APP_PASSWORD", "SecretAdminPass123!")

    def _app() -> None:
        from unittest.mock import patch

        import streamlit as st
        from src.security.auth import generate_admin_token
        from src.ui.app import run_streamlit_app

        st.session_state["auth_token"] = generate_admin_token("SecretAdminPass123!")
        with patch("src.checks.coordinator.FleetCheckCoordinator.check_all_apps", return_value=[]):
            run_streamlit_app()

    at = AppTest.from_function(_app).run()
    assert not at.exception

    sidebar_labels = [b.label for b in at.sidebar.button]
    assert sidebar_labels == ["Dashboard", "DB-Inspektor", "KI-Brain", "Abmelden"]
    assert "System & Historie" not in sidebar_labels
    assert "KI-View" not in sidebar_labels
    assert [b.proto.icon for b in at.sidebar.button] == ["📡", "🗄️", "🧠", "🚪"]
    assert at.sidebar.button[0].proto.type == "primary"
    # Trennlinie (st.divider) unterhalb von KI-Brain vor Abmelden vorhanden
    assert len(at.sidebar.divider) == 1

    ki_nav_btn = next(b for b in at.sidebar.button if b.label == "KI-Brain")
    at = ki_nav_btn.click().run()
    assert not at.exception
    assert "🧠 KI-Brain" in [h.value for h in at.header]
    active_ki_btn = next(b for b in at.sidebar.button if b.label == "KI-Brain")
    assert active_ki_btn.proto.type == "primary"


def test_no_deprecated_use_container_width_and_mobile_ki_kpi_css() -> None:
    from src.ui.styles import _STATIC_CSS

    src_root = Path(__file__).resolve().parent.parent / "src"
    offenders = [
        str(p)
        for p in src_root.rglob("*.py")
        if "use_container_width" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert ".st-key-ki_kpi_row div[data-testid=\"stHorizontalBlock\"]" in _STATIC_CSS


def test_ki_ui_cache_prevents_redundant_loads_and_invalidates_on_write(tmp_path: Path) -> None:
    from src.ui.views.ki_task_overlay import invalidate_ki_ui_cache, load_cached_ui_state

    repo = create_brain_repository(sqlite_path=tmp_path / "cache_ui.db")
    brain = KIWebsearchBrain.for_dashboard(repo)
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


def test_ki_view_benchmark_report_in_check_tab_and_quota_reset_in_tools_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain.models import BenchmarkEntry, BenchmarkReport, ProbeStatus
    from ki_brain.quotas import record_quota_usage
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "bench_check_tab.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    repo = create_brain_repository(sqlite_path=db_file)
    record_quota_usage(repo, tool_counts={"gemini_google_maps": 25})
    repo.save_benchmark_run(
        BenchmarkReport(
            run_id="bench_tab5_run",
            started_at="2026-11-18T10:00:00+00:00",
            finished_at="2026-11-18T10:00:05+00:00",
            test_query="Python AI",
            test_url="https://de.wikipedia.org/wiki/Python",
            entries=[
                BenchmarkEntry(
                    item_type="tool",
                    item_id="yahoo_search",
                    name="Yahoo Search",
                    status=ProbeStatus.OK,
                    latency_ms=120,
                    result_count=5,
                    quality_score=91.5,
                    details="5 Treffer",
                )
            ],
            healthy_tools_count=1,
            healthy_models_count=0,
        )
    )
    repo.close()

    def _app() -> None:
        import os

        from src.config.settings import AppSettings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        render_ki_view(AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"]), role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception

    # Check-Tab (tabs[4]) zeigt letzten Benchmark-Report
    assert any("bench_tab5_run" in c.value for c in at.tabs[4].caption)
    bench_df = at.tabs[4].dataframe[0].value
    assert "yahoo_search" in list(bench_df["ID"])

    # Tools-Tab (tabs[2]) zeigt Kontingent & Reset-Button
    tools_df_before = at.tabs[2].dataframe[0].value
    assert any("4975/5000/Mo" in str(v) for v in tools_df_before["Kontingent / Kosten"])
    reset_q_btn = next(b for b in at.tabs[2].button if b.key == "ki_reset_usage_quotas_btn")
    at = reset_q_btn.click().run()
    assert not at.exception
    tools_df_after = at.tabs[2].dataframe[0].value
    assert any("5000/5000/Mo" in str(v) for v in tools_df_after["Kontingent / Kosten"])


def test_ki_vault_shows_gemini_env_redundancy_and_readonly_dialog_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "vault_env_hint.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyLegacyEnvKey")
    with create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file))) as b:
        b.set_secret("GEMINI_FREE_API_KEY", "AIzaSyVaultFreeKey")

    def _app() -> None:
        from src.config.settings import load_app_settings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        render_ki_view(load_app_settings(allow_sibling_env_fallback=False), role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception
    assert any("App Settings → Secrets" in i.value for i in at.tabs[3].info)

    # Klick auf Legacy GEMINI_API_KEY (stammt nur aus ENV/Secrets) zeigt klaren Hinweis statt Dialog-Lösch-Button
    legacy_btn = next(b for b in at.tabs[3].button if b.key == "ki_vault_row_btn_GEMINI_API_KEY")
    at = legacy_btn.click().run()
    assert not at.exception
    assert not any(b.key == "ki_dlg_del_btn_GEMINI_API_KEY" for b in at.button)
    assert any("stammt aktuell aus **Streamlit Secrets" in i.value for i in at.info)


def test_ki_tools_tab_single_tool_probe_button(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain.models import BrainFetchedPage, ProbeStatus
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "single_probe_ui.db"
    monkeypatch.setenv("BRAIN_SQLITE_PATH", str(db_file))
    with create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file))) as b:
        b.set_secret("BROWSERBASE_API_KEY", "bb_ui_single_test_key")

    monkeypatch.setattr(
        "ki_brain.benchmark.execute_extractor_tool",
        lambda tool, url, secrets=None: BrainFetchedPage(
            url=url, title="Wiki", text_content="Sauberer Testtext " * 15, extractor_used=tool.id, latency_ms=95
        ),
    )

    def _app() -> None:
        import os

        from src.config.settings import AppSettings
        from src.domain.enums import AuthRole
        from src.ui.views.ki_view import render_ki_view

        render_ki_view(AppSettings(brain_sqlite_path=os.environ["BRAIN_SQLITE_PATH"]), role=AuthRole.ADMIN)

    at = AppTest.from_function(_app).run()
    assert not at.exception
    tool_sel = next(s for s in at.tabs[2].selectbox if s.key == "ki_tool_quick_sel")
    tool_sel.set_value("browserbase_fetch").run()
    probe_btn = next(b for b in at.tabs[2].button if b.key == "ki_tool_probe_btn")
    at = probe_btn.click().run()
    assert not at.exception
    assert any("browserbase_fetch" in s.value and "OK" in s.value for s in at.success)

    reopened = create_dashboard_brain(AppSettings(brain_sqlite_path=str(db_file)))
    bb = reopened.get_tool("browserbase_fetch")
    assert bb is not None and bb.last_status == ProbeStatus.OK
    assert reopened.load_usage_quotas()["tools"]["browserbase_fetch"]["used"] == 1
    reopened.close()

