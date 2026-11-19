"""Umfassende Tests für das KI & Websearch Brain Framework (`ki_brain`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from ki_brain import (
    AIModelSpec,
    BrainSecrets,
    BrainSSRFError,
    BrainTier,
    BrainValidationError,
    KIWebsearchBrain,
    ModelProvider,
    ProbeStatus,
    ToolCategory,
    ToolSpec,
    create_brain_repository,
)
from ki_brain.security import (
    is_public_ip_address,
    resolve_safe_sqlite_path,
    validate_safe_external_url,
)
from ki_brain.tools.registry import execute_search_tool
from pydantic import BaseModel


class _SampleSchema(BaseModel):
    topic: str
    score: int


def test_ki_brain_security_ssrf_and_path_traversal(tmp_path: Path) -> None:
    assert not is_public_ip_address("127.0.0.1")
    assert not is_public_ip_address("169.254.169.254")
    assert not is_public_ip_address("10.1.2.3")
    assert is_public_ip_address("1.1.1.1")

    with pytest.raises(BrainSSRFError):
        validate_safe_external_url("http://127.0.0.1:8080/admin")
    with pytest.raises(BrainSSRFError):
        validate_safe_external_url("https://metadata.google.internal/computeMetadata/v1/")
    with pytest.raises(BrainSSRFError):
        validate_safe_external_url(
            "https://rebind.example.com",
            dns_resolver=lambda _host, _port: ["10.0.0.5"],
        )

    assert "secret" not in repr(BrainSecrets(gemini_api_key="AIzaSySecret123", groq_api_key="gsk_secret"))
    with pytest.raises(BrainValidationError):
        resolve_safe_sqlite_path(tmp_path, "../../etc/passwd")


def test_default_catalog_contains_research_and_alert_bot_models_and_tools(tmp_path: Path) -> None:
    db_file = tmp_path / "ki_brain.db"
    repo = create_brain_repository(sqlite_path=db_file)
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())

    model_ids = {m.id for m in brain.list_models()}
    for expected in (
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-4-hydrogen",
        "gemini-2.5-flash",
        "deep-research-preview-04-2026",
    ):
        assert expected in model_ids
    assert "gemini-2.0-flash" not in model_ids
    assert "gemini-4-argon" not in model_ids

    tool_ids = {t.id for t in brain.list_tools()}
    for expected_tool in (
        "yahoo_search",
        "searxng_public",
        "duckduckgo_html",
        "duckduckgo_lite",
        "duckduckgo_instant",
        "google_news_rss",
        "wikipedia_api",
        "wikidata_api",
        "hackernews_algolia",
        "reddit_search",
        "arxiv_api",
        "direct_http_fetch",
        "jina_reader",
        "browserbase_fetch",
        "gemini_google_search",
        "gemini_google_maps",
        "gemini_url_context",
        "open_meteo",
    ):
        assert expected_tool in tool_ids

    deep_agent = brain.get_model("deep-research-preview-04-2026")
    maps_tool = brain.get_tool("gemini_google_maps")
    assert deep_agent is not None and deep_agent.is_free is False
    assert maps_tool is not None and maps_tool.free_quota_limit == 5000 and maps_tool.free_quota_period == "monthly"

    # Persistente Tombstone-Löschung + Reset
    assert brain.remove_tool("wikidata_api") is True
    brain.close()

    reopened_repo = create_brain_repository(sqlite_path=db_file)
    reopened = KIWebsearchBrain.for_dashboard(reopened_repo, BrainSecrets())
    assert "wikidata_api" not in {t.id for t in reopened.list_tools()}
    reopened.reset_defaults()
    assert "wikidata_api" in {t.id for t in reopened.list_tools()}
    reopened.close()


def test_dynamic_crud_custom_tools_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = create_brain_repository(sqlite_path=":memory:")
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())

    custom_tool = ToolSpec(
        id="custom_tech_api",
        name="Custom Tech Search API",
        category=ToolCategory.WEB_SEARCH,
        executor_type="custom_json_api",
        endpoint_url="https://api.example.com/search",
        config={"query_param": "q", "items_path": "data.items", "title_field": "name", "url_field": "link", "snippet_field": "summary"},
    )
    saved_tool = brain.add_or_update_tool(custom_tool)
    assert saved_tool.id == "custom_tech_api"

    with pytest.raises(BrainSSRFError):
        brain.add_or_update_tool(custom_tool.model_copy(update={"id": "bad_tool", "endpoint_url": "http://127.0.0.1/"}))

    monkeypatch.setattr(
        "ki_brain.tools.extractors.safe_http_request",
        lambda *_a, **_kw: (
            200,
            "https://api.example.com/search",
            "application/json",
            b'{"data": {"items": [{"name": "Test Hit", "link": "https://example.org/a", "summary": "Detail"}]}}',
        ),
    )
    hits = execute_search_tool(saved_tool, "AI Agents")
    assert len(hits) == 1
    assert hits[0].title == "Test Hit"
    assert hits[0].url == "https://example.org/a"

    rss_tool = ToolSpec(
        id="custom_rss_feed_1",
        name="Custom RSS Feed",
        category=ToolCategory.NEWS_SEARCH,
        executor_type="custom_rss_feed",
        endpoint_url="https://feeds.example.org/news.xml",
    )
    monkeypatch.setattr(
        "ki_brain.tools.extractors.safe_http_request",
        lambda *_a, **_kw: (
            200,
            "https://feeds.example.org/news.xml",
            "application/xml",
            b"<rss><channel><item><title>AI Update</title><link>https://news.example.org/1</link><description>Neu</description></item></channel></rss>",
        ),
    )
    rss_hits = execute_search_tool(rss_tool, "AI")
    assert len(rss_hits) == 1 and rss_hits[0].title == "AI Update"

    custom_model = AIModelSpec(
        id="groq-llama-3.3-70b",
        name="Groq Llama 3.3 70B",
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model_id="llama-3.3-70b-versatile",
        tier_hint=BrainTier.STANDARD,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    )
    brain.add_or_update_model(custom_model)
    assert brain.get_model("groq-llama-3.3-70b") is not None
    brain.set_model_enabled("groq-llama-3.3-70b", False)
    assert brain.get_model("groq-llama-3.3-70b").enabled is False  # type: ignore[union-attr]
    brain.close()


def test_auto_sync_discovers_gemini_groq_and_openrouter_models(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = create_brain_repository(sqlite_path=":memory:")
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        SimpleNamespace(name="models/gemini-3.9-flash", display_name="Gemini 3.9 Flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.9-dead", display_name="Gemini Dead", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.1-flash-lite-preview", display_name="Gemini 3.1 Flash Lite Preview", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/text-embedding-004", display_name="Embed", supported_actions=["embedContent"]),
    ]
    probed_gemini_calls: list[str] = []

    def _fake_gemini_create(**kwargs: object) -> MagicMock:
        mid = str(kwargs.get("model") or "")
        probed_gemini_calls.append(mid)
        if mid == "gemini-3.9-dead":
            raise RuntimeError("404 NOT_FOUND model unavailable")
        resp = MagicMock()
        resp.output_text = "OK"
        resp.steps = []
        return resp

    mock_client.interactions.create.side_effect = _fake_gemini_create
    secrets = BrainSecrets(
        gemini_api_key="AIzaSyTestKey123",
        groq_api_key="gsk_test_123",
        openrouter_api_key="sk-or-test-123",
    )
    brain = KIWebsearchBrain.for_dashboard(repo, secrets, genai_client=mock_client)

    def _fake_http(_method: str, url: str, json_body: dict[str, object] | None = None, **_kw: object) -> tuple[int, str, str, bytes]:
        if url.endswith("/chat/completions"):
            req_model = str((json_body or {}).get("model") or "")
            if "dead" in req_model:
                return 429, url, "application/json", b'{"error": "rate_limit"}'
            return 200, url, "application/json", b'{"choices": [{"message": {"content": "OK"}}]}'
        if "groq.com" in url:
            return 200, url, "application/json", b'{"data": [{"id": "llama-3.1-8b-instant"}]}'
        if "openrouter.ai" in url:
            return (
                200,
                url,
                "application/json",
                b'{"data": [{"id": "qwen/qwen-2.5:free", "name": "Qwen Free", "pricing": {"prompt": "0", "completion": "0"}}, {"id": "broken/dead:free", "name": "Broken Dead", "pricing": {"prompt": "0", "completion": "0"}}]}',
            )
        return 404, url, "text/plain", b""

    monkeypatch.setattr("ki_brain.discovery.safe_http_request", _fake_http)
    monkeypatch.setattr("ki_brain.llm_runner.safe_http_request", _fake_http)

    summary = brain.auto_sync()
    assert "gemini-3.9-flash" in summary.added_models
    assert "groq/llama-3.1-8b-instant" in summary.added_models
    assert "or/qwen/qwen-2.5-free" in summary.added_models
    assert "gemini-3.9-dead" in summary.skipped_unhealthy
    assert "or/broken/dead-free" in summary.skipped_unhealthy
    assert "gemini-3.1-flash-lite-preview" in summary.skipped_previews
    assert brain.get_model("gemini-3.9-flash") is not None
    assert brain.get_model("gemini-3.9-flash").last_status == ProbeStatus.OK  # type: ignore[union-attr]
    assert brain.get_model("gemini-3.9-dead") is None
    assert brain.get_model("gemini-3.1-flash-lite-preview") is None
    assert brain.get_model("or/broken/dead-free") is None
    assert brain.get_model("text-embedding-004") is None

    # Manuell gelöschte + defekt verworfene Modelle kommen beim nächsten Auto-Sync NICHT zurück
    assert brain.remove_model("gemini-3.9-flash") is True
    probed_gemini_calls.clear()
    summary_again = brain.auto_sync()
    assert brain.get_model("gemini-3.9-flash") is None
    assert brain.get_model("gemini-3.9-dead") is None
    assert brain.get_model("or/broken/dead-free") is None
    assert "gemini-3.9-flash" in summary_again.skipped_excluded
    assert "gemini-3.9-dead" in summary_again.skipped_excluded
    assert "or/broken/dead-free" in summary_again.skipped_excluded
    assert probed_gemini_calls == []

    # Wenn exclude_previews=False, kann Preview aufgenommen und manuell wieder entfernt werden
    summary_prev = brain.auto_sync(exclude_previews=False)
    assert "gemini-3.1-flash-lite-preview" in summary_prev.added_models
    assert brain.remove_model("gemini-3.1-flash-lite-preview") is True
    assert brain.get_model("deep-research-preview-04-2026") is not None
    brain.close()


def test_benchmark_optimizes_fast_standard_deep_and_demotes_quota_429(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = create_brain_repository(sqlite_path=":memory:")
    mock_client = MagicMock()
    mock_client.models.list.return_value = [SimpleNamespace(name="models/gemini-3.5-flash-lite")]

    def _fake_create(**kwargs: object) -> MagicMock:
        tools_arg = kwargs.get("tools") or []
        if tools_arg:
            raise RuntimeError("429 TOO_MANY_REQUESTS quota billing required")
        resp = MagicMock()
        resp.output_text = "OK"
        resp.steps = []
        return resp

    mock_client.interactions.create.side_effect = _fake_create
    secrets = BrainSecrets(gemini_api_key="AIzaSyTestKey123")
    brain = KIWebsearchBrain.for_dashboard(repo, secrets, genai_client=mock_client)

    # Probe_tool sauber simulieren (ohne echte Netzaufrufe)
    def _fake_search(tool: ToolSpec, *_args: object, **_kwargs: object) -> list[object]:
        from ki_brain.models import BrainCitation

        if tool.id == "duckduckgo_html":
            return []  # Simuliert Cloud-Blockade -> Blocked
        return [BrainCitation(url=f"https://example.org/{tool.id}", title=f"Hit from {tool.id}", snippet="Details", source_tool=tool.id)]

    def _fake_extract(tool: ToolSpec, url: str, **_kwargs: object) -> object:
        from ki_brain.models import BrainFetchedPage

        return BrainFetchedPage(url=url, title="Python", text_content="Python ist eine Programmiersprache " * 20, extractor_used=tool.id, latency_ms=110)

    monkeypatch.setattr("ki_brain.benchmark.execute_search_tool", _fake_search)
    monkeypatch.setattr("ki_brain.benchmark.execute_extractor_tool", _fake_extract)

    report = brain.run_benchmark()
    assert report.healthy_models_count >= 5
    assert report.healthy_tools_count >= 8

    # Quota-limited & Missing-Key Tools nicht in optimierten Sequenzen bevorzugt
    gg_tool = brain.get_tool("gemini_google_search")
    bb_tool = brain.get_tool("browserbase_fetch")
    ddg_tool = brain.get_tool("duckduckgo_html")
    assert gg_tool is not None and gg_tool.last_status == ProbeStatus.QUOTA_LIMITED
    assert bb_tool is not None and bb_tool.last_status == ProbeStatus.AUTH_MISSING
    assert ddg_tool is not None and ddg_tool.last_status == ProbeStatus.BLOCKED

    fast_seq = brain.get_sequence(BrainTier.FAST)
    std_seq = brain.get_sequence(BrainTier.STANDARD)
    deep_seq = brain.get_sequence(BrainTier.DEEP)

    assert fast_seq.single_pass is True
    assert fast_seq.enable_page_scraping is False
    assert fast_seq.max_search_tools == 2
    assert "deep-research-preview-04-2026" not in fast_seq.model_sequence
    assert "duckduckgo_html" not in fast_seq.search_tool_sequence

    assert std_seq.enable_page_scraping is True
    assert std_seq.max_search_tools == 4

    assert deep_seq.max_pages_to_scrape == 5
    assert deep_seq.use_deep_agent is False
    assert deep_seq.paid_use_deep_agent is True
    assert "deep-research-preview-04-2026" not in deep_seq.model_sequence
    assert "deep-research-preview-04-2026" in deep_seq.paid_model_sequence
    assert len(deep_seq.search_tool_sequence) >= 5

    # Manual Override wird respektiert, bis force_overwrite_sequences=True
    locked = fast_seq.model_copy(update={"search_tool_sequence": ["wikipedia_api"], "manual_override": True})
    brain.update_sequence(locked)
    brain.run_benchmark()
    assert brain.get_sequence(BrainTier.FAST).search_tool_sequence == ["wikipedia_api"]
    brain.run_benchmark(force_overwrite_sequences=True)
    assert len(brain.get_sequence(BrainTier.FAST).search_tool_sequence) > 1
    brain.close()


def test_brain_execution_search_ask_structured_and_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = create_brain_repository(sqlite_path=":memory:")
    mock_client = MagicMock()
    mock_interaction = MagicMock()
    mock_interaction.output_text = "Zusammenfassung aus Quellen [Wikipedia](https://de.wikipedia.org/wiki/Python)."
    mock_interaction.steps = []
    mock_client.interactions.create.return_value = mock_interaction

    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets(gemini_api_key="AIzaSyTest123"), genai_client=mock_client)

    from ki_brain.models import BrainCitation, BrainFetchedPage

    monkeypatch.setattr(
        "ki_brain.execution.execute_search_tool",
        lambda tool, q, **_kw: [BrainCitation(url=f"https://example.org/{tool.id}", title=f"{tool.id} {q}", snippet="Info", source_tool=tool.id)],
    )
    monkeypatch.setattr(
        "ki_brain.execution.execute_extractor_tool",
        lambda tool, url, **_kw: BrainFetchedPage(url=url, title="Page", text_content="Volltext Inhalt " * 10, extractor_used=tool.id, latency_ms=50),
    )

    search_res = brain.search("Python 3.13", tier=BrainTier.FAST)
    assert len(search_res.tools_executed) == 2
    assert len(search_res.citations) >= 2

    ask_res = brain.ask("Was ist neu in Python?", tier=BrainTier.STANDARD)
    assert "Zusammenfassung" in ask_res.text
    assert ask_res.pages_fetched >= 1
    assert ask_res.used_heuristic_fallback is False

    # Structured Output
    mock_interaction.output_text = '{"topic": "KI", "score": 95}'
    parsed, used_model = brain.generate_structured("Bewerte KI", _SampleSchema, tier=BrainTier.FAST)
    assert parsed.topic == "KI" and parsed.score == 95
    assert used_model

    # Export & Import Roundtrip
    exported_json = brain.export_snapshot_json()
    exported_yaml = brain.export_snapshot_yaml()
    assert "yahoo_search" in exported_yaml
    brain.remove_tool("arxiv_api")
    brain.import_snapshot_json(exported_json)
    assert brain.get_tool("arxiv_api") is not None

    # Heuristik-Fallback ohne API Keys
    no_key_brain = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=":memory:"), BrainSecrets())
    fb_res = no_key_brain.ask("Python", tier=BrainTier.FAST)
    assert fb_res.used_heuristic_fallback is True
    assert "Recherche-Ergebnis" in fb_res.text
    no_key_brain.close()
    brain.close()


def test_cleanup_unhealthy_items_and_write_only_secrets_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ki_brain.models import BenchmarkEntry, BenchmarkReport, BrainFetchedPage

    db_file = tmp_path / "vault_and_cleanup.db"
    repo = create_brain_repository(sqlite_path=db_file)
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())

    # 1. Modelle & Tools auf Error / Quota 429 / HTTP 202 / sonstige Blocked / Missing Key setzen und bereinigen
    m_err = brain.get_model("gemini-3.6-flash")
    t_q429 = brain.get_tool("gemini_google_search")
    t_202 = brain.get_tool("duckduckgo_lite")
    t_blocked = brain.get_tool("duckduckgo_html")
    t_missing = brain.get_tool("browserbase_fetch")
    assert m_err is not None and t_q429 is not None and t_202 is not None and t_blocked is not None and t_missing is not None

    brain.add_or_update_model(m_err.model_copy(update={"last_status": ProbeStatus.ERROR}))
    brain.add_or_update_tool(t_q429.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED}))
    brain.add_or_update_tool(
        t_202.model_copy(
            update={"last_status": ProbeStatus.BLOCKED, "last_error": "HTTP 202 (Anti-Bot-Challenge / Leerantwort auf lite.duckduckgo.com)"}
        )
    )
    brain.add_or_update_tool(t_blocked.model_copy(update={"last_status": ProbeStatus.BLOCKED, "last_error": "0 Treffer (kein 202)"}))
    brain.add_or_update_tool(t_missing.model_copy(update={"last_status": ProbeStatus.AUTH_MISSING}))

    repo.save_benchmark_run(
        BenchmarkReport(
            run_id="init_run",
            started_at="2026-11-14T10:00:00+00:00",
            finished_at="2026-11-14T10:00:05+00:00",
            test_query="Python AI",
            test_url="https://de.wikipedia.org/wiki/Python_(Programmiersprache)",
            entries=[
                BenchmarkEntry(
                    item_type="tool",
                    item_id="browserbase_fetch",
                    name=t_missing.name,
                    status=ProbeStatus.AUTH_MISSING,
                    latency_ms=0,
                    quality_score=0.0,
                    details="Key BROWSERBASE_API_KEY fehlt.",
                )
            ],
            healthy_tools_count=0,
            healthy_models_count=0,
        )
    )

    rem_models, rem_tools = brain.cleanup_unhealthy_items()
    assert rem_models == ["gemini-3.6-flash"]
    assert set(rem_tools) == {"duckduckgo_lite", "gemini_google_search"}
    assert brain.get_model("gemini-3.6-flash") is None
    assert brain.get_tool("duckduckgo_lite") is None
    assert brain.get_tool("gemini_google_search") is None
    # Sonstige Blocked-Tools (ohne HTTP 202) und Missing-Key-Einträge bleiben erhalten!
    kept_blocked = brain.get_tool("duckduckgo_html")
    assert kept_blocked is not None and kept_blocked.last_status == ProbeStatus.BLOCKED
    assert brain.get_tool("browserbase_fetch") is not None

    # Nach Cleanup darf auto_sync bereinigte Items nicht wieder aufnehmen
    sync_after_clean = brain.auto_sync()
    assert brain.get_model("gemini-3.6-flash") is None
    assert brain.get_tool("duckduckgo_lite") is None
    assert brain.get_tool("gemini_google_search") is None
    assert "gemini-3.6-flash" in sync_after_clean.skipped_excluded
    assert "duckduckgo_lite" in sync_after_clean.skipped_excluded
    assert "gemini_google_search" in sync_after_clean.skipped_excluded

    # 2. Write-Only Secret eintragen + sofortigen Direkt-Test ausführen
    raw_secret = "bb_live_super_secret_token_999"
    brain.set_secret("BROWSERBASE_API_KEY", raw_secret)

    statuses = {s.env_name: s for s in brain.list_secret_statuses()}
    bb_status = statuses["BROWSERBASE_API_KEY"]
    assert bb_status.is_configured is True
    assert "Brain Vault" in bb_status.source
    assert raw_secret not in bb_status.model_dump_json()

    raw_db_row = repo._executor.execute("SELECT encrypted_value FROM brain_secrets WHERE env_name = ?", ("BROWSERBASE_API_KEY",)).rows[0]  # noqa: SLF001
    assert str(raw_db_row["encrypted_value"]).startswith("enc_v2:")
    assert raw_secret not in str(raw_db_row["encrypted_value"])

    unlocked_tool = brain.get_tool("browserbase_fetch")
    assert unlocked_tool is not None and unlocked_tool.last_status == ProbeStatus.UNTESTED

    monkeypatch.setattr(
        "ki_brain.benchmark.execute_extractor_tool",
        lambda tool, url, secrets=None: BrainFetchedPage(
            url=url, title="Python", text_content="Python Inhalt " * 20, extractor_used=tool.id, latency_ms=95
        ),
    )
    probed_entries = brain.probe_items_for_secret("BROWSERBASE_API_KEY")
    assert len(probed_entries) == 1
    assert probed_entries[0].item_id == "browserbase_fetch"
    assert probed_entries[0].status == ProbeStatus.OK
    assert brain.get_tool("browserbase_fetch").last_status == ProbeStatus.OK  # type: ignore[union-attr]
    latest_bench = brain.get_latest_benchmark()
    assert latest_bench is not None and latest_bench.healthy_tools_count == 1

    # Komplettes Überschreiben mit neuem Key & Persistenz über Neustart
    brain.set_secret("BROWSERBASE_API_KEY", "bb_live_overwritten_token_111")
    brain.close()

    reopened = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=db_file), BrainSecrets())
    assert reopened._secrets.resolve_key("BROWSERBASE_API_KEY") == "bb_live_overwritten_token_111"  # noqa: SLF001
    assert reopened.delete_secret("BROWSERBASE_API_KEY") is True
    assert not reopened._secrets.has_key("BROWSERBASE_API_KEY")  # noqa: SLF001
    reopened.close()


def test_http_probe_diagnostics_and_ddg_entity_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from ki_brain.benchmark import BrainBenchmarkRunner
    from ki_brain.llm_runner import UnifiedLLMRunner
    from ki_brain.tools.search_web import search_duckduckgo_instant

    # 1. DuckDuckGo Instant Answer Einzelwort-Retry bei Multi-Word-Query
    ddg_queries: list[str] = []

    def _fake_ddg_http(_method: str, _url: str, params: dict[str, object] | None = None, **_kw: object) -> tuple[int, str, str, bytes]:
        q = str((params or {}).get("q") or "")
        ddg_queries.append(q)
        if q == "Python":
            return 200, "https://api.duckduckgo.com/", "application/json", b'{"Heading": "Python", "AbstractURL": "https://www.python.org/", "AbstractText": "Programmiersprache"}'
        return 200, "https://api.duckduckgo.com/", "application/json", b'{"AbstractURL": "", "AbstractText": "", "RelatedTopics": []}'

    monkeypatch.setattr("ki_brain.tools.search_web.safe_http_request", _fake_ddg_http)
    ddg_hits = search_duckduckgo_instant("Python AI Tools")
    assert ddg_queries == ["Python AI Tools", "Python"]
    assert len(ddg_hits) == 1 and ddg_hits[0].url == "https://www.python.org/"

    # 2. HTTP-Diagnose bei Blocked / Quota 429 im BenchmarkRunner
    repo = create_brain_repository(sqlite_path=":memory:")
    secrets = BrainSecrets()
    bench = BrainBenchmarkRunner(repo, secrets, UnifiedLLMRunner(secrets))
    ddg_html_tool = repo.get_tool("duckduckgo_html")
    assert ddg_html_tool is not None

    from ki_brain.tools.http_utils import record_http_probe

    def _fake_probe_search(tool: ToolSpec, q: str, **_kw: object) -> list[object]:
        if tool.id == "duckduckgo_html":
            record_http_probe("https://html.duckduckgo.com/html/", 202, 0)
            return []
        if tool.id == "yahoo_search":
            record_http_probe("https://de.search.yahoo.com/search", 429, 120)
            return []
        if tool.id == "duckduckgo_instant":
            assert q == "Python"
            record_http_probe("https://api.duckduckgo.com/", 200, 512)
            return []
        return []

    monkeypatch.setattr("ki_brain.benchmark.execute_search_tool", _fake_probe_search)
    ut_202, entry_202 = bench.probe_tool(ddg_html_tool, test_query="Beliebige Query")
    assert ut_202.last_status == ProbeStatus.BLOCKED
    assert "HTTP 202" in entry_202.details

    yahoo_tool = repo.get_tool("yahoo_search")
    assert yahoo_tool is not None
    ut_429, entry_429 = bench.probe_tool(yahoo_tool)
    assert ut_429.last_status == ProbeStatus.QUOTA_LIMITED
    assert "HTTP 429" in entry_429.details

    ddg_inst = repo.get_tool("duckduckgo_instant")
    assert ddg_inst is not None
    ut_200, entry_200 = bench.probe_tool(ddg_inst, test_query="Mehrwort Query")
    assert ut_200.last_status == ProbeStatus.BLOCKED
    assert "HTTP 200" in entry_200.details and "Python" in entry_200.details
    repo.close()


def test_keyless_reddit_fallbacks_and_no_reddit_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from ki_brain.models import BrainCitation
    from ki_brain.tools.search_knowledge import search_reddit

    repo = create_brain_repository(sqlite_path=":memory:")
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())
    secret_names = {s.env_name for s in brain.list_secret_statuses()}
    assert "REDDIT_CLIENT_ID" not in secret_names
    assert "REDDIT_CLIENT_SECRET" not in secret_names
    assert {"BRAVE_SEARCH_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"} <= secret_names
    reddit_spec = brain.get_tool("reddit_search")
    assert reddit_spec is not None and reddit_spec.requires_api_key is False

    # Stufe 1: PullPush liefert Treffer ohne API-Key
    monkeypatch.setattr(
        "ki_brain.tools.search_knowledge.safe_http_request",
        lambda _m, url, **_kw: (
            200,
            url,
            "application/json",
            b'{"data": [{"title": "Local LLM Benchmarks", "full_link": "https://www.reddit.com/r/LocalLLaMA/comments/abc/test/", "subreddit": "LocalLLaMA", "score": 88, "selftext": "Vergleich"}]}',
        )
        if "pullpush.io" in url
        else (403, url, "text/plain", b""),
    )
    hits_pp = search_reddit("Local LLM", BrainSecrets())
    assert len(hits_pp) == 1 and "r/LocalLLaMA" in hits_pp[0].title

    # Stufe 2: PullPush blockiert -> Reddit Atom RSS funktioniert
    rss_xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Python Tip</title><link href="https://www.reddit.com/r/Python/comments/xyz/tip/"/></entry></feed>'
    monkeypatch.setattr(
        "ki_brain.tools.search_knowledge.safe_http_request",
        lambda _m, url, **_kw: (200, url, "application/atom+xml", rss_xml)
        if "reddit.com/search.rss" in url
        else (403, url, "text/plain", b""),
    )
    hits_rss = search_reddit("Python", BrainSecrets())
    assert len(hits_rss) == 1 and hits_rss[0].url.startswith("https://www.reddit.com/r/Python")

    # Stufe 3: PullPush + RSS/JSON blockiert -> Yahoo site:reddit.com Fallback
    monkeypatch.setattr(
        "ki_brain.tools.search_knowledge.safe_http_request",
        lambda *_a, **_kw: (403, "https://blocked", "text/plain", b""),
    )
    monkeypatch.setattr(
        "ki_brain.tools.search_knowledge.search_yahoo",
        lambda q, **_kw: [BrainCitation(url="https://www.reddit.com/r/de/comments/123/", title="Reddit Thread", snippet="Text", source_tool="yahoo_search")]
        if "site:reddit.com" in q
        else [],
    )
    hits_site = search_reddit("KI News", BrainSecrets())
    assert len(hits_site) == 1 and "Reddit:" in hits_site[0].title
    brain.close()


def test_brave_tavily_serper_and_sync_new_builtins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    db_file = tmp_path / "sync_apis.db"
    repo = create_brain_repository(sqlite_path=db_file)
    # Simuliere bestehende ältere DB vor v0.8.0 (known_builtin_ids kannte neue APIs noch nicht)
    repo.set_meta("known_builtin_ids", json.dumps({"models": ["gemini-3.5-flash-lite"], "tools": ["yahoo_search"]}))
    for tid in ("brave_search", "tavily_search", "serper_search", "openalex_api", "stackexchange_api", "tagesschau_rss", "gemini_google_maps"):
        repo.delete_tool(tid)

    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())
    monkeypatch.setattr("ki_brain.discovery.safe_http_request", lambda *_a, **_kw: (404, "https://searx.space", "text/plain", b""))
    sync_sum = brain.auto_sync(verify_discovered=False)
    for new_tid in ("brave_search", "tavily_search", "serper_search", "openalex_api", "stackexchange_api", "tagesschau_rss", "gemini_google_maps"):
        assert new_tid in sync_sum.added_tools
        assert brain.get_tool(new_tid) is not None
    assert brain.get_tool("brave_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    assert brain.get_tool("tavily_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    assert brain.get_tool("serper_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    assert brain.get_tool("gemini_google_maps").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]

    # Executors mit API-Keys testen
    captured_headers: list[dict[str, str]] = []

    def _fake_api_http(_method: str, url: str, headers: dict[str, str] | None = None, **_kw: object) -> tuple[int, str, str, bytes]:
        captured_headers.append(headers or {})
        if "search.brave.com" in url:
            return 200, url, "application/json", b'{"web": {"results": [{"url": "https://brave.example.org/1", "title": "Brave Hit", "description": "Fast"}]}}'
        if "tavily.com" in url:
            return 200, url, "application/json", b'{"results": [{"url": "https://tavily.example.org/1", "title": "Tavily Hit", "content": "AI Search"}]}'
        if "serper.dev" in url:
            return 200, url, "application/json", b'{"organic": [{"link": "https://serper.example.org/1", "title": "Serper Hit", "snippet": "SERP"}]}'
        return 404, url, "text/plain", b""

    monkeypatch.setattr("ki_brain.tools.search_apis.safe_http_request", _fake_api_http)
    keyed = BrainSecrets(brave_search_api_key="brv_123", tavily_api_key="tvly_456", serper_api_key="srp_789")
    b_hits = execute_search_tool(brain.get_tool("brave_search"), "Python", secrets=keyed)  # type: ignore[arg-type]
    t_hits = execute_search_tool(brain.get_tool("tavily_search"), "Python", secrets=keyed)  # type: ignore[arg-type]
    s_hits = execute_search_tool(brain.get_tool("serper_search"), "Python", secrets=keyed)  # type: ignore[arg-type]
    assert b_hits[0].title == "Brave Hit" and captured_headers[0]["X-Subscription-Token"] == "brv_123"
    assert t_hits[0].title == "Tavily Hit"
    assert s_hits[0].title == "Serper Hit" and captured_headers[2]["X-API-KEY"] == "srp_789"
    brain.close()


def test_searx_space_discovery_ssrf_and_pool_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from ki_brain.tools.searxng import (
        build_searxng_tool,
        discover_verified_searxng_pool,
        parse_searx_space_instances,
    )

    payload = {
        "instances": {
            "http://insecure.example.org": {"network_type": "normal", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 99.9}}}},
            "https://onionservice.onion": {"network_type": "tor", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 99.9}}}},
            "https://127.0.0.1": {"network_type": "normal", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 99.9}}}},
            "https://down.example.com": {"network_type": "normal", "http": {"status_code": 503}, "timing": {"search": {"all": {"uptime": 99.9}}}},
            "https://nojson.example.com": {"network_type": "normal", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 99.0, "mean": 0.3}}}},
            "https://good-fast.example.com": {"network_type": "normal", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 99.5, "mean": 0.4}}}},
            "https://good-slow.example.com": {"network_type": "normal", "http": {"status_code": 200}, "timing": {"search": {"all": {"uptime": 95.0, "mean": 1.2}}}},
        }
    }
    parsed = parse_searx_space_instances(payload)
    assert "http://insecure.example.org/search" not in parsed
    assert "https://onionservice.onion/search" not in parsed
    assert "https://127.0.0.1/search" not in parsed
    assert "https://down.example.com/search" not in parsed
    assert parsed[0] == "https://good-fast.example.com/search"

    def _fake_searx_http(_m: str, url: str, **_kw: object) -> tuple[int, str, str, bytes]:
        if "searx.space" in url:
            return 200, url, "application/json", json.dumps(payload).encode("utf-8")
        if "nojson.example.com" in url:
            return 403, url, "text/html", b"Forbidden"
        if "good-fast.example.com" in url:
            return 200, url, "application/json", b'{"results": [{"url": "https://docs.python.org/3/", "title": "Python Docs", "content": "Official"}]}'
        if "good-slow.example.com" in url:
            return 200, url, "application/json", b'{"results": [{"url": "https://peps.python.org/", "title": "Python PEPs", "content": "PEPs"}]}'
        return 404, url, "text/plain", b""

    pool = discover_verified_searxng_pool(max_verify=6, max_keep=3, http_request_fn=_fake_searx_http)
    assert pool == ["https://good-fast.example.com/search", "https://good-slow.example.com/search"]

    searx_tool = build_searxng_tool(pool)
    called_eps: list[str] = []

    def _fake_rotated_http(_m: str, url: str, **_kw: object) -> tuple[int, str, str, bytes]:
        called_eps.append(url)
        if "good-fast" in url:
            return 429, url, "text/plain", b"Too Many Requests"
        return 200, url, "application/json", b'{"results": [{"url": "https://peps.python.org/", "title": "Python PEPs", "content": "Fallback ok"}]}'

    monkeypatch.setattr("ki_brain.tools.search_apis.safe_http_request", _fake_rotated_http)
    hits = execute_search_tool(searx_tool, "Python")
    assert called_eps == ["https://good-fast.example.com/search", "https://good-slow.example.com/search"]
    assert len(hits) == 1 and hits[0].title == "Python PEPs"


def test_resolve_blocked_tools_applies_rules_and_unblocks(monkeypatch: pytest.MonkeyPatch) -> None:
    from ki_brain.models import BenchmarkEntry, BenchmarkReport, BrainCitation
    from ki_brain.tools.http_utils import record_http_probe

    repo = create_brain_repository(sqlite_path=":memory:")
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets())

    # Simuliere alte Blocked-Einträge ohne Rules + blockiertes DDG HTML + Yahoo Alternate
    ddg_inst = brain.get_tool("duckduckgo_instant")
    ddg_html = brain.get_tool("duckduckgo_html")
    reddit_t = brain.get_tool("reddit_search")
    yahoo_t = brain.get_tool("yahoo_search")
    assert ddg_inst and ddg_html and reddit_t and yahoo_t

    brain.add_or_update_tool(ddg_inst.model_copy(update={"config": {}, "last_status": ProbeStatus.BLOCKED, "last_error": "0 Treffer"}))
    brain.add_or_update_tool(ddg_html.model_copy(update={"last_status": ProbeStatus.BLOCKED, "last_error": "0 Treffer"}))
    brain.add_or_update_tool(reddit_t.model_copy(update={"requires_api_key": True, "api_key_env": "REDDIT_CLIENT_ID", "endpoint_url": "https://oauth.reddit.com/search", "last_status": ProbeStatus.BLOCKED}))
    brain.add_or_update_tool(yahoo_t.model_copy(update={"endpoint_url": "https://de.search.yahoo.com/search", "last_status": ProbeStatus.BLOCKED}))

    repo.save_benchmark_run(
        BenchmarkReport(
            run_id="before_unblock",
            started_at="2026-11-14T10:00:00+00:00",
            finished_at="2026-11-14T10:00:02+00:00",
            test_query="Python AI",
            test_url="https://de.wikipedia.org/wiki/Python",
            entries=[BenchmarkEntry(item_type="tool", item_id="duckduckgo_html", name=ddg_html.name, status=ProbeStatus.BLOCKED, latency_ms=50, details="alt")],
            healthy_tools_count=0,
            healthy_models_count=0,
        )
    )

    monkeypatch.setattr(
        "ki_brain.unblock.discover_verified_searxng_pool",
        lambda **_kw: ["https://searx.verified.example/search"],
    )

    def _fake_unblock_exec(tool: ToolSpec, query: str, **_kw: object) -> list[BrainCitation]:
        if tool.id == "duckduckgo_instant" and query == "Python":
            return [BrainCitation(url="https://www.python.org/", title="Python", snippet="Rule ok", source_tool=tool.id)]
        if tool.id == "reddit_search" and not tool.requires_api_key:
            return [BrainCitation(url="https://www.reddit.com/r/Python/1", title="Reddit OK", snippet="Keyless", source_tool=tool.id)]
        if tool.id == "yahoo_search" and "de.search.yahoo.com" not in tool.endpoint_url:
            return [BrainCitation(url="https://python.org/", title="Yahoo US OK", snippet="Alt", source_tool=tool.id)]
        if tool.id == "searxng_public":
            return [BrainCitation(url="https://searx.verified.example/", title="SearXNG OK", snippet="Pool", source_tool=tool.id)]
        if tool.id == "duckduckgo_html":
            record_http_probe("https://html.duckduckgo.com/html/", 202, 0)
            return []
        return []

    monkeypatch.setattr("ki_brain.benchmark.execute_search_tool", _fake_unblock_exec)
    summary = brain.resolve_blocked_tools()

    assert "duckduckgo_instant" in summary.unblocked_ids
    assert "reddit_search" in summary.unblocked_ids
    assert "yahoo_search" in summary.unblocked_ids
    assert "duckduckgo_html" in summary.still_blocked
    assert "HTTP 202" in summary.still_blocked["duckduckgo_html"]
    assert any("searxng_public" in a for a in summary.activated_alternatives)

    saved_inst = brain.get_tool("duckduckgo_instant")
    saved_reddit = brain.get_tool("reddit_search")
    saved_searx = brain.get_tool("searxng_public")
    assert saved_inst is not None and saved_inst.config.get("probe_query") == "Python"
    assert saved_reddit is not None and saved_reddit.requires_api_key is False
    assert saved_searx is not None and saved_searx.last_status == ProbeStatus.OK
    assert "searxng_public" in brain.get_sequence(BrainTier.FAST).search_tool_sequence
    brain.close()


def test_llm_runner_transient_503_retry_and_status_specific_cleanup() -> None:
    from ki_brain.exceptions import BrainProviderError
    from ki_brain.llm_runner import UnifiedLLMRunner

    mock_client = MagicMock()
    calls = {"count": 0}

    def _flaky_create(**_kwargs: object) -> MagicMock:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 UNAVAILABLE temporary model capacity spike")
        resp = MagicMock()
        resp.status = "completed"
        resp.output_text = "Stabil"
        resp.steps = []
        return resp

    mock_client.interactions.create.side_effect = _flaky_create
    runner = UnifiedLLMRunner(BrainSecrets(gemini_api_key="AIzaSyTest123"), genai_client=mock_client)
    spec = AIModelSpec(
        id="gemini-3.5-flash-lite",
        name="Gemini 3.5 Flash-Lite",
        provider=ModelProvider.GOOGLE_GEMINI,
        model_id="gemini-3.5-flash-lite",
    )
    txt, _ = runner.run_text(spec, "Ping")
    assert txt == "Stabil"
    assert calls["count"] == 2

    mock_client.interactions.create.side_effect = RuntimeError("404 NOT_FOUND unavailable")
    with pytest.raises(RuntimeError):
        runner.run_text(spec, "Ping")
    assert mock_client.interactions.create.call_count == 3  # Kein Retry bei 404

    mock_client.interactions.create.side_effect = lambda **_kw: SimpleNamespace(status="failed", errors="internal error", output_text="", steps=[])
    with pytest.raises(BrainProviderError):
        runner.run_text(spec, "Ping")


def test_searxng_html_fallback_open_meteo_and_legacy_db_upgrade(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ki_brain.db_executors import SqliteBrainExecutor
    from ki_brain.storage import BrainRepository
    from ki_brain.tools.extractors import query_open_meteo
    from ki_brain.tools.searxng import build_searxng_tool, discover_verified_searxng_pool

    # 1. SearXNG nutzt HTML-Fallback, wenn searx.space blockiert und Instanzen 403 für JSON liefern
    def _html_only_http(_m: str, url: str, params: dict[str, object] | None = None, **_kw: object) -> tuple[int, str, str, bytes]:
        if "searx.space" in url:
            return 403, url, "text/plain", b"Forbidden"
        if (params or {}).get("format") == "json":
            return 403, url, "text/plain", b"JSON disabled"
        return (
            200,
            url,
            "text/html",
            b'<article class="result"><h3><a href="https://python.org/">Python.org</a></h3><p class="content">Offizielle Seite</p></article>',
        )

    pool = discover_verified_searxng_pool(max_verify=3, max_keep=2, http_request_fn=_html_only_http)
    assert len(pool) == 2

    monkeypatch.setattr("ki_brain.tools.search_apis.safe_http_request", _html_only_http)
    searx_tool = build_searxng_tool(pool)
    hits = execute_search_tool(searx_tool, "Python")
    assert len(hits) == 1 and hits[0].url == "https://python.org/"

    # 2. Open-Meteo fällt bei blockierter Geocoding-API auf eingebaute Stadtkoordinaten zurück
    def _meteo_http(_m: str, url: str, **_kw: object) -> tuple[int, str, str, bytes]:
        if "geocoding-api.open-meteo.com" in url:
            return 429, url, "text/plain", b"Rate limit"
        return 200, url, "application/json", b'{"current": {"temperature_2m": 16.4, "wind_speed_10m": 12.0, "precipitation": 0.0}}'

    monkeypatch.setattr("ki_brain.tools.extractors.safe_http_request", _meteo_http)
    weather_hits = query_open_meteo("Berlin")
    assert len(weather_hits) == 1 and "16.4" in weather_hits[0].snippet

    # 3. Bestehende Alt-DB (seeded_v1='1' ohne seeded_catalog_rev) erhält searxng_public automatisch ohne Excluded-IDs zu reaktivieren
    legacy_db = tmp_path / "legacy_v1.db"
    ex = SqliteBrainExecutor(legacy_db)
    ex.execute_batch(
        [
            ("CREATE TABLE brain_models (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)", ()),
            ("CREATE TABLE brain_tools (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)", ()),
            ("CREATE TABLE brain_sequences (tier TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)", ()),
            ("CREATE TABLE brain_benchmark_runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, payload_json TEXT NOT NULL)", ()),
            ("CREATE TABLE brain_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)", ()),
            ("CREATE TABLE brain_secrets (env_name TEXT PRIMARY KEY, encrypted_value TEXT NOT NULL, updated_at TEXT NOT NULL)", ()),
            ("INSERT INTO brain_meta (key, value) VALUES ('seeded_v1', '1')", ()),
            ("INSERT INTO brain_meta (key, value) VALUES ('known_builtin_ids', '{\"models\": [\"gemini-3.5-flash-lite\"], \"tools\": [\"yahoo_search\"]}')", ()),
            ("INSERT INTO brain_meta (key, value) VALUES ('excluded_ids', '{\"models\": [\"gemini-3.6-flash\"], \"tools\": []}')", ()),
        ]
    )
    migrated = BrainRepository(ex)
    migrated_tools = {t.id for t in migrated.list_tools()}
    migrated_models = {m.id for m in migrated.list_models()}
    assert "searxng_public" in migrated_tools
    assert "gemini_google_maps" in migrated_tools
    assert "gemini-4-hydrogen" in migrated_models
    assert "gemini-3.6-flash" not in migrated_models
    assert migrated.get_meta("seeded_catalog_rev") == "4"
    deep_m = migrated.get_model("deep-research-preview-04-2026")
    assert deep_m is not None and deep_m.is_free is False
    migrated.close()


def test_discover_recovers_quota_429_and_paid_models_without_readding_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "billing_upgrade.db"
    repo = create_brain_repository(sqlite_path=db_file)
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        SimpleNamespace(name="models/gemini-3.5-flash-lite", display_name="Gemini 3.5 Flash-Lite", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-4-hydrogen", display_name="Gemini 4 Hydrogen", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-4-argon", display_name="Gemini 4 Argon", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-4-carbon", display_name="Gemini 4 Carbon", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-2.5-pro", display_name="Gemini 2.5 Pro", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.6-flash", display_name="Gemini 3.6 Flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.1-flash-lite", display_name="Gemini 3.1 Flash-Lite", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-4-dead", display_name="Gemini 4 Dead", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-2.5-flash-preview", display_name="Gemini 2.5 Flash Preview", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/deep-research-max-preview-04-2026", display_name="Deep Research Max Preview (Apr-21-2026)", supported_actions=[]),
        SimpleNamespace(name="models/deep-research-pro-preview-12-2025", display_name="Deep Research Pro Preview (Dec-12-2025) [Rerouted]", supported_actions=[]),
        SimpleNamespace(name="models/deep-research-preview-04-2026", display_name="Gemini Deep Research Agent", supported_actions=[]),
    ]

    def _fake_create(**kwargs: object) -> MagicMock:
        mid = str(kwargs.get("model") or "")
        if mid == "gemini-4-dead":
            raise RuntimeError("404 NOT_FOUND model unavailable")
        resp = MagicMock()
        resp.status = "completed"
        resp.output_text = "OK"
        resp.steps = []
        return resp

    mock_client.interactions.create.side_effect = _fake_create
    monkeypatch.setattr(
        "ki_brain.discovery.safe_http_request",
        lambda *_a, **_kw: (404, "https://searx.space", "text/plain", b""),
    )

    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets(gemini_api_key="AIzaSyPaidKey999"), genai_client=mock_client)

    # Ausgangszustand analog Nutzer-Screenshot: bestehende Quota-429-Modelle, Deep-Research-Previews & Excluded-IDs
    for mid in ("gemini-3.5-flash-lite", "gemini-4-hydrogen"):
        m_spec = brain.get_model(mid)
        assert m_spec is not None
        brain.add_or_update_model(
            m_spec.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED, "last_error": "429 RESOURCE_EXHAUSTED"})
        )
    gg_tool = brain.get_tool("gemini_google_search")
    assert gg_tool is not None
    brain.add_or_update_tool(
        gg_tool.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED, "last_error": "429 QUOTA"})
    )
    brain.add_or_update_model(
        AIModelSpec(
            id="deep-research-max-preview-04-2026",
            name="Deep Research Max Preview (Apr-21-2026)",
            provider=ModelProvider.GEMINI_DEEP_AGENT,
            model_id="deep-research-max-preview-04-2026",
            tier_hint=BrainTier.DEEP,
            is_deep_agent=True,
        )
    )
    brain.add_or_update_model(
        AIModelSpec(
            id="deep-research-pro-preview-12-2025",
            name="Deep Research Pro Preview (Dec-12-2025) [Rerouted]",
            provider=ModelProvider.GEMINI_DEEP_AGENT,
            model_id="deep-research-pro-preview-12-2025",
            tier_hint=BrainTier.DEEP,
            is_deep_agent=True,
        )
    )
    repo.delete_model("gemini-3.6-flash")
    repo.delete_model("gemini-3.1-flash-lite")
    repo.add_excluded_ids(model_ids=["gemini-3.6-flash", "gemini-4-argon", "gemini-2.5-flash-preview"])
    repo.add_excluded_ids(model_ids=["gemini-3.1-flash-lite"], reason="manual")
    repo.add_excluded_ids(model_ids=["gemini-4-dead"], reason="error")
    repo.add_excluded_ids(model_ids=["gemini-2.5-pro-preview"], reason="preview")

    # 1. Discover erkennt Quota-Recovery auf bestehenden Modellen, re-prüft Altlasten/Quota-IDs & holt Paid-Modelle
    summary = brain.auto_sync()
    assert summary.rechecked_models == 2
    assert set(summary.recovered_models) == {"gemini-3.5-flash-lite", "gemini-4-hydrogen"}
    assert brain.get_model("gemini-3.5-flash-lite").last_status == ProbeStatus.OK  # type: ignore[union-attr]
    assert brain.get_model("gemini-4-hydrogen").last_status == ProbeStatus.OK  # type: ignore[union-attr]
    assert brain.get_tool("gemini_google_search").last_status == ProbeStatus.OK  # type: ignore[union-attr]

    assert {"gemini-4-argon", "gemini-4-carbon", "gemini-2.5-pro", "gemini-3.6-flash"} <= set(summary.added_models)
    assert brain.get_model("gemini-4-argon").tier_hint == BrainTier.DEEP  # type: ignore[union-attr]
    assert brain.get_model("gemini-4-argon").is_free is False  # type: ignore[union-attr]
    assert brain.get_model("gemini-2.5-pro").is_free is False  # type: ignore[union-attr]
    deep_after_sync = brain.get_sequence(BrainTier.DEEP)
    assert "gemini-4-argon" not in deep_after_sync.model_sequence
    assert "gemini-4-argon" in deep_after_sync.paid_model_sequence
    assert "gemini-3.6-flash" not in repo.list_excluded_ids()[0]
    assert "gemini-4-argon" not in repo.list_excluded_ids()[0]

    # Manual/Error/Preview-Ausschlüsse bleiben unangetastet, vorhandene Previews werden gezählt
    assert brain.get_model("gemini-3.1-flash-lite") is None
    assert brain.get_model("gemini-4-dead") is None
    assert brain.get_model("gemini-2.5-flash-preview") is None
    assert "gemini-3.1-flash-lite" in summary.skipped_excluded
    assert "gemini-4-dead" in summary.skipped_excluded
    assert "gemini-2.5-flash-preview" in summary.skipped_previews
    assert summary.existing_previews_count == 2
    assert "gemini-3.5-flash-lite" in brain.get_sequence(BrainTier.FAST).model_sequence
    brain.close()


def test_secret_rotation_resets_quota_429_and_reports_vault_override(tmp_path: Path) -> None:
    db_file = tmp_path / "secret_rotation.db"
    brain = KIWebsearchBrain.for_dashboard(
        create_brain_repository(sqlite_path=db_file),
        BrainSecrets(gemini_api_key="AIzaSyEnvFreeKey"),
    )

    flash = brain.get_model("gemini-3.5-flash-lite")
    hydro = brain.get_model("gemini-4-hydrogen")
    gg_tool = brain.get_tool("gemini_google_search")
    bb_tool = brain.get_tool("browserbase_fetch")
    assert flash and hydro and gg_tool and bb_tool

    brain.add_or_update_model(flash.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED, "last_error": "429 Free"}))
    brain.add_or_update_model(hydro.model_copy(update={"last_status": ProbeStatus.ERROR, "last_error": "500 Err"}))
    brain.add_or_update_tool(gg_tool.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED, "last_error": "429 Tool"}))
    brain.add_or_update_tool(bb_tool.model_copy(update={"last_status": ProbeStatus.AUTH_MISSING, "last_error": "Missing"}))

    init_statuses = {s.env_name: s for s in brain.list_secret_statuses()}
    assert init_statuses["GEMINI_API_KEY"].source == "ENV / Streamlit Secrets"

    # Vault-Überschreibung setzt Quota 429 & Error auf UNTESTED zurück und weist auf Überschreibung hin
    brain.set_secret("GEMINI_API_KEY", "AIzaSyPaidVaultKey")
    vault_statuses = {s.env_name: s for s in brain.list_secret_statuses()}
    assert vault_statuses["GEMINI_API_KEY"].source == "Brain Vault (verschlüsselt, überschreibt ENV/Secrets)"
    assert brain.get_model("gemini-3.5-flash-lite").last_status == ProbeStatus.UNTESTED  # type: ignore[union-attr]
    assert brain.get_model("gemini-4-hydrogen").last_status == ProbeStatus.UNTESTED  # type: ignore[union-attr]
    assert brain.get_tool("gemini_google_search").last_status == ProbeStatus.UNTESTED  # type: ignore[union-attr]
    assert brain.get_tool("browserbase_fetch").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]

    # Löschen des Vault-Keys mit ENV-Fallback setzt Status erneut zurück; ohne Fallback -> AUTH_MISSING
    brain.add_or_update_model(flash.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED, "last_error": "429 Old"}))
    assert brain.delete_secret("GEMINI_API_KEY") is True
    after_del = {s.env_name: s for s in brain.list_secret_statuses()}
    assert after_del["GEMINI_API_KEY"].source == "ENV / Streamlit Secrets"
    assert brain.get_model("gemini-3.5-flash-lite").last_status == ProbeStatus.UNTESTED  # type: ignore[union-attr]

    brain.set_secret("BROWSERBASE_API_KEY", "bb_temp_vault_key")
    brain.add_or_update_tool(bb_tool.model_copy(update={"last_status": ProbeStatus.OK, "last_error": ""}))
    assert brain.delete_secret("BROWSERBASE_API_KEY") is True
    assert brain.get_tool("browserbase_fetch").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    brain.close()


def test_yahoo_session_cookies_dual_dom_fallback_and_ssrf_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from ki_brain.tools import yahoo_client
    from ki_brain.tools.http_utils import consume_probe_trace, safe_http_request, start_probe_trace
    from ki_brain.tools.search_web import search_yahoo

    with yahoo_client._YBV_LOCK:  # noqa: SLF001
        yahoo_client._YBV_CACHE.clear()  # noqa: SLF001

    de_html = (
        b'<div class="algo"><h3 class="title">'
        b'<a aria-label="Clean DE Title" href="https://r.search.yahoo.com/RU=https%3a%2f%2fpython.org%2f/RK=2">'
        b"<span>https://python.org \xe2\x80\xba docs</span>Clean DE Title</a></h3>"
        b'<div class="compText"><p>DE Snippet</p></div></div>'
    )
    us_html = (
        b'<div class="algo"><a href="https://example.org/us-hit">'
        b"<div>https://example.org</div>"
        b'<h3 class="title"><span>US Layout Title</span></h3></a>'
        b'<p class="fc-falcon">US Snippet</p></div>'
    )

    seen_cookie_jars: list[dict[str, str]] = []

    class _FakeResp:
        def __init__(self, status: int, headers: dict[str, str], body: bytes = b"") -> None:
            self.status_code = status
            self.headers = headers
            self._body = body

        def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
            return [self._body] if self._body else []

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

    def _fake_session_request(self: requests.Session, method: str, url: str, **_kw: object) -> _FakeResp:
        seen_cookie_jars.append(dict(self.cookies.get_dict()))
        if "de.search.yahoo.com/search" in url and self.cookies.get("YBV") != "v0.2-verified":
            self.cookies.set("YBV", "v0.1-challenge", domain="de.search.yahoo.com", path="/")
            return _FakeResp(307, {"Location": "/_bv/v.gif?orig=%2Fsearch%3Fp%3DPython&_bvc=1"})
        if "de.search.yahoo.com/_bv/v.gif" in url:
            assert self.cookies.get("YBV") == "v0.1-challenge"
            self.cookies.set("YBV", "v0.2-verified", domain="de.search.yahoo.com", path="/")
            return _FakeResp(307, {"Location": "/search?p=Python"})
        if "de.search.yahoo.com/search" in url and self.cookies.get("YBV") == "v0.2-verified":
            return _FakeResp(200, {"Content-Type": "text/html"}, de_html)
        return _FakeResp(404, {})

    monkeypatch.setattr("ki_brain.tools.http_utils.validate_safe_external_url", lambda u, dns_resolver=None: u)
    monkeypatch.setattr(requests.Session, "request", _fake_session_request)

    collected: dict[str, str] = {}
    st, final_url, _, body = safe_http_request(
        "GET",
        "https://de.search.yahoo.com/search",
        params={"p": "Python"},
        cookies={"sB": "v=1"},
        collected_cookies=collected,
    )
    assert st == 200
    assert final_url == "https://de.search.yahoo.com/search?p=Python"
    assert collected["YBV"] == "v0.2-verified"
    assert seen_cookie_jars[0].get("sB") == "v=1"
    assert seen_cookie_jars[1].get("YBV") == "v0.1-challenge"
    assert seen_cookie_jars[2].get("YBV") == "v0.2-verified"
    de_hits = yahoo_client.parse_yahoo_html(body)
    assert len(de_hits) == 1 and de_hits[0].title == "Clean DE Title" and de_hits[0].url == "https://python.org/"

    # US layout (`a:has(h3)`) + fallback rotation when DE edge returns HTTP 500
    called_eps: list[str] = []

    def _fake_yahoo_fallback(_m: str, ep: str, **_kw: object) -> tuple[int, str, str, bytes]:
        called_eps.append(ep)
        if "de.search.yahoo.com" in ep:
            return 500, ep, "text/html", b""
        return 200, ep, "text/html", us_html

    monkeypatch.setattr("ki_brain.tools.search_web.safe_http_request", _fake_yahoo_fallback)
    hits_fb = search_yahoo("Python AI")
    assert called_eps == ["https://de.search.yahoo.com/search", "https://search.yahoo.com/search"]
    assert len(hits_fb) == 1 and hits_fb[0].title == "US Layout Title" and hits_fb[0].url == "https://example.org/us-hit"

    # Rate limit 429 stops immediately without probing fallbacks
    called_429: list[str] = []

    def _fake_yahoo_429(_m: str, ep: str, **_kw: object) -> tuple[int, str, str, bytes]:
        called_429.append(ep)
        return 429, ep, "text/plain", b"Too Many Requests"

    monkeypatch.setattr("ki_brain.tools.search_web.safe_http_request", _fake_yahoo_429)
    start_probe_trace()
    assert search_yahoo("Python AI") == []
    consume_probe_trace()
    assert called_429 == ["https://de.search.yahoo.com/search"]

    # SSRF error is never swallowed by Yahoo fallback loop
    monkeypatch.setattr(
        "ki_brain.tools.search_web.safe_http_request",
        lambda *_a, **_kw: (_ for _ in ()).throw(BrainSSRFError("SSRF blocked redirect")),
    )
    with pytest.raises(BrainSSRFError):
        search_yahoo("Python AI")


def test_cost_aware_model_pricing_and_tier_sequencing() -> None:
    from ki_brain import compute_cost_score, estimate_blended_cost_per_1m
    from ki_brain.optimizer import compute_model_quality, optimize_tier_sequences

    free_gemma = AIModelSpec(
        id="gemma-3-27b-it",
        name="Gemma 3 27B",
        model_id="gemma-3-27b-it",
        tier_hint=BrainTier.STANDARD,
        last_status=ProbeStatus.OK,
        last_latency_ms=480,
    )
    tiny_8b = AIModelSpec(
        id="groq/llama-3.1-8b-instant",
        name="Groq Llama 3.1 8B",
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model_id="llama-3.1-8b-instant",
        tier_hint=BrainTier.FAST,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        last_status=ProbeStatus.OK,
        last_latency_ms=350,
    )
    flash_25_lite = AIModelSpec(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash-Lite",
        model_id="gemini-2.5-flash-lite",
        tier_hint=BrainTier.FAST,
        last_status=ProbeStatus.OK,
        last_latency_ms=420,
    )
    llama_70b = AIModelSpec(
        id="groq/llama-3.3-70b-versatile",
        name="Groq Llama 3.3 70B",
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model_id="llama-3.3-70b-versatile",
        tier_hint=BrainTier.STANDARD,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        last_status=ProbeStatus.OK,
        last_latency_ms=450,
    )
    flash_25 = AIModelSpec(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        model_id="gemini-2.5-flash",
        tier_hint=BrainTier.STANDARD,
        last_status=ProbeStatus.OK,
        last_latency_ms=480,
    )
    flash_38 = AIModelSpec(
        id="gemini-3.8-flash",
        name="Gemini 3.8 Flash",
        model_id="gemini-3.8-flash",
        tier_hint=BrainTier.STANDARD,
        last_status=ProbeStatus.OK,
        last_latency_ms=480,
    )
    hydrogen_4 = AIModelSpec(
        id="gemini-4-hydrogen",
        name="Gemini 4 Hydrogen",
        model_id="gemini-4-hydrogen",
        tier_hint=BrainTier.STANDARD,
        last_status=ProbeStatus.OK,
        last_latency_ms=480,
    )
    argon_4 = AIModelSpec(
        id="gemini-4-argon",
        name="Gemini 4 Argon",
        model_id="gemini-4-argon",
        tier_hint=BrainTier.DEEP,
        is_free=False,
        free_quota_period="none",
        last_status=ProbeStatus.OK,
        last_latency_ms=480,
    )
    broken_free = AIModelSpec(
        id="or/broken/model-free",
        name="Broken Free",
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model_id="broken/model:free",
        tier_hint=BrainTier.STANDARD,
        last_status=ProbeStatus.QUOTA_LIMITED,
        last_latency_ms=150,
    )

    assert estimate_blended_cost_per_1m(free_gemma) == 0.0
    assert compute_cost_score(free_gemma) == 100.0
    assert estimate_blended_cost_per_1m(flash_25_lite) == 0.18
    assert estimate_blended_cost_per_1m(flash_25) == 0.85
    assert estimate_blended_cost_per_1m(hydrogen_4) == 3.0
    assert estimate_blended_cost_per_1m(argon_4) == 4.0

    raw_fleet = [argon_4, hydrogen_4, flash_38, flash_25, flash_25_lite, llama_70b, tiny_8b, broken_free]
    scored_fleet = [
        m.model_copy(update={"quality_score": compute_model_quality(m, m.last_status, m.last_latency_ms)})
        for m in raw_fleet
    ]

    seqs = optimize_tier_sequences(scored_fleet, [], {})
    fast_seq = seqs[BrainTier.FAST].model_sequence
    std_seq = seqs[BrainTier.STANDARD].model_sequence
    deep_seq = seqs[BrainTier.DEEP].model_sequence
    paid_deep_seq = seqs[BrainTier.DEEP].paid_model_sequence

    assert fast_seq[0] == "gemini-2.5-flash-lite"
    assert fast_seq.index("gemini-2.5-flash-lite") < fast_seq.index("gemini-4-hydrogen")

    # Standard: günstige & starke Standard-Modelle (70B / Flash) vor teurem Hydrogen & schwachem 8B
    assert std_seq[0] in {"groq/llama-3.3-70b-versatile", "gemini-2.5-flash"}
    assert std_seq.index("gemini-2.5-flash") < std_seq.index("gemini-4-hydrogen")
    assert std_seq.index("gemini-3.8-flash") < std_seq.index("gemini-4-hydrogen")
    assert std_seq.index("gemini-3.8-flash") < std_seq.index("groq/llama-3.1-8b-instant")

    # Deep 0%-Standard schließt Paid-Only Argon aus; Paid-Opt-In platziert Argon vor 8B
    assert "gemini-4-argon" not in deep_seq
    assert paid_deep_seq.index("gemini-4-argon") < paid_deep_seq.index("groq/llama-3.1-8b-instant")
    assert "or/broken/model-free" not in std_seq


def test_google_maps_grounding_citations_geo_smart_and_gemini3_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from ki_brain.llm_runner import (
        extract_citations_from_interaction,
        extract_grounding_tool_counts,
    )
    from ki_brain.models import BrainCitation
    from ki_brain.quotas import is_gemini_3_or_newer, is_geo_or_maps_query, select_grounding_model

    # 1. Citation & Grounding-Query-Count-Extraktion inkl. google_maps_result & place_citation
    fake_interaction = SimpleNamespace(
        usage=SimpleNamespace(
            grounding_tool_count=[
                SimpleNamespace(type="google_maps", count=2),
                SimpleNamespace(type="google_search", count=1),
            ]
        ),
        steps=[
            SimpleNamespace(
                type="google_maps_result",
                result=[
                    SimpleNamespace(
                        places=[
                            SimpleNamespace(
                                name="Zeit für Brot Berlin",
                                place_id="places/ChIJTestAlex1",
                                url="",
                                review_snippets=[SimpleNamespace(title="Tolle Zimtschnecken")],
                            )
                        ]
                    )
                ],
            ),
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(
                        type="text",
                        annotations=[
                            SimpleNamespace(
                                type="place_citation",
                                name="Cafe Einstein",
                                url="https://maps.google.com/?cid=4242",
                                place_id="",
                                review_snippets=[SimpleNamespace(title="Klassiker")],
                            ),
                            SimpleNamespace(
                                type="url_citation",
                                url="https://example.org/cafe-guide",
                                title="Café Guide",
                            ),
                        ],
                    )
                ],
            ),
        ],
    )
    cites = extract_citations_from_interaction(fake_interaction)
    assert len(cites) == 3
    assert cites[0].source_tool == "gemini_google_maps"
    assert "place_id:ChIJTestAlex1" in cites[0].url
    assert "Zimtschnecken" in cites[0].snippet
    assert cites[1].title == "Cafe Einstein" and cites[1].source_tool == "gemini_google_maps"
    assert cites[2].source_tool == "gemini_google_search"
    assert extract_grounding_tool_counts(fake_interaction) == {"google_maps": 2, "google_search": 1}

    # 2. Gemini-3+-Guard für kostenfreies Maps Grounding & Geo-Smart-Erkennung
    m25 = AIModelSpec(id="gemini-2.5-flash", name="G2.5", model_id="gemini-2.5-flash")
    m35 = AIModelSpec(id="gemini-3.5-flash-lite", name="G3.5 Lite", model_id="gemini-3.5-flash-lite")
    gemma3 = AIModelSpec(id="gemma-3-27b-it", name="Gemma 3", model_id="gemma-3-27b-it")
    assert is_gemini_3_or_newer(m25) is False
    assert is_gemini_3_or_newer(gemma3) is False
    assert is_gemini_3_or_newer(m35) is True
    assert is_geo_or_maps_query("Cafés am Alexanderplatz in Berlin") is True
    assert is_geo_or_maps_query("Python 3.13 asyncio Änderungen") is False

    secrets = BrainSecrets(gemini_api_key="AIzaSyTestKey123")
    assert select_grounding_model([m25], secrets, executor_type="gemini_google_maps", allow_paid=False) is None
    assert select_grounding_model([m25], secrets, executor_type="gemini_google_maps", allow_paid=True) == m25
    assert select_grounding_model([m25, m35], secrets, executor_type="gemini_google_maps", allow_paid=False) == m35

    # 3. Geo-Smart-Routing + Kontingent-Abbuchung über usage.grounding_tool_count
    repo = create_brain_repository(sqlite_path=":memory:")
    seen_create_calls: list[dict[str, Any]] = []

    def _fake_maps_create(**kwargs: Any) -> SimpleNamespace:
        seen_create_calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output_text="Empfehlung am Alexanderplatz: Zeit für Brot.",
            usage=SimpleNamespace(grounding_tool_count=[SimpleNamespace(type="google_maps", count=2)]),
            steps=fake_interaction.steps,
        )

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = _fake_maps_create
    brain = KIWebsearchBrain.for_dashboard(repo, secrets, genai_client=mock_client)

    def _fake_non_ai_search(tool: ToolSpec, q: str, **_kw: object) -> list[BrainCitation]:
        if tool.executor_type in {"gemini_google_search", "gemini_google_maps"}:
            from ki_brain.tools.registry import execute_search_tool as real_exec

            return real_exec(tool, q, **_kw)  # type: ignore[arg-type]
        return [BrainCitation(url=f"https://example.org/{tool.id}", title=tool.name, snippet="OK", source_tool=tool.id)]

    monkeypatch.setattr("ki_brain.execution.execute_search_tool", _fake_non_ai_search)

    seq_std = brain.get_sequence(BrainTier.STANDARD)
    brain.update_sequence(
        seq_std.model_copy(
            update={
                "search_tool_sequence": ["yahoo_search", "gemini_google_maps", "searxng_public"],
                "max_search_tools": 3,
                "manual_override": True,
            }
        )
    )

    non_geo = brain.search("Python 3.13 asyncio Neuerungen", tier=BrainTier.STANDARD)
    assert "gemini_google_maps" not in non_geo.tools_executed
    assert seen_create_calls == []

    geo = brain.search("Cafés am Alexanderplatz in Berlin", tier=BrainTier.STANDARD)
    assert "gemini_google_maps" in geo.tools_executed
    assert seen_create_calls[-1]["tools"] == [{"type": "google_maps"}]
    quotas = brain.load_usage_quotas()
    assert quotas["tools"]["gemini_google_maps"]["used"] == 2

    forced = brain.search(
        "Python Konferenz",
        tier=BrainTier.STANDARD,
        use_maps=True,
        latitude=52.5219,
        longitude=13.4132,
    )
    assert "gemini_google_maps" in forced.tools_executed
    assert seen_create_calls[-1]["tools"] == [{"type": "google_maps", "latitude": 52.5219, "longitude": 13.4132}]

    suppressed = brain.search("Cafés am Alexanderplatz in Berlin", tier=BrainTier.STANDARD, use_maps=False)
    assert "gemini_google_maps" not in suppressed.tools_executed
    brain.close()


def test_strict_zero_cost_enforcement_quota_exhaustion_and_allow_paid_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime

    from ki_brain import evaluate_tool_quota
    from ki_brain.models import BrainCitation
    from ki_brain.quotas import record_quota_usage

    repo = create_brain_repository(sqlite_path=":memory:")
    mock_client = MagicMock()
    called_models: list[str] = []

    def _fake_create(**kwargs: Any) -> SimpleNamespace:
        target = str(kwargs.get("model") or kwargs.get("agent") or "")
        called_models.append(target)
        if kwargs.get("agent"):
            return SimpleNamespace(id="bg_run_1", status="completed", output_text="Deep Research Paid Ergebnis", steps=[])
        if kwargs.get("response_format"):
            return SimpleNamespace(status="completed", output_text='{"topic": "Budget", "score": 100}', steps=[])
        return SimpleNamespace(status="completed", output_text=f"Antwort von {target}", steps=[])

    mock_client.interactions.create.side_effect = _fake_create
    mock_client.interactions.get.return_value = SimpleNamespace(
        id="bg_run_1", status="completed", output_text="Deep Research Paid Ergebnis", steps=[]
    )
    mock_client.models.list.return_value = [SimpleNamespace(name="models/gemini-3.5-flash-lite")]

    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets(gemini_api_key="AIzaSyTestKey123"), genai_client=mock_client)
    paid_model = AIModelSpec(
        id="gemini-4-argon",
        name="Gemini 4 Argon",
        model_id="gemini-4-argon",
        tier_hint=BrainTier.DEEP,
        is_free=False,
        free_quota_period="none",
        last_status=ProbeStatus.OK,
    )
    paid_tool = ToolSpec(
        id="paid_premium_api",
        name="Paid Premium Search",
        category=ToolCategory.WEB_SEARCH,
        executor_type="custom_json_api",
        endpoint_url="https://api.example.com/paid",
        is_free=False,
        free_quota_period="none",
        paid_cost_per_1k=15.0,
        last_status=ProbeStatus.OK,
    )
    brain.add_or_update_model(paid_model)
    brain.add_or_update_tool(paid_tool)

    # Manuell erzwungene Deep-Sequenz mit Paid-Only vorne + Fallback auf Free-Modell
    deep_seq = brain.get_sequence(BrainTier.DEEP)
    brain.update_sequence(
        deep_seq.model_copy(
            update={
                "search_tool_sequence": ["paid_premium_api", "gemini_google_search", "wikipedia_api"],
                "paid_search_tool_sequence": ["paid_premium_api", "gemini_google_search", "wikipedia_api"],
                "model_sequence": ["gemini-4-argon", "deep-research-preview-04-2026", "gemini-2.5-flash"],
                "paid_model_sequence": ["gemini-4-argon", "deep-research-preview-04-2026", "gemini-2.5-flash"],
                "enable_page_scraping": False,
                "manual_override": True,
            }
        )
    )

    executed_tool_ids: list[str] = []

    def _fake_exec(tool: ToolSpec, *_a: object, **_kw: object) -> list[BrainCitation]:
        executed_tool_ids.append(tool.id)
        return [BrainCitation(url=f"https://example.org/{tool.id}", title=tool.id, snippet="Treffer", source_tool=tool.id)]

    monkeypatch.setattr("ki_brain.execution.execute_search_tool", _fake_exec)
    monkeypatch.setattr("ki_brain.benchmark.execute_search_tool", _fake_exec)
    monkeypatch.setattr(
        "ki_brain.benchmark.execute_extractor_tool",
        lambda tool, url, **_kw: SimpleNamespace(text_content="Volltext Inhalt " * 20),
    )

    # 1. Strikter 0%-Modus: Paid Tool, Argon & Deep Research Agent bleiben strikt außen vor
    res_search_free = brain.search("KI Sicherheit", tier=BrainTier.DEEP, allow_paid=False)
    assert "paid_premium_api" not in res_search_free.tools_executed
    assert "wikipedia_api" in res_search_free.tools_executed

    res_text_free = brain.generate_text("Hallo", tier=BrainTier.DEEP, allow_paid=False)
    assert res_text_free.model_used == "gemini-2.5-flash"

    parsed, used_struct = brain.generate_structured("Bewerte", _SampleSchema, tier=BrainTier.DEEP, allow_paid=False)
    assert parsed.score == 100 and used_struct == "gemini-2.5-flash"

    res_ask_free = brain.ask("KI Trends", tier=BrainTier.DEEP, allow_paid=False)
    assert res_ask_free.model_used == "gemini-2.5-flash"
    assert "gemini-4-argon" not in called_models
    assert "deep-research-preview-04-2026" not in called_models

    # 2. Explizites Opt-In (allow_paid=True) bezieht Paid-Tool und Paid-Modelle ein
    res_ask_paid = brain.ask("KI Trends", tier=BrainTier.DEEP, allow_paid=True)
    assert "paid_premium_api" in res_ask_paid.tools_executed
    assert res_ask_paid.model_used == "gemini-4-argon"
    assert "gemini-4-argon" in called_models

    # 3. Free-Kontingent erschöpfen (5.000/5.000) -> 0%-Sperre bis Periodenwechsel oder Reset
    record_quota_usage(repo, tool_counts={"gemini_google_search": 5000})
    gg_spec = brain.get_tool("gemini_google_search")
    assert gg_spec is not None
    cur_snap = evaluate_tool_quota(gg_spec, brain.load_usage_quotas())
    assert cur_snap.exhausted is True and cur_snap.remaining == 0 and cur_snap.is_free_available is False

    future_month = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=35)
    rollover_snap = evaluate_tool_quota(gg_spec, brain.load_usage_quotas(), now=future_month)
    assert rollover_snap.exhausted is False and rollover_snap.remaining == 5000 and rollover_snap.is_free_available is True

    executed_tool_ids.clear()
    brain.search("Python Release", tier=BrainTier.DEEP, allow_paid=False)
    assert "gemini_google_search" not in executed_tool_ids

    brain.search("Python Release", tier=BrainTier.DEEP, allow_paid=True)
    assert "gemini_google_search" in executed_tool_ids

    # 4. Benchmark im 0%-Modus schützt vor kostenpflichtigen Probes, ohne Status zu zerstören (kein Löschen bei Prune Failing)
    called_models.clear()
    bench_report = brain.run_benchmark(force_overwrite_sequences=True, include_paid=False)
    assert "gemini-4-argon" not in called_models
    argon_after = brain.get_model("gemini-4-argon")
    gg_after = brain.get_tool("gemini_google_search")
    assert argon_after is not None and argon_after.last_status == ProbeStatus.OK
    assert gg_after is not None and gg_after.last_status != ProbeStatus.QUOTA_LIMITED
    deep_opt = bench_report.sequences[BrainTier.DEEP.value]
    assert "gemini_google_search" not in deep_opt.search_tool_sequence
    assert "gemini_google_search" in deep_opt.paid_search_tool_sequence
    assert "gemini-4-argon" not in deep_opt.model_sequence
    assert "gemini-4-argon" in deep_opt.paid_model_sequence

    rem_m, rem_t = brain.cleanup_unhealthy_items()
    assert "gemini-4-argon" not in rem_m and "gemini_google_search" not in rem_t

    # 5. Kontingent-Reset gibt Tool wieder für 0%-Sequenz frei
    brain.reset_usage_quotas()
    reset_snap = evaluate_tool_quota(gg_spec, brain.load_usage_quotas())
    assert reset_snap.remaining == 5000 and reset_snap.is_free_available is True
    brain.close()


def test_legacy_rev2_to_rev3_upgrade_preserves_customizations(tmp_path: Path) -> None:
    from ki_brain.catalog import build_default_models, build_default_tools
    from ki_brain.db_executors import SqliteBrainExecutor
    from ki_brain.models import TierSequenceConfig
    from ki_brain.storage import BrainRepository

    rev2_db = tmp_path / "legacy_rev2.db"
    ex = SqliteBrainExecutor(rev2_db)
    seed_repo = BrainRepository(ex, seed_defaults=False)
    seed_repo._init_schema()  # noqa: SLF001

    old_models = [
        m.model_copy(update={"is_free": True, "free_quota_period": "unlimited"}) if m.id == "deep-research-preview-04-2026" else m
        for m in build_default_models()
    ]
    old_tools = [
        t.model_copy(update={"free_quota_limit": 0, "free_quota_period": "unlimited"})
        for t in build_default_tools()
        if t.id != "gemini_google_maps"
    ]
    for m in old_models:
        seed_repo.upsert_model(m)
    for t in old_tools:
        seed_repo.upsert_tool(t)

    seed_repo.upsert_sequence(
        TierSequenceConfig(
            tier=BrainTier.FAST,
            search_tool_sequence=["wikipedia_api"],
            model_sequence=["gemini-3.5-flash-lite"],
            manual_override=True,
        )
    )
    seed_repo.upsert_sequence(
        TierSequenceConfig(
            tier=BrainTier.DEEP,
            search_tool_sequence=["yahoo_search", "gemini_google_search"],
            model_sequence=["deep-research-preview-04-2026", "gemini-2.5-flash"],
            use_deep_agent=True,
            manual_override=False,
        )
    )
    seed_repo.set_meta("seeded_v1", "1")
    seed_repo.set_meta("seeded_catalog_rev", "2")
    import json

    seed_repo.set_meta(
        "known_builtin_ids",
        json.dumps({"models": [m.id for m in old_models], "tools": [t.id for t in old_tools]}),
    )
    seed_repo.close()

    upgraded = BrainRepository(SqliteBrainExecutor(rev2_db))
    assert upgraded.get_meta("seeded_catalog_rev") == "4"
    assert upgraded.get_tool("gemini_google_maps") is not None
    assert upgraded.get_tool("gemini_google_search").free_quota_limit == 5000  # type: ignore[union-attr]
    assert upgraded.get_model("deep-research-preview-04-2026").is_free is False  # type: ignore[union-attr]

    fast_seq = upgraded.get_sequence(BrainTier.FAST)
    assert fast_seq.manual_override is True and fast_seq.search_tool_sequence == ["wikipedia_api"]

    deep_seq = upgraded.get_sequence(BrainTier.DEEP)
    assert deep_seq.use_deep_agent is False
    assert deep_seq.paid_use_deep_agent is True
    assert "deep-research-preview-04-2026" not in deep_seq.model_sequence
    assert "deep-research-preview-04-2026" in deep_seq.paid_model_sequence
    upgraded.close()


def test_strict_three_app_access_control_and_admin_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ki_brain import ALLOWED_CLIENT_APPS, BrainAuthError

    db_file = tmp_path / "access_guard.db"
    assert ALLOWED_CLIENT_APPS == ("news_bot", "research_bot", "alert_bot")

    # 1. Fremde oder unauthentifizierte Aufrufe schlagen fail-closed fehl
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain(create_brain_repository(sqlite_path=db_file))
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(sqlite_path=db_file)
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(app_id="stranger_app", app_token="kib_stranger_123", sqlite_path=db_file)
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(app_id="news_bot", app_token="kib_news_bot_fake", sqlite_path=db_file)

    # 2. Dashboard-Admin erzeugt Token exklusiv für die 3 Apps (Fremd-App wird auch beim Ausstellen geblockt)
    admin = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=db_file))
    with pytest.raises(BrainAuthError):
        admin.issue_app_token("evil_bot")

    res_token = admin.issue_app_token("research_bot")
    assert res_token.startswith("kib_research_bot_")
    clients = {c.app_id: c for c in admin.list_app_clients()}
    assert set(clients.keys()) == {"news_bot", "research_bot", "alert_bot"}
    assert clients["research_bot"].has_token is True and clients["research_bot"].enabled is True
    assert clients["news_bot"].has_token is False and clients["news_bot"].enabled is False
    # Klartext-Token wird niemals in der DB gespeichert
    raw_row = admin.repository.get_app_client_record("research_bot")
    assert raw_row is not None and res_token not in raw_row["token_hash"]

    # 3. Autorisierte App (research_bot, auch mit Bindestrich-Normalisierung) erhält Zugriff, aber keine Admin-Vault-Rechte
    monkeypatch.setenv("BRAIN_APP_ID", "research-bot")
    monkeypatch.setenv("BRAIN_APP_TOKEN", res_token)
    with KIWebsearchBrain.from_env(sqlite_path=db_file) as bot_brain:
        assert bot_brain.client_app_id == "research_bot"
        assert bot_brain.is_admin_client is False
        assert len(bot_brain.list_models()) >= 8
        with pytest.raises(BrainAuthError):
            bot_brain.set_secret("GEMINI_FREE_API_KEY", "unauthorized")
        with pytest.raises(BrainAuthError):
            bot_brain.issue_app_token("news_bot")

    # 4. Token gilt nicht für andere App & Sperren/Widerrufen blockiert sofort
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(app_id="news_bot", app_token=res_token, sqlite_path=db_file)

    assert admin.set_app_enabled("research_bot", False) is True
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(app_id="research_bot", app_token=res_token, sqlite_path=db_file)

    assert admin.set_app_enabled("research_bot", True) is True
    with KIWebsearchBrain.from_env(app_id="research_bot", app_token=res_token, sqlite_path=db_file) as bot_again:
        assert bot_again.client_app_id == "research_bot"

    assert admin.revoke_app_token("research_bot") is True
    with pytest.raises(BrainAuthError):
        KIWebsearchBrain.from_env(app_id="research_bot", app_token=res_token, sqlite_path=db_file)
    admin.close()


def test_free_and_paid_tier_keys_isolation_and_cross_app_vault_decryption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from ki_brain.llm_runner import UnifiedLLMRunner

    # 1. Cross-App Vault-Entschlüsselung: Dashboard hat APP_PASSWORD + BRAIN_TURSO_KEY, Bot hat nur BRAIN_TURSO_KEY
    db_file = tmp_path / "tier_keys_vault.db"
    monkeypatch.setenv("APP_PASSWORD", "dashboard_only_admin_pw")
    monkeypatch.setenv("BRAIN_TURSO_KEY", "shared_turso_secret_token_999")
    monkeypatch.delenv("BRAIN_SECRET_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_FREE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_PAID_API_KEY", raising=False)

    admin = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=db_file), BrainSecrets())
    admin.set_secret("GEMINI_FREE_API_KEY", "AIzaSyFreeTierKey111")
    admin.set_secret("GEMINI_PAID_API_KEY", "AIzaSyPaidTierKey222")
    bot_token = admin.issue_app_token("alert_bot")
    admin.close()

    # Bot-Umgebung ohne Dashboard-APP_PASSWORD entschlüsselt beide Keys problemlos aus dem Vault
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    with KIWebsearchBrain.from_env(app_id="alert_bot", app_token=bot_token, sqlite_path=db_file) as bot:
        free_model = bot.get_model("gemini-3.5-flash-lite")
        paid_model = bot.get_model("deep-research-preview-04-2026")
        assert free_model is not None and paid_model is not None
        assert bot._secrets.resolve_model_key(free_model, allow_paid=False) == "AIzaSyFreeTierKey111"  # noqa: SLF001
        assert bot._secrets.resolve_model_key(free_model, allow_paid=True) == "AIzaSyPaidTierKey222"  # noqa: SLF001
        assert bot._secrets.resolve_model_key(paid_model, allow_paid=False) == ""  # noqa: SLF001
        assert bot._secrets.resolve_model_key(paid_model, allow_paid=True) == "AIzaSyPaidTierKey222"  # noqa: SLF001

    # 2. Wenn NUR Paid-Tier-Key existiert, darf allow_paid=False ihn NIEMALS verwenden (0%-Schutz!)
    paid_only_secrets = BrainSecrets(gemini_paid_api_key="AIzaSyOnlyPaid999")
    free_spec = AIModelSpec(id="gemini-3.5-flash-lite", name="Flash Lite", model_id="gemini-3.5-flash-lite", paid_api_key_env="GEMINI_PAID_API_KEY")
    assert paid_only_secrets.resolve_model_key(free_spec, allow_paid=False) == ""
    assert paid_only_secrets.has_model_key(free_spec, allow_paid=False) is False
    assert paid_only_secrets.resolve_model_key(free_spec, allow_paid=True) == "AIzaSyOnlyPaid999"

    # 3. UnifiedLLMRunner isoliert genai.Client-Instanzen strikt nach aufgelöstem Free- vs Paid-Key
    created_keys: list[str] = []

    class _FakeGenAIClient:
        def __init__(self, api_key: str, **_kw: object) -> None:
            self.api_key = api_key
            created_keys.append(api_key)
            self.interactions = SimpleNamespace(
                create=lambda **kw: SimpleNamespace(output_text=f"via:{api_key}", steps=[], status="completed")
            )

    monkeypatch.setattr("ki_brain.llm_runner.genai.Client", _FakeGenAIClient)
    runner = UnifiedLLMRunner(
        BrainSecrets(gemini_free_api_key="AIzaSyFreeA", gemini_paid_api_key="AIzaSyPaidB")
    )
    txt_free, _ = runner.run_text(free_spec, "Hi", allow_paid=False)
    txt_paid, _ = runner.run_text(free_spec, "Hi", allow_paid=True)
    assert txt_free == "via:AIzaSyFreeA"
    assert txt_paid == "via:AIzaSyPaidB"
    assert created_keys == ["AIzaSyFreeA", "AIzaSyPaidB"]


def test_run_full_pipeline_executes_all_four_steps_with_progress_and_options(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from ki_brain import BenchmarkReport, DiscoverySummary, UnblockSummary

    brain = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=tmp_path / "pipe.db"), BrainSecrets())
    brain.auto_sync = MagicMock(  # type: ignore[method-assign]
        return_value=DiscoverySummary(added_models=["m1"], checked_sources=["builtins"])
    )
    brain.run_benchmark = MagicMock(  # type: ignore[method-assign]
        return_value=BenchmarkReport(
            run_id="pipe01", started_at="2026-11-19T10:00:00+00:00", finished_at="2026-11-19T10:00:02+00:00",
            test_query="Python", test_url="https://example.org", healthy_tools_count=8, healthy_models_count=4,
        )
    )
    brain.resolve_blocked_tools = MagicMock(  # type: ignore[method-assign]
        return_value=UnblockSummary(checked_blocked=["ddg"], unblocked_ids=["ddg"])
    )
    brain.cleanup_unhealthy_items_with_status = MagicMock(  # type: ignore[method-assign]
        return_value=([("bad-m", ProbeStatus.ERROR)], [])
    )

    progress_events: list[tuple[float, str]] = []
    summary = brain.run_full_pipeline(
        exclude_previews=True,
        include_paid=False,
        on_progress=lambda f, m: progress_events.append((f, m)),
    )
    assert summary.executed_steps == ("discover", "benchmark", "unblock", "prune")
    assert summary.discovery is not None and summary.discovery.added_models == ["m1"]
    assert summary.benchmark is not None and summary.benchmark.run_id == "pipe01"
    assert summary.unblock is not None and summary.unblock.unblocked_ids == ["ddg"]
    assert summary.pruned_models == (("bad-m", ProbeStatus.ERROR),)
    assert progress_events[-1][0] == 1.0
    brain.auto_sync.assert_called_once()
    assert brain.auto_sync.call_args.kwargs["exclude_previews"] is True
    assert brain.run_benchmark.call_args.kwargs["include_paid"] is False

    # Optionen erlauben gezieltes Abwählen einzelner Schritte
    partial = brain.run_full_pipeline(run_discover=False, run_benchmark=True, run_unblock=False, run_prune=False)
    assert partial.executed_steps == ("benchmark",)
    assert partial.discovery is None and partial.unblock is None
    brain.close()


def test_rev4_sequence_sanitization_and_prune_removes_stale_deleted_preview(tmp_path: Path) -> None:
    import json

    from ki_brain.catalog import build_default_models, build_default_tools
    from ki_brain.db_executors import SqliteBrainExecutor
    from ki_brain.models import TierSequenceConfig
    from ki_brain.storage import BrainRepository

    rev3_db = tmp_path / "stale_rev3.db"
    seed_repo = BrainRepository(SqliteBrainExecutor(rev3_db), seed_defaults=False)
    seed_repo._init_schema()  # noqa: SLF001

    # Simuliert Live-DB nach Löschung von deep-research-preview-04-2026: Modell weg, aber noch in Deep-Rationale & paid_use_deep_agent
    models = [m for m in build_default_models() if m.id != "deep-research-preview-04-2026"]
    tools = build_default_tools()
    seed_repo.upsert_models_batch(models)
    seed_repo.upsert_tools_batch(tools)
    seed_repo.upsert_sequence(
        TierSequenceConfig(
            tier=BrainTier.DEEP,
            search_tool_sequence=["yahoo_search", "wikipedia_api"],
            model_sequence=["gemini-3.5-flash-lite"],
            paid_model_sequence=["deep-research-preview-04-2026", "gemini-3.5-flash-lite"],
            use_deep_agent=False,
            paid_use_deep_agent=True,
            manual_override=False,
            optimization_rationale="Deep (0%-Standard): Suche yahoo_search | KI gemini-3.5-flash-lite | Paid-Opt-In: deep-research-preview-04-2026.",
        )
    )
    seed_repo.set_meta("seeded_v1", "1")
    seed_repo.set_meta("seeded_catalog_rev", "3")
    seed_repo.set_meta(
        "known_builtin_ids",
        json.dumps({"models": [m.id for m in build_default_models()], "tools": [t.id for t in tools]}),
    )
    seed_repo.set_meta("excluded_ids", json.dumps({"models": ["deep-research-preview-04-2026"], "tools": []}))
    seed_repo.close()

    sanitized = BrainRepository(SqliteBrainExecutor(rev3_db))
    assert sanitized.get_meta("seeded_catalog_rev") == "4"
    deep_seq = sanitized.get_sequence(BrainTier.DEEP)
    assert "deep-research-preview-04-2026" not in deep_seq.paid_model_sequence
    assert deep_seq.paid_use_deep_agent is False
    assert "deep-research-preview-04-2026" not in deep_seq.optimization_rationale
    sanitized.close()

    # Auch direktes remove_model() bereinigt deep_agent-Flags & Rationale sofort
    live_brain = KIWebsearchBrain.for_dashboard(create_brain_repository(sqlite_path=tmp_path / "live_prune.db"))
    assert live_brain.get_sequence(BrainTier.DEEP).paid_use_deep_agent is True
    assert live_brain.remove_model("deep-research-preview-04-2026") is True
    deep_after = live_brain.get_sequence(BrainTier.DEEP)
    assert "deep-research-preview-04-2026" not in deep_after.paid_model_sequence
    assert deep_after.paid_use_deep_agent is False
    assert "deep-research-preview-04-2026" not in deep_after.optimization_rationale
    live_brain.close()


def test_probe_model_minimal_thinking_fallback_and_quick_verify_single_model(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeGenAIClient:
        def __init__(self) -> None:
            self.interactions = SimpleNamespace(create=self._create)
            self.models = SimpleNamespace(list=lambda: [])

        def _create(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(dict(kwargs))
            gc = kwargs.get("generation_config") or {}
            # Erstaufruf lehnt thinking_level mit HTTP 400 ab -> Runner muss sauber ohne thinking_level wiederholen
            if len(calls) == 1 and "thinking_level" in gc:
                raise RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported for this model")
            return SimpleNamespace(status="completed", output_text="OK", steps=[])

    fake_client = _FakeGenAIClient()
    repo = create_brain_repository(sqlite_path=tmp_path / "quick_verify.db")
    brain = KIWebsearchBrain.for_dashboard(repo, BrainSecrets(), genai_client=fake_client)

    # Unkonfigurierter Legacy GEMINI_API_KEY wird im Vault ausgeblendet
    names_before = {s.env_name for s in brain.list_secret_statuses()}
    assert "GEMINI_API_KEY" not in names_before
    assert "GEMINI_FREE_API_KEY" in names_before

    brain.set_secret("GEMINI_FREE_API_KEY", "AIzaSyQuickVerifyKey123")
    entries = brain.probe_items_for_secret("GEMINI_FREE_API_KEY", quick_verify=True)

    # quick_verify testet nur 1 schnelles Referenzmodell (Flash-Lite) und 0 Grounding-Tools
    assert len(entries) == 1
    assert entries[0].item_type == "model"
    assert "lite" in entries[0].item_id
    assert entries[0].status == ProbeStatus.OK
    # 1. Versuch schickte thinking_level='minimal', 2. Fallback-Versuch ohne thinking_level war OK
    assert len(calls) == 2
    assert calls[0]["generation_config"] == {"max_output_tokens": 32, "thinking_level": "minimal"}
    assert "thinking_level" not in (calls[1].get("generation_config") or {})
    brain.close()
