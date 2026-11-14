"""Umfassende Tests für das KI & Websearch Brain Framework (`ki_brain`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
    brain = KIWebsearchBrain(repo, BrainSecrets())

    model_ids = {m.id for m in brain.list_models()}
    for expected in (
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
        "gemini-4-argon",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "deep-research-preview-04-2026",
    ):
        assert expected in model_ids

    tool_ids = {t.id for t in brain.list_tools()}
    for expected_tool in (
        "yahoo_search",
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
        "gemini_url_context",
        "open_meteo",
    ):
        assert expected_tool in tool_ids

    # Persistente Tombstone-Löschung + Reset
    assert brain.remove_tool("wikidata_api") is True
    brain.close()

    reopened_repo = create_brain_repository(sqlite_path=db_file)
    reopened = KIWebsearchBrain(reopened_repo, BrainSecrets())
    assert "wikidata_api" not in {t.id for t in reopened.list_tools()}
    reopened.reset_defaults()
    assert "wikidata_api" in {t.id for t in reopened.list_tools()}
    reopened.close()


def test_dynamic_crud_custom_tools_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = create_brain_repository(sqlite_path=":memory:")
    brain = KIWebsearchBrain(repo, BrainSecrets())

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
        SimpleNamespace(name="models/gemini-3.9-flash-preview", display_name="Gemini 3.9 Flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.9-dead-preview", display_name="Gemini Dead", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/text-embedding-004", display_name="Embed", supported_actions=["embedContent"]),
    ]
    probed_gemini_calls: list[str] = []

    def _fake_gemini_create(**kwargs: object) -> MagicMock:
        mid = str(kwargs.get("model") or "")
        probed_gemini_calls.append(mid)
        if mid == "gemini-3.9-dead-preview":
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
    brain = KIWebsearchBrain(repo, secrets, genai_client=mock_client)

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
    assert "gemini-3.9-flash-preview" in summary.added_models
    assert "groq/llama-3.1-8b-instant" in summary.added_models
    assert "or/qwen/qwen-2.5-free" in summary.added_models
    assert "gemini-3.9-dead-preview" in summary.skipped_unhealthy
    assert "or/broken/dead-free" in summary.skipped_unhealthy
    assert brain.get_model("gemini-3.9-flash-preview") is not None
    assert brain.get_model("gemini-3.9-flash-preview").last_status == ProbeStatus.OK  # type: ignore[union-attr]
    assert brain.get_model("gemini-3.9-dead-preview") is None
    assert brain.get_model("or/broken/dead-free") is None
    assert brain.get_model("text-embedding-004") is None

    # Manuell gelöschte + defekt verworfene Modelle kommen beim nächsten Auto-Sync NICHT zurück
    assert brain.remove_model("gemini-3.9-flash-preview") is True
    probed_gemini_calls.clear()
    summary_again = brain.auto_sync()
    assert brain.get_model("gemini-3.9-flash-preview") is None
    assert brain.get_model("gemini-3.9-dead-preview") is None
    assert brain.get_model("or/broken/dead-free") is None
    assert "gemini-3.9-flash-preview" in summary_again.skipped_excluded
    assert "gemini-3.9-dead-preview" in summary_again.skipped_excluded
    assert "or/broken/dead-free" in summary_again.skipped_excluded
    assert probed_gemini_calls == []
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
    brain = KIWebsearchBrain(repo, secrets, genai_client=mock_client)

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
    assert deep_seq.use_deep_agent is True
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

    brain = KIWebsearchBrain(repo, BrainSecrets(gemini_api_key="AIzaSyTest123"), genai_client=mock_client)

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
    no_key_brain = KIWebsearchBrain(create_brain_repository(sqlite_path=":memory:"), BrainSecrets())
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
    brain = KIWebsearchBrain(repo, BrainSecrets())

    # 1. Modelle & Tools auf Error / Quota 429 / Blocked / Missing Key setzen und bereinigen
    m_err = brain.get_model("gemini-2.0-flash")
    t_q429 = brain.get_tool("gemini_google_search")
    t_blocked = brain.get_tool("duckduckgo_html")
    t_missing = brain.get_tool("browserbase_fetch")
    assert m_err is not None and t_q429 is not None and t_blocked is not None and t_missing is not None

    brain.add_or_update_model(m_err.model_copy(update={"last_status": ProbeStatus.ERROR}))
    brain.add_or_update_tool(t_q429.model_copy(update={"last_status": ProbeStatus.QUOTA_LIMITED}))
    brain.add_or_update_tool(t_blocked.model_copy(update={"last_status": ProbeStatus.BLOCKED}))
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
    assert rem_models == ["gemini-2.0-flash"]
    assert rem_tools == ["gemini_google_search"]
    assert brain.get_model("gemini-2.0-flash") is None
    assert brain.get_tool("gemini_google_search") is None
    # Blocked-Tools und Missing-Key-Einträge MÜSSEN zur Analyse/Key-Eintragung erhalten bleiben!
    kept_blocked = brain.get_tool("duckduckgo_html")
    assert kept_blocked is not None and kept_blocked.last_status == ProbeStatus.BLOCKED
    assert brain.get_tool("browserbase_fetch") is not None

    # Nach Cleanup darf auto_sync bereinigte Items nicht wieder aufnehmen
    sync_after_clean = brain.auto_sync()
    assert brain.get_model("gemini-2.0-flash") is None
    assert brain.get_tool("gemini_google_search") is None
    assert "gemini-2.0-flash" in sync_after_clean.skipped_excluded
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
    assert str(raw_db_row["encrypted_value"]).startswith("enc_v1:")
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

    reopened = KIWebsearchBrain(create_brain_repository(sqlite_path=db_file), BrainSecrets())
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
    brain = KIWebsearchBrain(repo, BrainSecrets())
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
    for tid in ("brave_search", "tavily_search", "serper_search", "openalex_api", "stackexchange_api", "tagesschau_rss"):
        repo.delete_tool(tid)

    brain = KIWebsearchBrain(repo, BrainSecrets())
    monkeypatch.setattr("ki_brain.discovery.safe_http_request", lambda *_a, **_kw: (404, "https://searx.space", "text/plain", b""))
    sync_sum = brain.auto_sync(verify_discovered=False)
    for new_tid in ("brave_search", "tavily_search", "serper_search", "openalex_api", "stackexchange_api", "tagesschau_rss"):
        assert new_tid in sync_sum.added_tools
        assert brain.get_tool(new_tid) is not None
    assert brain.get_tool("brave_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    assert brain.get_tool("tavily_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]
    assert brain.get_tool("serper_search").last_status == ProbeStatus.AUTH_MISSING  # type: ignore[union-attr]

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
    brain = KIWebsearchBrain(repo, BrainSecrets())

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


