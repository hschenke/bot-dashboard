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
        SimpleNamespace(name="models/text-embedding-004", display_name="Embed", supported_actions=["embedContent"]),
    ]
    secrets = BrainSecrets(
        gemini_api_key="AIzaSyTestKey123",
        groq_api_key="gsk_test_123",
        openrouter_api_key="sk-or-test-123",
    )
    brain = KIWebsearchBrain(repo, secrets, genai_client=mock_client)

    def _fake_http(_method: str, url: str, **_kw: object) -> tuple[int, str, str, bytes]:
        if "groq.com" in url:
            return 200, url, "application/json", b'{"data": [{"id": "llama-3.1-8b-instant"}]}'
        if "openrouter.ai" in url:
            return 200, url, "application/json", b'{"data": [{"id": "qwen/qwen-2.5:free", "name": "Qwen Free", "pricing": {"prompt": "0", "completion": "0"}}]}'
        return 404, url, "text/plain", b""

    monkeypatch.setattr("ki_brain.discovery.safe_http_request", _fake_http)
    summary = brain.auto_sync()
    assert "gemini-3.9-flash-preview" in summary.added_models
    assert "groq/llama-3.1-8b-instant" in summary.added_models
    assert "or/qwen/qwen-2.5-free" in summary.added_models
    assert brain.get_model("gemini-3.9-flash-preview") is not None
    assert brain.get_model("text-embedding-004") is None
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
