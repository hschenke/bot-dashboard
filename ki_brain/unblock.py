"""Automatische Diagnose-, Rule- und Fallback-Engine für `Blocked`-Tools (`🔓 Unblock Tools`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ki_brain.benchmark import BrainBenchmarkRunner
from ki_brain.enums import ProbeStatus, ToolCategory
from ki_brain.llm_runner import UnifiedLLMRunner
from ki_brain.models import BrainSecrets, ToolSpec
from ki_brain.optimizer import DEFAULT_TEST_QUERY, DEFAULT_TEST_URL, optimize_tier_sequences
from ki_brain.storage import BrainRepository
from ki_brain.tools.searxng import build_searxng_tool, discover_verified_searxng_pool

DEFAULT_PROBE_QUERY_RULES: dict[str, str] = {
    "duckduckgo_instant": "Python",
    "open_meteo": "Berlin",
    "wikidata_api": "Python",
    "gemini_google_maps": "Cafés am Alexanderplatz in Berlin",
}


def resolve_probe_query(tool: ToolSpec, fallback_query: str = DEFAULT_TEST_QUERY) -> str:
    """Ermittelt die optimale Testlauf-Query aus Tool-Regel (`config['probe_query']`) oder Rule-Map."""
    cfg_query = str((tool.config or {}).get("probe_query") or "").strip()
    if cfg_query:
        return cfg_query
    rule_query = DEFAULT_PROBE_QUERY_RULES.get(tool.executor_type.strip().lower(), "")
    return rule_query or fallback_query


@dataclass(frozen=True)
class UnblockSummary:
    """Ergebnis eines automatischen `Unblock Tools`-Durchlaufs."""

    checked_blocked: list[str] = field(default_factory=list)
    rules_applied: list[str] = field(default_factory=list)
    unblocked_ids: list[str] = field(default_factory=list)
    still_blocked: dict[str, str] = field(default_factory=dict)
    activated_alternatives: list[str] = field(default_factory=list)


def _apply_tool_remediation_rules(tool: ToolSpec) -> tuple[ToolSpec, str | None]:
    etype = tool.executor_type.strip().lower()
    cfg = dict(tool.config or {})
    if etype in DEFAULT_PROBE_QUERY_RULES and cfg.get("probe_query") != DEFAULT_PROBE_QUERY_RULES[etype]:
        target_q = DEFAULT_PROBE_QUERY_RULES[etype]
        cfg["probe_query"] = target_q
        return tool.model_copy(update={"config": cfg}), f"{tool.id}: Probe-Query-Regel '{target_q}' hinterlegt"

    if etype == "reddit_search" and (tool.requires_api_key or "oauth.reddit.com" in tool.endpoint_url):
        updated = tool.model_copy(
            update={
                "endpoint_url": "https://api.pullpush.io/reddit/search/submission/",
                "requires_api_key": False,
                "api_key_env": "",
                "description": "Key-freie Reddit-Community-Suche (PullPush, Reddit RSS/JSON & Yahoo-Fallback).",
            }
        )
        return updated, f"{tool.id}: Auf key-freien PullPush/Feed/Site-Fallback umgestellt"

    return tool, None


def _try_yahoo_alternate_endpoint(
    bench: BrainBenchmarkRunner,
    tool: ToolSpec,
    test_query: str,
    test_url: str,
) -> tuple[ToolSpec, Any] | None:
    if tool.executor_type.strip().lower() != "yahoo_search":
        return None
    candidates = [
        ep
        for ep in (
            "https://search.yahoo.com/search",
            "https://ch.search.yahoo.com/search",
            "https://at.search.yahoo.com/search",
            "https://uk.search.yahoo.com/search",
            "https://de.search.yahoo.com/search",
        )
        if ep != tool.endpoint_url
    ]
    for alt_url in candidates[:3]:
        alt_tool = tool.model_copy(update={"endpoint_url": alt_url})
        ut_alt, entry_alt = bench.probe_tool(alt_tool, test_query, test_url, None)
        if ut_alt.last_status == ProbeStatus.OK:
            return ut_alt, entry_alt
    return None


def resolve_blocked_tools(
    repo: BrainRepository,
    secrets: BrainSecrets,
    runner: UnifiedLLMRunner,
    *,
    test_query: str = DEFAULT_TEST_QUERY,
    test_url: str = DEFAULT_TEST_URL,
    on_progress: Callable[[float, str], None] | None = None,
) -> UnblockSummary:
    """Wendet Regeln auf `Blocked`-Tools an, aktiviert `searx.space`-Pool und testet live nach."""
    from ki_brain.quotas import select_grounding_model

    if on_progress:
        on_progress(0.08, "Analysiere Blocked-Tools und wende Remediation-Rules an...")
    all_tools = repo.list_tools(enabled_only=True)
    blocked = [t for t in all_tools if t.last_status == ProbeStatus.BLOCKED]
    _, excluded_tools = repo.list_excluded_ids()

    checked_ids = [t.id for t in blocked]
    rules_applied: list[str] = []
    activated_alts: list[str] = []
    reprobe_map: dict[str, ToolSpec] = {}

    for tool in blocked:
        fixed, note = _apply_tool_remediation_rules(tool)
        if note:
            repo.upsert_tool(fixed)
            rules_applied.append(note)
        reprobe_map[fixed.id] = fixed

    has_blocked_websearch = any(t.category == ToolCategory.WEB_SEARCH for t in blocked)
    searx_existing = repo.get_tool("searxng_public")
    need_searxng = (
        "searxng_public" not in excluded_tools
        and (searx_existing is None or searx_existing.source != "custom")
        and (
            (searx_existing is not None and searx_existing.last_status in {ProbeStatus.BLOCKED, ProbeStatus.ERROR})
            or (has_blocked_websearch and (searx_existing is None or searx_existing.last_status != ProbeStatus.OK))
        )
    )
    if need_searxng:
        if on_progress:
            on_progress(0.28, "Suche und prüfe erreichbare SearXNG-Instanzen (searx.space + Fallback-Pool)...")
        verified_pool = discover_verified_searxng_pool(max_verify=10, max_keep=4)
        if verified_pool:
            searx_tool = build_searxng_tool(verified_pool)
            repo.upsert_tool(searx_tool)
            reprobe_map[searx_tool.id] = searx_tool
            activated_alts.append(f"searxng_public ({searx_tool.endpoint_url})")

    bench = BrainBenchmarkRunner(repo, secrets, runner)
    models_enabled = repo.list_models(enabled_only=True)
    usage_state = repo.load_usage_quotas()
    updated_entries = []
    unblocked_ids: list[str] = []
    still_blocked: dict[str, str] = {}
    targets = list(reprobe_map.values())

    for idx, tool in enumerate(targets, start=1):
        if on_progress:
            on_progress(
                0.38 + 0.50 * (idx / max(1, len(targets))),
                f"Teste Tool live nach ({idx}/{len(targets)}): `{tool.id}`...",
            )
        g_model = (
            select_grounding_model(models_enabled, secrets, executor_type=tool.executor_type, allow_paid=False)
            if tool.category == ToolCategory.AI_GROUNDING
            else None
        )
        ut, entry = bench.probe_tool(tool, test_query, test_url, g_model, usage_state=usage_state)
        if ut.last_status == ProbeStatus.BLOCKED:
            yahoo_alt = _try_yahoo_alternate_endpoint(bench, ut, test_query, test_url)
            if yahoo_alt is not None:
                ut, entry = yahoo_alt
                rules_applied.append(f"{ut.id}: Alternativ-Endpunkt {ut.endpoint_url} aktiviert")
        repo.upsert_tool(ut)
        updated_entries.append(entry)
        if ut.last_status == ProbeStatus.OK and ut.id in checked_ids:
            unblocked_ids.append(ut.id)
        elif ut.last_status == ProbeStatus.BLOCKED:
            still_blocked[ut.id] = ut.last_error or "Blocked (0 Treffer)"

    if on_progress:
        on_progress(0.92, "Aktualisiere Fast-, Standard- und Deep-Sequenzen...")
    updated_seqs = repo.list_sequences()
    if unblocked_ids or activated_alts:
        seqs = optimize_tier_sequences(
            repo.list_models(),
            repo.list_tools(),
            updated_seqs,
            force_overwrite=False,
            usage_state=repo.load_usage_quotas(),
        )
        updated_seqs = {t: repo.upsert_sequence(seq) for t, seq in seqs.items()}

    latest = repo.get_latest_benchmark()
    if latest is not None and updated_entries:
        by_key = {(e.item_type, e.item_id): e for e in latest.entries}
        for new_entry in updated_entries:
            by_key[(new_entry.item_type, new_entry.item_id)] = new_entry
        merged = list(by_key.values())
        repo.save_benchmark_run(
            latest.model_copy(
                update={
                    "entries": merged,
                    "sequences": {t.value: cfg for t, cfg in updated_seqs.items()},
                    "healthy_tools_count": sum(1 for e in merged if e.item_type == "tool" and e.status == ProbeStatus.OK),
                    "healthy_models_count": sum(1 for e in merged if e.item_type == "model" and e.status == ProbeStatus.OK),
                }
            )
        )

    if on_progress:
        on_progress(1.0, "Unblock Tools abgeschlossen.")

    return UnblockSummary(
        checked_blocked=checked_ids,
        rules_applied=rules_applied,
        unblocked_ids=unblocked_ids,
        still_blocked=still_blocked,
        activated_alternatives=activated_alts,
    )
