"""Bot-spezifische Detail-Inspektionen (News-, Research- & Alert-Bot) inkl. Fehler-Aufschlüsselung."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.constants import MAX_DB_ISSUES_PER_APP, MAX_RECENT_ITEMS_PER_APP
from src.domain.exceptions import StorageError
from src.domain.models import DatabaseIssue, DatabaseRecentItem
from src.security.sanitization import sanitize_display_url
from src.storage.executor import SqlExecutor


@dataclass(frozen=True)
class BotDbInspectionDetails:
    """Strukturierte Detail-Ergebnisse aus einer einzelnen Bot-Datenbank."""

    items: tuple[DatabaseRecentItem, ...] = ()
    active_count: int = 0
    error_count: int = 0
    last_activity: str = ""
    issues: tuple[DatabaseIssue, ...] = ()


def _query_fallback(
    executor: SqlExecutor, primary_sql: str, fallback_sql: str, params: tuple[object, ...] = ()
) -> list[dict[str, object]]:
    """Führt primäre SQL-Query aus und fällt bei älteren Minimal-Schemata auf Fallback zurück."""
    try:
        return executor.execute(primary_sql, params).rows
    except StorageError:
        return executor.execute(fallback_sql, params).rows


def _first_ts(current: str, candidate: str) -> str:
    return current or candidate


def inspect_news_bot_details(executor: SqlExecutor) -> BotDbInspectionDetails:
    """Liest die jüngsten Artikel und AI-Briefings aus der News-Bot-Datenbank."""
    items: list[DatabaseRecentItem] = []
    last_ts = ""
    try:
        rows = executor.execute(
            "SELECT briefing_date, model_used, created_at FROM briefings ORDER BY id DESC LIMIT ?", (3,)
        ).rows
        for r in rows:
            ts = str(r.get("created_at", "") or r.get("briefing_date", ""))
            last_ts = _first_ts(last_ts, ts)
            items.append(DatabaseRecentItem("Briefing", f"AI-Briefing vom {r.get('briefing_date', '-')}", f"Modell: {r.get('model_used', 'unbekannt')}", "completed", ts))
    except StorageError:
        pass
    try:
        rows = executor.execute(
            "SELECT title, source, category, created_at, url FROM articles ORDER BY timestamp DESC LIMIT ?",
            (MAX_RECENT_ITEMS_PER_APP,),
        ).rows
        for r in rows:
            ts = str(r.get("created_at", "") or "")
            last_ts = _first_ts(last_ts, ts)
            items.append(
                DatabaseRecentItem(
                    str(r.get("category", "") or "Artikel"),
                    str(r.get("title", "") or "Ohne Titel")[:120],
                    str(r.get("source", "") or "RSS"),
                    "saved",
                    ts,
                    sanitize_display_url(str(r.get("url", "") or "")),
                )
            )
    except StorageError:
        pass
    trimmed = tuple(items[:MAX_RECENT_ITEMS_PER_APP])
    return BotDbInspectionDetails(items=trimmed, active_count=len(trimmed), last_activity=last_ts)


def _fetch_unresolved_research_failures(
    executor: SqlExecutor, last_ok: str
) -> tuple[int, tuple[DatabaseIssue, ...]]:
    """Zählt und beschreibt nur fehlgeschlagene Jobs, die neuer sind als der letzte erfolgreiche Lauf."""
    where_order = (
        "FROM research_jobs WHERE status = 'failed' "
        "AND (? = '' OR COALESCE(NULLIF(completed_at, ''), created_at) > ?) "
        "ORDER BY COALESCE(NULLIF(completed_at, ''), created_at) DESC LIMIT ?"
    )
    rows = _query_fallback(
        executor,
        f"SELECT topic, phase, progress_percent, error_message, created_at, completed_at {where_order}",
        f"SELECT topic, phase, progress_percent, created_at, completed_at {where_order}",
        (last_ok, last_ok, 100),
    )
    issues = tuple(
        DatabaseIssue(
            source="Research-Job",
            title=str(r.get("topic", "Unbekanntes Thema"))[:120],
            status=f"failed (Phase: {r.get('phase', '-') or '-'}, {int(r.get('progress_percent', 0) or 0)}%)",
            detail=str(r.get("error_message", "") or "").strip()[:220] or "Letzter Research-Job fehlgeschlagen.",
            timestamp=str(r.get("completed_at", "") or r.get("created_at", "") or ""),
        )
        for r in rows[:MAX_DB_ISSUES_PER_APP]
    )
    return len(rows), issues


def inspect_research_bot_details(executor: SqlExecutor) -> BotDbInspectionDetails:
    """Liest aktive/jüngste Jobs und noch ungelöste Fehler aus der Research-Bot-Datenbank."""
    items: list[DatabaseRecentItem] = []
    active_cnt, err_cnt, last_ts = 0, 0, ""
    issues: tuple[DatabaseIssue, ...] = ()
    try:
        counts = executor.execute(
            "SELECT SUM(CASE WHEN status IN ('running', 'queued') THEN 1 ELSE 0 END) AS active_cnt, "
            "MAX(CASE WHEN status = 'completed' THEN COALESCE(NULLIF(completed_at, ''), created_at) ELSE NULL END) AS last_ok, "
            "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_cnt FROM research_jobs"
        ).rows
        if counts:
            active_cnt = int(counts[0].get("active_cnt", 0) or 0)
            last_ok = str(counts[0].get("last_ok", "") or "")
            if int(counts[0].get("failed_cnt", 0) or 0) > 0:
                err_cnt, issues = _fetch_unresolved_research_failures(executor, last_ok)
        jobs = executor.execute(
            "SELECT topic, status, phase, progress_percent, created_at, completed_at "
            "FROM research_jobs ORDER BY created_at DESC LIMIT ?",
            (MAX_RECENT_ITEMS_PER_APP,),
        ).rows
        for r in jobs:
            ts = str(r.get("completed_at", "") or r.get("created_at", "") or "")
            last_ts = _first_ts(last_ts, ts)
            items.append(
                DatabaseRecentItem(
                    "Research-Job",
                    str(r.get("topic", "Unbekanntes Thema"))[:120],
                    f"Phase: {r.get('phase', '-')} ({int(r.get('progress_percent', 0) or 0)}%)",
                    str(r.get("status", "unknown")),
                    ts,
                )
            )
    except StorageError:
        pass
    return BotDbInspectionDetails(tuple(items), active_cnt, err_cnt, last_ts, issues)


def _fetch_alert_rule_issues(executor: SqlExecutor) -> tuple[DatabaseIssue, ...]:
    """Liest alle aktiven/fehlerhaften Alert-Regeln mit Fehlerzähler und letzter Fehlermeldung."""
    clause = (
        "FROM alert_rules WHERE status = 'error' OR (status = 'active' AND consecutive_failures > 0) "
        "ORDER BY consecutive_failures DESC, updated_at DESC LIMIT ?"
    )
    rows = _query_fallback(
        executor,
        f"SELECT title, detector_type, status, consecutive_failures, last_error, last_checked_at, updated_at, target_url {clause}",
        f"SELECT title, detector_type, status, consecutive_failures, last_checked_at, updated_at, target_url {clause}",
        (MAX_DB_ISSUES_PER_APP,),
    )
    issues: list[DatabaseIssue] = []
    for r in rows:
        rule_st = str(r.get("status", "error") or "error")
        fails = int(r.get("consecutive_failures", 0) or 0)
        last_err = str(r.get("last_error", "") or "").strip()
        issues.append(
            DatabaseIssue(
                source=f"Alert-Regel ({r.get('detector_type', '-')})",
                title=str(r.get("title", "Regel"))[:120],
                status=f"{rule_st} ({fails}× fehlgeschlagen)" if fails > 0 else rule_st,
                detail=last_err[:220] or "Letzter Regel-Check fehlgeschlagen (keine Detailmeldung gespeichert).",
                timestamp=str(r.get("last_checked_at", "") or r.get("updated_at", "") or ""),
                url=sanitize_display_url(str(r.get("target_url", "") or "")),
            )
        )
    return tuple(issues)


def _build_alert_rule_item(r: dict[str, object]) -> tuple[DatabaseRecentItem, str]:
    ts = str(r.get("last_checked_at", "") or "")
    rule_st = str(r.get("status", "active") or "active")
    val_txt = str(r.get("last_value_text", "") or "Noch kein Messwert")
    fails = int(r.get("consecutive_failures", 0) or 0)
    last_err = str(r.get("last_error", "") or "").strip()
    if rule_st != "paused" and (rule_st == "error" or fails > 0):
        err_hint = f"⚠️ {fails}× Fehler: {last_err[:70]}" if last_err else f"⚠️ {fails}× Fehler"
        subtitle = f"{val_txt[:50]} · {err_hint}" if val_txt != "Noch kein Messwert" else err_hint
    else:
        subtitle = val_txt[:100]
    item = DatabaseRecentItem(
        f"Wächter ({r.get('detector_type', '-')})",
        str(r.get("title", "Regel"))[:120],
        subtitle[:120],
        rule_st,
        ts,
        sanitize_display_url(str(r.get("target_url", "") or "")),
    )
    return item, ts


def inspect_alert_bot_details(executor: SqlExecutor) -> BotDbInspectionDetails:
    """Liest aktive Wächter-Regeln, akute Regelfehler (ohne pausierte Regeln) und Alarme."""
    items: list[DatabaseRecentItem] = []
    active_cnt, err_cnt, last_ts = 0, 0, ""
    issues: tuple[DatabaseIssue, ...] = ()
    try:
        counts = executor.execute(
            "SELECT SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_cnt, "
            "SUM(CASE WHEN status = 'error' OR (status = 'active' AND consecutive_failures > 0) THEN 1 ELSE 0 END) AS err_cnt "
            "FROM alert_rules"
        ).rows
        if counts:
            active_cnt = int(counts[0].get("active_cnt", 0) or 0)
            err_cnt = int(counts[0].get("err_cnt", 0) or 0)
        if err_cnt > 0:
            issues = _fetch_alert_rule_issues(executor)
        rules = _query_fallback(
            executor,
            "SELECT title, detector_type, status, last_value_text, last_checked_at, target_url, consecutive_failures, last_error FROM alert_rules ORDER BY updated_at DESC LIMIT ?",
            "SELECT title, detector_type, status, last_value_text, last_checked_at, target_url, consecutive_failures FROM alert_rules ORDER BY updated_at DESC LIMIT ?",
            (5,),
        )
        for r in rules:
            item, ts = _build_alert_rule_item(r)
            last_ts = _first_ts(last_ts, ts)
            items.append(item)
        events = executor.execute(
            "SELECT rule_title, headline, measured_display, created_at, source_url FROM alert_events ORDER BY created_at DESC LIMIT ?",
            (4,),
        ).rows
        for r in events:
            ts = str(r.get("created_at", "") or "")
            last_ts = _first_ts(last_ts, ts)
            items.append(
                DatabaseRecentItem(
                    "Alarm-Treffer",
                    str(r.get("headline", r.get("rule_title", "Alarm")))[:120],
                    str(r.get("measured_display", ""))[:100],
                    "triggered",
                    ts,
                    sanitize_display_url(str(r.get("source_url", "") or "")),
                )
            )
    except StorageError:
        pass
    return BotDbInspectionDetails(tuple(items[:MAX_RECENT_ITEMS_PER_APP]), active_cnt, err_cnt, last_ts, issues)
