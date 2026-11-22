"""Aufgeschlüsselte Gesundheits-Diagnosen für Web-App, Worker & Datenbank."""

from __future__ import annotations

from src.domain.enums import HealthStatus, WorkflowRunState
from src.domain.models import (
    AppFleetSnapshot,
    DatabaseHealthSummary,
    DatabaseIssue,
    HealthDiagnostic,
    UrlCheckResult,
    WorkflowRunSummary,
)

_ISSUE_STATES: frozenset[HealthStatus] = frozenset(
    {HealthStatus.DEGRADED, HealthStatus.OFFLINE, HealthStatus.ASLEEP}
)


def _format_db_issue_line(issue: DatabaseIssue) -> str:
    head = f"{issue.source}: {issue.title} [{issue.status}]"
    return f"{head} — {issue.detail}" if issue.detail else head


def _diagnose_url(url_check: UrlCheckResult) -> HealthDiagnostic:
    return HealthDiagnostic(
        component="Web-App",
        status=url_check.status,
        summary=url_check.message or url_check.status.value,
    )


def _diagnose_worker(wf: WorkflowRunSummary) -> HealthDiagnostic:
    if wf.state == WorkflowRunState.FAILURE:
        detail = wf.message or wf.status_label or "Failed"
        return HealthDiagnostic(
            component="Worker",
            status=HealthStatus.DEGRADED,
            summary=f"Failed ({wf.workflow_file}): {detail}",
        )
    if wf.state in {WorkflowRunState.SUCCESS, WorkflowRunState.RUNNING, WorkflowRunState.QUEUED}:
        label = wf.status_label or wf.state.value
        return HealthDiagnostic(
            component="Worker",
            status=HealthStatus.ONLINE,
            summary=f"{label} ({wf.workflow_file})",
        )
    return HealthDiagnostic(
        component="Worker",
        status=HealthStatus.UNKNOWN,
        summary=wf.message or wf.status_label or "Unknown",
    )


def _diagnose_db(db: DatabaseHealthSummary) -> HealthDiagnostic:
    issue_lines = tuple(_format_db_issue_line(item) for item in db.issues)
    if db.status == HealthStatus.DEGRADED:
        if db.error_count > 0:
            summary = (
                f"Verbindung Online ({db.backend_label}, {db.latency_ms} ms), "
                f"aber {db.error_count} Fehler in App-Daten"
            )
        else:
            summary = db.message or f"Degraded ({db.backend_label})"
        return HealthDiagnostic(
            component="Datenbank",
            status=HealthStatus.DEGRADED,
            summary=summary,
            details=issue_lines,
        )
    if db.status == HealthStatus.OFFLINE:
        return HealthDiagnostic(
            component="Datenbank",
            status=HealthStatus.OFFLINE,
            summary=db.message or "Datenbank nicht erreichbar",
            details=issue_lines,
        )
    return HealthDiagnostic(
        component="Datenbank",
        status=db.status,
        summary=db.message or f"Online ({db.backend_label})",
    )


def collect_health_diagnostics(
    url_check: UrlCheckResult,
    workflow_check: WorkflowRunSummary,
    db_check: DatabaseHealthSummary,
) -> tuple[HealthDiagnostic, ...]:
    """Liefert den aufgeschlüsselten Gesundheitszustand für Web-App, Worker und Datenbank."""
    return (
        _diagnose_url(url_check),
        _diagnose_worker(workflow_check),
        _diagnose_db(db_check),
    )


def get_issue_diagnostics(snapshot: AppFleetSnapshot) -> list[HealthDiagnostic]:
    """Filtert nur die Komponenten heraus, die aktuell `Degraded`, `Offline` oder `Sleeping` sind."""
    diagnostics = collect_health_diagnostics(
        snapshot.url_check,
        snapshot.workflow_check,
        snapshot.db_check,
    )
    return [diag for diag in diagnostics if diag.status in _ISSUE_STATES]


def format_degraded_cause_short(snapshot: AppFleetSnapshot) -> str:
    """Erstellt ein kurzes englisches/kompaktes Ursachen-Label für Karten-Header & Logs."""
    parts: list[str] = []
    if snapshot.url_check.status in {HealthStatus.DEGRADED, HealthStatus.OFFLINE}:
        url_tag = "Offline" if snapshot.url_check.status == HealthStatus.OFFLINE else "/_stcore/health"
        parts.append(f"URL: {url_tag}")
    if snapshot.workflow_check.state == WorkflowRunState.FAILURE:
        parts.append("Worker: Failed")
    if snapshot.db_check.status == HealthStatus.OFFLINE:
        parts.append("DB: Offline")
    elif snapshot.db_check.status == HealthStatus.DEGRADED:
        if snapshot.db_check.error_count > 0:
            parts.append(f"DB: {snapshot.db_check.error_count} Fehler")
        else:
            parts.append("DB: Degraded")
    return " · ".join(parts)
