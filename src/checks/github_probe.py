"""GitHub Actions Workflow-Monitor & 1-Klick-Trigger über die GitHub REST API."""

from __future__ import annotations

from typing import Any

import requests

from src.config.constants import GITHUB_API_TIMEOUT_SECONDS
from src.domain.enums import WorkflowRunState
from src.domain.exceptions import ProbeError, ValidationError
from src.domain.models import ManagedAppSpec, WorkflowRunSummary
from src.security.sanitization import validate_github_repo_slug, validate_workflow_filename


def _map_workflow_state(status: str, conclusion: str) -> tuple[WorkflowRunState, str]:
    clean_status = (status or "").strip().lower()
    clean_conclusion = (conclusion or "").strip().lower()

    if clean_status == "in_progress":
        return WorkflowRunState.RUNNING, "Läuft gerade (Active Worker)"
    if clean_status in {"queued", "waiting", "requested", "pending"}:
        return WorkflowRunState.QUEUED, "In Warteschlange (Queued)"
    if clean_status == "completed":
        if clean_conclusion == "success":
            return WorkflowRunState.SUCCESS, "Erfolgreich abgeschlossen"
        if clean_conclusion in {"failure", "timed_out", "startup_failure"}:
            return WorkflowRunState.FAILURE, f"Fehlgeschlagen ({clean_conclusion})"
        if clean_conclusion in {"cancelled", "skipped"}:
            return WorkflowRunState.CANCELLED, f"Abgebrochen / Übersprungen ({clean_conclusion})"
    return WorkflowRunState.UNKNOWN, f"Status: {clean_status or 'unbekannt'}"


class GitHubWorkflowProbe:
    """Liest den aktuellen Worker-Status auf GitHub Actions und startet Läufe per Dispatch."""

    def __init__(
        self,
        github_token: str = "",
        session: requests.Session | None = None,
    ) -> None:
        self._token = github_token.strip()
        self._session = session or requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "BotDashboard-GitHubProbe/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._session.headers.update(headers)

    def check_primary_workflow(self, spec: ManagedAppSpec) -> WorkflowRunSummary:
        """Prüft den letzten Lauf des primären Hintergrund-Workflows einer App."""
        return self.check_workflow(spec, spec.primary_workflow)

    def check_workflow(self, spec: ManagedAppSpec, workflow_file: str) -> WorkflowRunSummary:
        """Fragt den jüngsten Workflow-Run für `workflow_file` im Repository der App ab."""
        try:
            repo_slug = validate_github_repo_slug(spec.github_repo)
            wf_name = validate_workflow_filename(workflow_file)
        except ValidationError as exc:
            return WorkflowRunSummary(
                app_id=spec.app_id,
                workflow_file=workflow_file,
                state=WorkflowRunState.UNKNOWN,
                status_label="Konfigurationsfehler",
                conclusion="",
                run_url=spec.github_actions_url,
                branch="main",
                started_at="",
                updated_at="",
                message=str(exc),
            )

        api_url = (
            f"https://api.github.com/repos/{repo_slug}/actions/workflows/{wf_name}/runs?per_page=1"
        )
        try:
            resp = self._session.get(api_url, timeout=GITHUB_API_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                return WorkflowRunSummary(
                    app_id=spec.app_id,
                    workflow_file=wf_name,
                    state=WorkflowRunState.UNKNOWN,
                    status_label=f"GitHub HTTP {resp.status_code}",
                    conclusion="",
                    run_url=spec.github_actions_url,
                    branch="main",
                    started_at="",
                    updated_at="",
                    message=f"GitHub API antwortete mit Status {resp.status_code}.",
                )
            data: dict[str, Any] = resp.json()
        except requests.RequestException as exc:
            return WorkflowRunSummary(
                app_id=spec.app_id,
                workflow_file=wf_name,
                state=WorkflowRunState.UNKNOWN,
                status_label="API nicht erreichbar",
                conclusion="",
                run_url=spec.github_actions_url,
                branch="main",
                started_at="",
                updated_at="",
                message=f"GitHub API Fehler: {exc.__class__.__name__}",
            )

        runs = data.get("workflow_runs", [])
        if not isinstance(runs, list) or not runs:
            return WorkflowRunSummary(
                app_id=spec.app_id,
                workflow_file=wf_name,
                state=WorkflowRunState.UNKNOWN,
                status_label="Noch keine Läufe",
                conclusion="",
                run_url=spec.github_actions_url,
                branch="main",
                started_at="",
                updated_at="",
                message=f"Keine Ausführungen für {wf_name} gefunden.",
            )

        latest = runs[0]
        raw_status = str(latest.get("status", "") or "")
        raw_conclusion = str(latest.get("conclusion", "") or "")
        state, label = _map_workflow_state(raw_status, raw_conclusion)

        return WorkflowRunSummary(
            app_id=spec.app_id,
            workflow_file=wf_name,
            state=state,
            status_label=label,
            conclusion=raw_conclusion,
            run_url=str(latest.get("html_url", "") or spec.github_actions_url),
            branch=str(latest.get("head_branch", "main") or "main"),
            started_at=str(latest.get("run_started_at", "") or latest.get("created_at", "") or ""),
            updated_at=str(latest.get("updated_at", "") or ""),
            message=f"{wf_name}: {label}",
        )

    def trigger_workflow_dispatch(
        self,
        spec: ManagedAppSpec,
        workflow_file: str | None = None,
        branch: str = "main",
        inputs: dict[str, str] | None = None,
    ) -> str:
        """Startet einen GitHub Actions Workflow per `workflow_dispatch`."""
        if not self._token:
            raise ProbeError("Kein GITHUB_TOKEN für Workflow-Dispatch konfiguriert.")

        repo_slug = validate_github_repo_slug(spec.github_repo)
        wf_name = validate_workflow_filename(workflow_file or spec.primary_workflow)
        api_url = (
            f"https://api.github.com/repos/{repo_slug}/actions/workflows/{wf_name}/dispatches"
        )
        payload: dict[str, Any] = {"ref": branch.strip() or "main"}
        if inputs:
            payload["inputs"] = inputs

        try:
            resp = self._session.post(
                api_url,
                json=payload,
                timeout=GITHUB_API_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise ProbeError(f"GitHub Dispatch fehlgeschlagen: {exc}") from exc

        if resp.status_code not in {200, 201, 204}:
            raise ProbeError(
                f"GitHub Dispatch abgelehnt (HTTP {resp.status_code}). Prüfe Token-Rechte (Actions: Write)."
            )
        return f"Workflow '{wf_name}' in {repo_slug} erfolgreich gestartet."
