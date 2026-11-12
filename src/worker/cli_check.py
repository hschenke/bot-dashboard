"""CLI-Entrypoint (`python -m src.worker.cli_check`) für automatische Health-Checks."""

from __future__ import annotations

import logging
import sys

from src.checks.coordinator import FleetCheckCoordinator
from src.config.settings import load_app_settings
from src.storage.repository import create_dashboard_repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_cli_fleet_check() -> int:
    """Prüft alle konfigurierten Apps und protokolliert das Ergebnis im Dashboard-Repository."""
    settings = load_app_settings()
    repository = create_dashboard_repository(settings)
    coordinator = FleetCheckCoordinator(settings)

    try:
        snapshots = coordinator.check_all_apps(repository=repository)
        for snap in snapshots:
            logger.info(
                "%s (%s): Gesamt=%s | URL=%s (%d ms) | Worker=%s | DB=%s (%d Zeilen)",
                snap.spec.title,
                snap.spec.app_id.value,
                snap.overall_status.value,
                snap.url_check.status.value,
                snap.url_check.latency_ms,
                snap.workflow_check.state.value,
                snap.db_check.status.value,
                snap.db_check.total_rows,
            )
        return 0
    finally:
        repository.close()


if __name__ == "__main__":
    sys.exit(run_cli_fleet_check())
