"""CLI-Entrypoint (`python -m src.worker.cli_check`) für automatische Health-Checks & Wakeups."""

from __future__ import annotations

import argparse
import logging
import sys

from src.checks.coordinator import FleetCheckCoordinator
from src.config.settings import load_app_settings
from src.domain.enums import HealthStatus
from src.storage.repository import create_dashboard_repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_and_wake_dashboard(coordinator: FleetCheckCoordinator, wake_sleeping: bool) -> None:
    dash_res = coordinator.check_dashboard_url()
    if dash_res is None:
        logger.warning(
            "STREAMLIT_APP_URL ist nicht gesetzt – Selbst-Überwachung/Wakeup des Dashboards übersprungen."
        )
        return

    logger.info(
        "Holli Dashboard: URL=%s (awake=%s, %d ms) | %s",
        dash_res.status.value,
        dash_res.is_streamlit_awake,
        dash_res.latency_ms,
        dash_res.message,
    )
    if wake_sleeping and not dash_res.is_streamlit_awake and dash_res.status in {
        HealthStatus.ASLEEP,
        HealthStatus.DEGRADED,
    }:
        ok, msg = coordinator.url_probe.wake_app(dash_res.target_url)
        logger.info("Wakeup Holli Dashboard: ok=%s | %s", ok, msg)


def run_cli_fleet_check(wake_sleeping: bool = False) -> int:
    """Prüft alle konfigurierten Apps + Dashboard-URL und weckt schlafende Apps optional auf."""
    settings = load_app_settings()
    repository = create_dashboard_repository(settings)
    coordinator = FleetCheckCoordinator(settings)

    try:
        snapshots = coordinator.check_all_apps(repository=repository)
        configured_urls = sum(1 for snap in snapshots if snap.spec.streamlit_url.strip())
        if configured_urls == 0:
            logger.warning(
                "Keine überwachten App-URLs konfiguriert! "
                "Bitte GitHub Actions Secrets (NEWS_BOT_URL, RESEARCH_BOT_URL, ALERT_BOT_URL, STREAMLIT_APP_URL) setzen."
            )

        for snap in snapshots:
            logger.info(
                "%s (%s): Gesamt=%s | URL=%s (awake=%s, %d ms) | Worker=%s | DB=%s (%d Zeilen)",
                snap.spec.title,
                snap.spec.app_id.value,
                snap.overall_status.value,
                snap.url_check.status.value,
                snap.url_check.is_streamlit_awake,
                snap.url_check.latency_ms,
                snap.workflow_check.state.value,
                snap.db_check.status.value,
                snap.db_check.total_rows,
            )

        if wake_sleeping:
            for title, ok, msg in coordinator.wake_sleeping_apps(snapshots):
                logger.info("Wakeup %s: ok=%s | %s", title, ok, msg)

        _check_and_wake_dashboard(coordinator, wake_sleeping=wake_sleeping)
        return 0
    finally:
        repository.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holli Dashboard Health-Check & Wakeup CLI")
    parser.add_argument(
        "--wake",
        action="store_true",
        help="Sendet Wakeup-Signale an schlafende Streamlit Community Cloud Apps inkl. Dashboard.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run_cli_fleet_check(wake_sleeping=args.wake))
