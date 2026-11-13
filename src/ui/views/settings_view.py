"""System-Status, Konfigurations-Matrix und persistierte Check-Historie."""

from __future__ import annotations

import streamlit as st

from src.config.settings import AppSettings
from src.storage.repository import SqlDashboardRepository
from src.ui.formatters import format_iso_timestamp


def render_settings_view(
    settings: AppSettings,
    repository: SqlDashboardRepository,
) -> None:
    """Zeigt die Konfigurations-Bereitschaft und das historische Prüfprotokoll."""
    st.header("⚙️ System-Konfiguration & Check-Historie")

    st.subheader("🔐 Konfigurations-Matrix")
    matrix_rows = [
        {
            "Komponente": "Google Gemini API (KI-View)",
            "Status": "✅ Konfiguriert" if settings.has_gemini_configured else "⚪ Heuristik-Modus (kein Key)",
            "Details": f"Primärmodell: {settings.gemini_fast_model}",
        },
        {
            "Komponente": "GitHub Actions Token",
            "Status": "✅ Konfiguriert" if settings.has_github_token else "⚪ Public Read-Only (kein Dispatch)",
            "Details": "Benötigt für 1-Klick Workflow-Start",
        },
        {
            "Komponente": "Dashboard-Speicher",
            "Status": "✅ Aktiv",
            "Details": repository.backend_name,
        },
        {
            "Komponente": "Dashboard Self-Wakeup (STREAMLIT_APP_URL)",
            "Status": "✅ Konfiguriert" if settings.streamlit_app_url.strip() else "⚪ Nicht gesetzt",
            "Details": settings.streamlit_app_url.strip()
            or "In Streamlit Secrets & GitHub Actions Secrets für Auto-Wakeup hinterlegen",
        },
    ]
    for spec in settings.managed_apps:
        db_mode = "Turso Cloud" if spec.has_turso_credentials else f"Lokal ({spec.local_sqlite_path})"
        matrix_rows.append(
            {
                "Komponente": f"{spec.icon} {spec.title}",
                "Status": "✅ Verbunden" if spec.streamlit_url.strip() else "⚠️ URL fehlt",
                "Details": f"URL: {spec.streamlit_url or '–'} | Repo: {spec.github_repo or '–'} | DB: {db_mode}",
            }
        )
    st.dataframe(matrix_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📜 Persistierte Check-Historie")
    logs = repository.list_recent_logs(limit=40)
    if not logs:
        st.info("Noch keine Prüfprotokolle gespeichert.")
        return

    log_rows = [
        {
            "Zeitpunkt": format_iso_timestamp(entry.created_at),
            "App": entry.app_id,
            "Gesamt": entry.overall_status,
            "URL": f"{entry.url_status} ({entry.url_latency_ms} ms)",
            "Worker": entry.workflow_state,
            "DB": f"{entry.db_status} ({entry.db_total_rows} Zeilen)",
            "Zusammenfassung": entry.summary_text,
        }
        for entry in logs
    ]
    st.dataframe(log_rows, use_container_width=True, hide_index=True)
