# Projekt-Richtlinien für Bot-Dashboard

## Automatischer Start-Ablauf
- **Automatisches `git pull`**: Führe zu Beginn jeder neuen Konversation in diesem Projekt zuerst `git pull` aus, um das lokale Repository auf den neuesten Stand zu bringen, und gib kurz Rückmeldung.

## Automatischer Abschluss-Ablauf
- **Qualitätscheck vor jedem Commit**: Führe vor jedem Commit `.venv\Scripts\ruff.exe check .` und `.venv\Scripts\pytest.exe` aus.
- **Automatisches Committen & `git push`**: Committe abgeschlossene Features/Fixes sauber und pushe sie direkt nach `origin main`, damit GitHub Actions und Streamlit Community Cloud sofort aktuell sind.
- **Proaktive Versions-Tag Empfehlung (Semantic Versioning)**: Erhöhe bei neuen Meilensteinen die Version in `src/__version__.py` und setze/empfehle passende SemVer-Tags (`v0.X.0` für Features, `v0.X.Y` für Fixes).

## Architektur & Clean Code Standards (Strikte Pflicht)
- **Kompakte Module & Funktionen**: Dateien übersichtlich halten (Ziel: < 200 Zeilen pro Modul), Funktionen kompakt halten (Ziel: max. 20–30 Zeilen, SRP, Guard Clauses statt tiefer Verschachtelung).
- **Strikte Schichten-Trennung**:
  1. `src/domain`: Unveränderliche Datenmodelle (`@dataclass(frozen=True)`), Enums, Pydantic-KI-Schemata und typisierte Exceptions.
  2. `src/config`: Konstanten ohne Magic Values, YAML-Defaults und `AppSettings` (`repr=False` für alle Secrets).
  3. `src/security`: SSRF-Guard (inkl. DNS-Rebinding- & Redirect-Hop-Prüfung), Input-Whitelist-Validierung, HMAC-Auth & Brute-Force-Schutz.
  4. `src/storage`: `SqlExecutor`-Abstraktion + einheitliches `SqlDashboardRepository` für lokales SQLite (WAL) & Turso Cloud (Hrana v2 HTTPS).
  5. `src/checks`: `SafeUrlProbe` (HTTP & Streamlit `/_stcore/health`), `GitHubWorkflowProbe` (Workflow-Status & `workflow_dispatch`), `BotDatabaseInspector` (Multi-DB-Inspektion) und `FleetCheckCoordinator`.
  6. `src/ai`: `DashboardAIGateway` auf Basis der modernen `google-genai` Interactions API (`>= 2.30.1`) mit Structured Output & deterministischem Heuristik-Fallback.
  7. `src/worker`: CLI-Healthcheck-Runner für GitHub Actions (`scheduled_healthcheck.yaml`).
  8. `src/ui`: Reine Streamlit-Präsentationsschicht (`overview_view`, `urls_running_view`, `db_view`, `ki_view`, `settings_view`).

## Security Baseline (`security-standard`)
- **Zero Hardcoded Secrets**: Alle Keys/Passwörter ausschließlich über Umgebungsvariablen / `st.secrets`.
- **Parameterized SQL & Table Whitelisting**: Ausnahmslos `?`-Platzhalter für Parameter und Whitelist-Prüfung für Tabellennamen.
- **SSRF & DoS Prevention**: Strikte IP-/Schema-Prüfung vor jedem HTTP-Abruf, Redirect-Validierung, explizite Timeouts und Payload-Größenlimits.
