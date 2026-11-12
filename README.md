# 🎛️ Holli Dashboard (Streamlit Fleet Control Center)

Zentrale Monitoring-, Datenbank-, Worker- und KI-Kommandozentrale für deine **auf Streamlit Community Cloud gehosteten Applikationen** (z. B. News-Bot, Research-Bot, Alert-Bot).
Alle Konfigurationen, URLs, Repos und Datenbank-Credentials werden sicher und flexibel über **Secrets** (`st.secrets` / `.streamlit/secrets.toml`) bzw. Umgebungsvariablen gesteuert.

Konzipiert als eigenständige **Streamlit Web-App** auf derselben kostenlosen **0-€-Cloud-Architektur** (**Streamlit Community Cloud + GitHub Actions + Turso LibSQL / SQLite + Google Gemini**).

---

## ✨ Kernfunktionen & Ansichten

- **📡 Flotten-Übersicht**:
  - Alle 3 Bots nebeneinander mit Live-Gesamtstatus, Antwortzeit (`ms`), Worker-Status, Datenbank-Zeilenbestand und 1-Klick-Startlinks.
- **🌐 Running & URLs**:
  - **Streamlit Cloud Health-Check**: Prüft sowohl die Haupt-URL als auch den offiziellen `/_stcore/health`-Endpunkt und erkennt, ob eine App wach (`🟢 Online`) oder im Ruhezustand (`🌙 Ruhezustand`) ist.
  - **GitHub Actions Worker-Monitor & Trigger**: Zeigt den Echtzeit-Status von `daily_digest.yaml`, `run_research.yaml` und `check_alerts.yaml` (`running`, `queued`, `success`, `failure`) und erlaubt den direkten **1-Klick-Start (`workflow_dispatch`)** aus dem Dashboard.
- **🗄️ Datenbank-Inspektor (`Turso Cloud` & `SQLite`)**:
  - Verbindet sich mit den 3 Bot-Datenbanken (über Turso HTTPS Hrana v2 oder lokal per Read-Only-SQLite).
  - Zeigt Tabellen-Bestände (`articles`, `briefings`, `research_jobs`, `research_reports`, `alert_rules`, `alert_events`, `alert_check_logs`), aktive Aufträge/Wächter, Fehlerzähler und die jüngsten Einträge.
- **🤖 KI-View (Google Gemini Fleet Intelligence)**:
  - Erstellt auf Knopfdruck eine strukturierte Gesamtdiagnose (`FleetAIAnalysis` mit `Fleet Health Score 0–100`, Einzelbewertungen, Anomalie-Warnungen & Handlungsempfehlungen) über die moderne `google-genai` Interactions API (inkl. deterministischem Heuristik-Fallback ohne API-Key).
  - Interaktiver **KI-Operations-Assistent (Q&A)** für Freitext-Fragen zum Live-Zustand der 3 Apps.
- **⚙️ System & Check-Historie**:
  - Konfigurations-Matrix und persistiertes Prüfprotokoll (`fleet_check_logs`).

---

## 📁 Projektstruktur

```text
bot-dashboard/
├── .github/workflows/
│   ├── ci.yaml                        # Automatisches Linting (Ruff + Bandit) & Tests (Pytest)
│   └── scheduled_healthcheck.yaml     # Optionaler 30-Min.-Cron & manueller Flotten-Check
├── .streamlit/
│   ├── config.toml                    # Dark-Theme, XSRF-Schutz & sichere Server-Defaults
│   └── secrets.toml.example           # Vorlage für Streamlit Cloud Secrets
├── config/
│   └── settings.yaml                  # Deklarative Definition der 3 überwachten Apps & KI-Prompts
├── src/
│   ├── ai/                            # Gemini Interactions API Gateway & Heuristik-Diagnose
│   ├── checks/                        # SafeUrlProbe, GitHubWorkflowProbe, BotDatabaseInspector, Coordinator
│   ├── config/                        # Konstanten & AppSettings (Secrets, ENV & YAML-Defaults)
│   ├── domain/                        # Enums, Exception-Hierarchie, Dataclasses & Pydantic-Schemata
│   ├── security/                      # SSRF-Guard, Whitelist-Validierung, HMAC-Auth & Rate-Limiter
│   ├── storage/                       # SqlExecutor (SQLite & Turso) + SqlDashboardRepository
│   ├── ui/                            # Schlanke Streamlit-Views (Overview, Running/URLs, DB, KI, Settings)
│   └── worker/                        # CLI-Entrypoint für Hintergrund-Checks
├── tests/                             # Umfassende Unit- & Integrationstests
├── streamlit_app.py                   # Root-Entrypoint für Streamlit Cloud
└── requirements.txt                   # Abhängigkeiten
```

---

## 🚀 Lokaler Start

```powershell
# 1. Virtuelle Umgebung erstellen & aktivieren
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Abhängigkeiten installieren
python -m pip install -r requirements.txt

# 3. Optional: .env anlegen (lokal erkennt das Dashboard automatisch auch ../alert-bot/.env etc.)
Copy-Item .env.example .env

# 4. Web-App starten
streamlit run streamlit_app.py
```

### Qualitätssicherung (Linter & Tests)
```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -v
```

---

## ☁️ Deployment auf Streamlit Community Cloud

1. Auf [share.streamlit.io](https://share.streamlit.io) neue App aus `hschenke/bot-dashboard` (`main`, `streamlit_app.py`) erstellen.
2. Unter **App Settings -> Secrets** die Werte aus `.streamlit/secrets.toml.example` eintragen.
