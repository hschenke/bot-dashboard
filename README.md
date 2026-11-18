# 🎛️ Holli Dashboard & 🧠 KI & Websearch Brain

Zentrale Monitoring-, Datenbank-, Worker- und **KI & Websearch Brain**-Kommandozentrale für deine **auf Streamlit Community Cloud gehosteten Applikationen** (`News-Bot`, `Research-Bot`, `Alert-Bot`).
Alle Konfigurationen, URLs, Repos und Datenbank-Credentials werden sicher über **Secrets** (`st.secrets` / `.streamlit/secrets.toml`) bzw. Umgebungsvariablen gesteuert.

Zusätzlich stellt dieses Repository mit [`ki_brain`](ki_brain/__init__.py) ein **eigenständiges, kollisionsfreies Python-Framework** bereit, das direkt in `research-bot`, `alert-bot` und `news-bot` eingebunden werden kann.

---

## ✨ Kernfunktionen & Ansichten

- **📡 Dashboard (Zentrale Flotten-Übersicht & Worker-Monitor)**:
  - **KPI-Header & Sofort-Check**: Flotten-Kennzahlen (Online-Apps, Awake-Apps, DBs Online) und 1-Klick-Gesamtcheck.
  - **Health & Action**: Streamlit `/_stcore/health`-Check (`🟢 Awake` / `🌙 Sleeping`), GitHub Actions Status (`Queued`, `Running`, `Success`, `Failed`), Einzelprüfung, Aufwecken und `workflow_dispatch`.
- **🗄️ Datenbank-Inspektor (`Turso Cloud` & `SQLite`)**:
  - Verbindet sich mit den 3 Bot-Datenbanken (Turso HTTPS Hrana v2 oder lokal per Read-Only-SQLite).
- **🧠 KI-View (Zentrales KI & Websearch Brain)**:
  - Vereint kostenfreie Modelle (Gemini Flash/Lite/Hydrogen, Gemma, Groq, OpenRouter `:free`) sowie Paid-Opt-In-Modelle (Deep Research Agent, Argon/Pro) und Websearch-/Grounding-/Extraktions-Tools (Yahoo, SearXNG, DuckDuckGo, Google News RSS, Wikipedia, Wikidata, Hacker News, Reddit, arXiv, Gemini Search & Maps Grounding mit 5.000 Free-Queries/Monat ab Gemini 3+, Jina Reader, Direct HTTP, Browserbase, Open-Meteo).
  - **Strikte 0%-Kosten-Priorität & Kontingent-Tracking**: Sequencing und Ausführungs-Pipeline schließen kostenpflichtige Modelle sowie erschöpfte Free-Kontingente standardmäßig aus und beziehen sie erst über explizites Request-Opt-In (`allow_paid=True`) ein.
  - **Dynamische Verwaltung**: Neue Modelle & Tools (`custom_json_api`, `custom_rss_feed`) manuell oder per **Auto-Sync** (`client.models.list()`, OpenRouter `:free`, Groq) hinzufügen, bearbeiten, aktivieren/deaktivieren oder entfernen.
  - **Testlauf & Sequenz-Optimierung**: Ermittelt per 1-Klick-Benchmark die optimale Ausführungs-Sequenz für **3 Stufen: `Fast`, `Standard` und `Deep`** (0%-Standard + Paid-Opt-In-Kette).
- **⚙️ System & Check-Historie**:
  - Konfigurations-Matrix und persistiertes Prüfprotokoll (`fleet_check_logs`).

---

## 🔌 Einbindung von `ki_brain` in andere Projekte (`Research-Bot`, `Alert-Bot`)

```bash
pip install "git+https://github.com/hschenke/bot-dashboard.git"
```

```python
from ki_brain import KIWebsearchBrain

# Nutzt optional dieselbe Turso Cloud DB (BRAIN_TURSO_URL / TURSO_URL); standardmäßig strikt 0% Kosten.
with KIWebsearchBrain.from_env() as brain:
    hits = brain.search("Aktuelle Entwicklungen KI", tier="fast")
    maps_hits = brain.search("Cafés am Alexanderplatz Berlin", tier="standard")  # Geo-Smart Maps Grounding
    answer = brain.ask("Tiefgehende Marktanalyse", tier="deep")
    paid = brain.ask("Tiefenanalyse mit Paid-Modellen", tier="deep", allow_paid=True)
```

---

## 📁 Projektstruktur

```text
bot-dashboard/
├── .github/workflows/
│   ├── ci.yaml                        # Automatisches Linting (Ruff + Bandit) & Tests (Pytest)
│   └── scheduled_healthcheck.yaml     # Cron-Healthcheck & Wakeup
├── .streamlit/
│   ├── config.toml                    # Dark-Theme & sichere Server-Defaults
│   └── secrets.toml.example           # Vorlage für Streamlit Cloud Secrets
├── config/
│   └── settings.yaml                  # Deklarative Definition der 3 überwachten Apps
├── ki_brain/                          # Eigenständiges KI & Websearch Brain Framework
│   ├── tools/                         # Key-freie Such-, Wissens-, Extraktions- & Custom-Tools
│   ├── brain.py                       # Hauptfassade (CRUD, Benchmark, Search, Ask)
│   ├── benchmark.py & optimizer.py    # Testlauf & Stufen-Optimierung (Fast / Standard / Deep)
│   ├── discovery.py                   # Auto-Sync (Gemini, Groq, OpenRouter Free, Builtins)
│   ├── llm_runner.py & execution.py   # Gemini Interactions API, Deep Agent, OpenAI-Compat
│   └── storage.py & db_executors.py   # Turso Cloud & SQLite WAL Persistenz
├── src/
│   ├── ai/brain_bridge.py             # Anbindung von AppSettings an ki_brain
│   ├── checks/                        # URL-, GitHub- & DB-Probes + FleetCheckCoordinator
│   ├── config/                        # Konstanten & AppSettings
│   ├── domain/                        # Unveränderliche Datenmodelle, Enums & Exceptions
│   ├── security/                      # SSRF-Guard, Whitelisting, HMAC-Auth & Rate-Limiter
│   ├── storage/                       # SqlExecutor (SQLite & Turso) + Dashboard-Repository
│   ├── ui/                            # Streamlit-Views (Dashboard, DB, KI Brain, System)
│   └── worker/                        # CLI-Entrypoint für Hintergrund-Checks
├── tests/                             # Unit-, Security-, Brain- & Streamlit-AppTest-Suite
├── streamlit_app.py                   # Root-Entrypoint für Streamlit Cloud
└── requirements.txt                   # Abhängigkeiten
```

---

## 🚀 Lokaler Start & Qualitätscheck

```bash
uv sync --extra dev
uv run --extra dev ruff check .
uv run --extra dev pytest -v
uv run streamlit run streamlit_app.py
```
