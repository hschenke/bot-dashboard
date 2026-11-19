# 🎛️ Holli Dashboard & 🧠 KI & Websearch Brain

Zentrale Monitoring-, Datenbank-, Worker- und **KI & Websearch Brain**-Kommandozentrale für deine **auf Streamlit Community Cloud gehosteten Applikationen** (`News-Bot`, `Research-Bot`, `Alert-Bot`).
Alle Konfigurationen, URLs, Repos und Datenbank-Credentials werden sicher über **Secrets** (`st.secrets` / `.streamlit/secrets.toml`) bzw. Umgebungsvariablen gesteuert.

Zusätzlich stellt dieses Repository mit [`ki_brain`](ki_brain/__init__.py) ein **eigenständiges, kollisionsfreies Python-Framework** bereit, das direkt in `research-bot`, `alert-bot` und `news-bot` eingebunden werden kann.

---

## ✨ Kernfunktionen & Ansichten

- **📡 Dashboard (Zentrale Flotten-Übersicht & Worker-Monitor)**:
  - **Mobile-kompakte KPIs & Action-Buttons**: Flotten-Kennzahlen (`Apps Online`, `Apps Awake`, `DBs Online`) sowie `⚡ Check All` & `☀️ Wake All` direkt darunter nebeneinander (inkl. Fail-Safe Task-Overlay bis alle Apps wach sind).
  - **Health & Action**: Streamlit `/_stcore/health`-Check (`🟢 Awake` / `🌙 Sleeping`), GitHub Actions Status (`Queued`, `Running`, `Success`, `Failed`), Einzelprüfung, Aufwecken und `workflow_dispatch`.
- **🗄️ Datenbank-Inspektor (`Turso Cloud` & `SQLite`)**:
  - Verbindet sich mit den 3 Bot-Datenbanken (Turso HTTPS Hrana v2 oder lokal per Read-Only-SQLite).
- **🧠 KI-Brain (Zentrales KI & Websearch Brain)**:
  - **Einheitlicher Ablauf (`🚀 Sync & Optimize`)**: Kombiniert Discover, Benchmark & Optimize, Unblock Tools und Prune Failing mit wählbaren Optionen.
  - **5 strukturierte Tabs (`Sequencing | KI | Tools | Vault | Check`)**: Free- & Paid-Tier API-Keys sowie die **3-App-Zugriffskontrolle** (`news_bot`, `research_bot`, `alert_bot`) werden zentral & Write-Only im **Vault** verwaltet.
  - **Strikte 0%-Kosten-Priorität & Free-/Paid-Key-Trennung**: Standardanfragen (`allow_paid=False`) nutzen ausschließlich den Free-Tier-Key (`GEMINI_FREE_API_KEY` bzw. Fallback `GEMINI_API_KEY`); der Paid-Tier-Key (`GEMINI_PAID_API_KEY`) und Paid-Only-Modelle werden erst bei explizitem `allow_paid=True` aktiv.

---

## 🔌 Einbindung von `ki_brain` in deine 3 Apps (`News-Bot`, `Research-Bot`, `Alert-Bot`)

```bash
pip install "git+https://github.com/hschenke/bot-dashboard.git"
```

```python
from ki_brain import KIWebsearchBrain

# Exklusiver Zugriff für news_bot, research_bot & alert_bot via BRAIN_APP_ID + BRAIN_APP_TOKEN (im Vault generierbar).
# Fremde Apps oder fehlende/widerrufene Tokens werden strikt mit BrainAuthError abgelehnt.
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
