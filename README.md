# 🎛️ Holli Dashboard & 🧠 AI & Websearch Brain

Zentrale Monitoring-, Datenbank-, Worker- und **AI & Websearch Brain**-Kommandozentrale für deine **auf Streamlit Community Cloud gehosteten Applikationen** (`News-Bot`, `Research-Bot`, `Alert-Bot`).
Alle Konfigurationen, URLs, Repos und Datenbank-Credentials werden sicher über **Secrets** (`st.secrets` / `.streamlit/secrets.toml`) bzw. Umgebungsvariablen gesteuert.

Zusätzlich stellt dieses Repository mit [`ai_brain`](ai_brain/__init__.py) ein **eigenständiges, kollisionsfreies Python-Framework** bereit, das direkt in `research-bot`, `alert-bot` und `news-bot` eingebunden werden kann.

---

## ✨ Kernfunktionen & Ansichten

- **📡 Dashboard (Zentrale Flotten-Übersicht & Worker-Monitor)**:
  - **Mobile-kompakte KPIs & Action-Buttons**: Flotten-Kennzahlen (`Apps Online`, `Apps Awake`, `DBs Online`) sowie `⚡ Check All` & `☀️ Wake All` direkt darunter nebeneinander (inkl. Fail-Safe Task-Overlay bis alle Apps wach sind).
  - **Health & Action**: Streamlit `/_stcore/health`-Check (`🟢 Awake` / `🌙 Sleeping`), GitHub Actions Status (`Queued`, `Running`, `Success`, `Failed`), Einzelprüfung, Aufwecken und `workflow_dispatch`.
- **🗄️ Datenbank-Inspektor (`Turso Cloud` & `SQLite`)**:
  - Verbindet sich mit den 3 Bot-Datenbanken (Turso HTTPS Hrana v2 oder lokal per Read-Only-SQLite).
- **🧠 AI-Brain (Zentrales AI & Websearch Brain)**:
  - **Einheitlicher Ablauf (`🚀 Sync & Optimize`)**: Kombiniert Discover, Benchmark & Optimize, Unblock Tools und Prune Failing mit wählbaren Optionen.
  - **6 strukturierte Tabs (`Sequencing | AI | Tools | Vault | Check | Logs`)**: Free- & Paid-Tier API-Keys sowie **1 gemeinsamer kompakter Flotten-Key (`AI_BRAIN_KEY`)** für alle 3 Apps werden zentral & Write-Only im **Vault** verwaltet; der Tab **📜 Logs** protokolliert jeden Zugriff (Wann, Wer, Wo, Aktion, Free/Paid, Latenz & Erfolg).
  - **Strikte 0%-Kosten-Priorität & Free-/Paid-Key-Trennung**: Standardanfragen (`allow_paid=False`) nutzen ausschließlich den Free-Tier-Key (`GEMINI_FREE_API_KEY` bzw. Fallback `GEMINI_API_KEY`); der Paid-Tier-Key (`GEMINI_PAID_API_KEY`) und Paid-Only-Modelle werden erst bei explizitem `allow_paid=True` aktiv.

---

## 🔌 1-Secret Plug-&-Play Einbindung von `ai_brain` (`News-Bot`, `Research-Bot`, `Alert-Bot`)

In allen 3 Bot-Repos genügt **1 Eintrag** in `requirements*.txt` (zieht automatisch immer die aktuelle `main`-Version ohne Streamlit-Ballast) sowie **derselbe kompakte Flotten-Key (`AI_BRAIN_KEY`)** aus dem Dashboard-Tab **🔐 Vault**:

```bash
pip install "git+https://github.com/hschenke/bot-dashboard.git@main"
```

```python
import ai_brain
from ai_brain import AppBrainPolicy, BrainTier

# Liest AI_BRAIN_KEY automatisch aus ENV, st.secrets & .streamlit/secrets.toml (0 Kollision mit App-DB).
# Fremde Aufrufe oder gesperrte/rotierte Keys werden strikt mit BrainAuthError abgelehnt.
if ai_brain.is_configured():
    with ai_brain.connect() as brain:
        # Persistente App-Optionen (Fast / Standard / Deep, Maps & Free/Paid) in Turso speichern/laden:
        policy = brain.get_app_policy()
        brain.set_app_policy(AppBrainPolicy(llm_tier=BrainTier.STANDARD, search_tier=BrainTier.FAST, allow_paid=False))

        # tier="auto" nutzt automatisch die in der App-Option gespeicherten Stufen (UI & Worker synchron):
        hits = brain.search("Aktuelle Entwicklungen KI", tier="auto")
        answer = brain.ask("Tiefgehende Marktanalyse", tier="auto")
        # Expliziter Tier-Override weiterhin jederzeit möglich:
        deep_paid = brain.ask("Tiefenanalyse mit Paid-Modellen", tier="deep", allow_paid=True)
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
├── ai_brain/                          # Eigenständiges AI & Websearch Brain Framework
│   ├── tools/                         # Key-freie Such-, Wissens-, Extraktions- & Custom-Tools
│   ├── brain.py                       # Hauptfassade (CRUD, Benchmark, Search, Ask)
│   ├── benchmark.py & optimizer.py    # Testlauf & Stufen-Optimierung (Fast / Standard / Deep)
│   ├── discovery.py                   # Auto-Sync (Gemini, Groq, OpenRouter Free, Builtins)
│   ├── llm_runner.py & execution.py   # Gemini Interactions API, Deep Agent, OpenAI-Compat
│   └── storage.py & db_executors.py   # Turso Cloud & SQLite WAL Persistenz
├── src/
│   ├── ai/brain_bridge.py             # Anbindung von AppSettings an ai_brain
│   ├── checks/                        # URL-, GitHub- & DB-Probes + FleetCheckCoordinator
│   ├── config/                        # Konstanten & AppSettings
│   ├── domain/                        # Unveränderliche Datenmodelle, Enums & Exceptions
│   ├── security/                      # SSRF-Guard, Whitelisting, HMAC-Auth & Rate-Limiter
│   ├── storage/                       # SqlExecutor (SQLite & Turso) + Dashboard-Repository
│   ├── ui/                            # Streamlit-Views (Dashboard, DB, AI-Brain)
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
