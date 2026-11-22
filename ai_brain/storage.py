"""Persistenz-Repository für Modelle, Tools, Tier-Sequenzen und Testläufe."""

from __future__ import annotations

import datetime
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_brain.catalog import build_default_models, build_default_sequences, build_default_tools
from ai_brain.db_executors import BrainSqlExecutor, SqliteBrainExecutor, TursoBrainExecutor
from ai_brain.enums import BrainTier
from ai_brain.exceptions import BrainStorageError
from ai_brain.models import (
    AIModelSpec,
    AppBrainPolicy,
    BenchmarkReport,
    BrainAccessLogEntry,
    TierSequenceConfig,
    ToolSpec,
    default_app_policy,
)
from ai_brain.quotas import USAGE_QUOTAS_META_KEY, parse_usage_quotas
from ai_brain.security import validate_identifier

_INITIALIZED_BACKENDS: set[str] = set()
_INITIALIZED_LOCK = threading.Lock()
MAX_ACCESS_LOG_ENTRIES: int = 500
APP_POLICIES_META_KEY: str = "app_policies"


@dataclass(frozen=True)
class BrainDashboardState:
    """Gebündelter Lese-Snapshot für schnelle UI-Darstellung in einer einzigen DB-Batch-Abfrage."""

    models: list[AIModelSpec]
    tools: list[ToolSpec]
    sequences: dict[BrainTier, TierSequenceConfig]
    latest_benchmark: BenchmarkReport | None
    secret_timestamps: dict[str, str]
    excluded_model_ids: tuple[str, ...] = ()
    excluded_model_reasons: dict[str, str] = field(default_factory=dict)
    usage_quotas: dict[str, dict[str, dict[str, Any]]] = field(default_factory=lambda: {"tools": {}, "models": {}})
    fleet_client: dict[str, Any] | None = None
    access_logs: list[BrainAccessLogEntry] = field(default_factory=list)


_SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS brain_models (
        id TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_tools (
        id TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_sequences (
        tier TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_benchmark_runs (
        run_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_secrets (
        env_name TEXT PRIMARY KEY,
        encrypted_value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_app_clients (
        app_id TEXT PRIMARY KEY,
        token_hash TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        last_used_at TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS brain_access_logs (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        caller_app TEXT NOT NULL,
        environment TEXT NOT NULL,
        action TEXT NOT NULL,
        tier TEXT NOT NULL DEFAULT '',
        budget_mode TEXT NOT NULL DEFAULT 'free',
        status TEXT NOT NULL,
        latency_ms INTEGER NOT NULL DEFAULT 0,
        model_used TEXT NOT NULL DEFAULT '',
        tools_used TEXT NOT NULL DEFAULT '',
        result_count INTEGER NOT NULL DEFAULT 0,
        summary TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_brain_access_logs_created_at
    ON brain_access_logs (created_at DESC)
    """,
)


def _row_to_access_log(row: dict[str, Any]) -> BrainAccessLogEntry:
    raw_tools = str(row.get("tools_used") or "").strip()
    tools = [t.strip() for t in raw_tools.split(",") if t.strip()] if raw_tools else []
    return BrainAccessLogEntry(
        id=str(row.get("id") or ""),
        created_at=str(row.get("created_at") or ""),
        caller_app=str(row.get("caller_app") or ""),
        environment=str(row.get("environment") or ""),
        action=str(row.get("action") or ""),
        tier=str(row.get("tier") or ""),
        budget_mode=str(row.get("budget_mode") or "free"),
        status=str(row.get("status") or "Success"),
        latency_ms=int(row.get("latency_ms") or 0),
        model_used=str(row.get("model_used") or ""),
        tools_used=tools,
        result_count=int(row.get("result_count") or 0),
        summary=str(row.get("summary") or ""),
    )


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _parse_excluded_payload(raw: str) -> tuple[set[str], set[str]]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return set(), set()
    models = {str(x).strip().lower() for x in data.get("models", []) if str(x).strip()}
    tools = {str(x).strip().lower() for x in data.get("tools", []) if str(x).strip()}
    return models, tools


def _parse_reasons_payload(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}, {}
    m_raw = data.get("models") if isinstance(data.get("models"), dict) else {}
    t_raw = data.get("tools") if isinstance(data.get("tools"), dict) else {}
    m_reasons = {str(k).strip().lower(): str(v).strip() for k, v in m_raw.items() if str(k).strip() and str(v).strip()}
    t_reasons = {str(k).strip().lower(): str(v).strip() for k, v in t_raw.items() if str(k).strip() and str(v).strip()}
    return m_reasons, t_reasons


class BrainRepository:
    """Einheitliches Repository (SQLite & Turso) für den dynamischen Brain-Zustand."""

    def __init__(self, executor: BrainSqlExecutor, seed_defaults: bool = True) -> None:
        self._executor = executor
        self._cache_key = (
            f"turso:{executor._pipeline_url}"  # noqa: SLF001
            if isinstance(executor, TursoBrainExecutor)
            else ""
        )
        already_ready = False
        if self._cache_key:
            with _INITIALIZED_LOCK:
                already_ready = self._cache_key in _INITIALIZED_BACKENDS
        if not already_ready:
            self._init_schema()
            if seed_defaults:
                self.ensure_seeded()
            if self._cache_key:
                with _INITIALIZED_LOCK:
                    _INITIALIZED_BACKENDS.add(self._cache_key)

    @property
    def backend_name(self) -> str:
        return self._executor.backend_name

    def _init_schema(self) -> None:
        self._executor.execute_batch([(sql, ()) for sql in _SCHEMA_SQL])

    def get_meta(self, key: str, default: str = "") -> str:
        res = self._executor.execute("SELECT value FROM brain_meta WHERE key = ?", (key,))
        return str(res.rows[0]["value"]) if res.rows else default

    def set_meta(self, key: str, value: str) -> None:
        self._executor.execute(
            """
            INSERT INTO brain_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def ensure_seeded(self) -> None:
        if self.get_meta("seeded_v1") == "1":
            return
        now = _utc_now_iso()
        stmts: list[tuple[str, tuple[str, ...]]] = [
            (
                "INSERT OR IGNORE INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)",
                (model.id, model.model_dump_json(), now),
            )
            for model in build_default_models()
        ]
        stmts.extend(
            (
                "INSERT OR IGNORE INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)",
                (tool.id, tool.model_dump_json(), now),
            )
            for tool in build_default_tools()
        )
        stmts.extend(
            (
                "INSERT OR IGNORE INTO brain_sequences (tier, payload_json, updated_at) VALUES (?, ?, ?)",
                (tier.value, seq.model_copy(update={"updated_at": seq.updated_at or now}).model_dump_json(), now),
            )
            for tier, seq in build_default_sequences().items()
        )
        stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("seeded_v1", "1")))
        self._executor.execute_batch(stmts)

    def list_excluded_ids(self) -> tuple[set[str], set[str]]:
        """Liefert persistent ausgeschlossene (gelöschte/defekte) Modell- und Tool-IDs."""
        return _parse_excluded_payload(self.get_meta("excluded_ids", "{}"))

    def list_excluded_reasons(self) -> tuple[dict[str, str], dict[str, str]]:
        """Liefert optionale Ausschlussgründe (`manual`, `quota_limited`, `error`, `preview`)."""
        return _parse_reasons_payload(self.get_meta("excluded_reasons", "{}"))

    def _save_excluded_ids(self, models: set[str], tools: set[str]) -> None:
        self.set_meta(
            "excluded_ids",
            json.dumps({"models": sorted(models), "tools": sorted(tools)}),
        )

    def _save_excluded_reasons(self, m_reasons: dict[str, str], t_reasons: dict[str, str]) -> None:
        self.set_meta(
            "excluded_reasons",
            json.dumps({"models": dict(sorted(m_reasons.items())), "tools": dict(sorted(t_reasons.items()))}),
        )

    def add_excluded_ids(
        self,
        *,
        model_ids: list[str] | set[str] | tuple[str, ...] = (),
        tool_ids: list[str] | set[str] | tuple[str, ...] = (),
        reason: str = "",
        model_reasons: dict[str, str] | None = None,
        tool_reasons: dict[str, str] | None = None,
    ) -> None:
        """Markiert entfernte oder nicht funktionierende IDs dauerhaft inkl. Ausschlussgrund."""
        clean_m = {str(x).strip().lower() for x in model_ids if str(x).strip()}
        clean_t = {str(x).strip().lower() for x in tool_ids if str(x).strip()}
        if not clean_m and not clean_t:
            return
        models, tools = self.list_excluded_ids()
        updated_m = models | clean_m
        updated_t = tools | clean_t
        if updated_m != models or updated_t != tools:
            self._save_excluded_ids(updated_m, updated_t)

        m_res_map, t_res_map = self.list_excluded_reasons()
        orig_m, orig_t = dict(m_res_map), dict(t_res_map)
        clean_default = reason.strip()
        for mid in clean_m:
            r = (model_reasons or {}).get(mid, clean_default).strip()
            if r:
                m_res_map[mid] = r
        for tid in clean_t:
            r = (tool_reasons or {}).get(tid, clean_default).strip()
            if r:
                t_res_map[tid] = r
        if m_res_map != orig_m or t_res_map != orig_t:
            self._save_excluded_reasons(m_res_map, t_res_map)

    def remove_excluded_id(self, *, model_id: str = "", tool_id: str = "") -> None:
        """Hebt den Auto-Sync-Ausschluss bei expliziter Wiederaufnahme wieder auf."""
        models, tools = self.list_excluded_ids()
        m_reasons, t_reasons = self.list_excluded_reasons()
        ids_changed = False
        reasons_changed = False
        clean_m = model_id.strip().lower()
        clean_t = tool_id.strip().lower()
        if clean_m and clean_m in models:
            models.discard(clean_m)
            ids_changed = True
        if clean_t and clean_t in tools:
            tools.discard(clean_t)
            ids_changed = True
        if clean_m and m_reasons.pop(clean_m, None) is not None:
            reasons_changed = True
        if clean_t and t_reasons.pop(clean_t, None) is not None:
            reasons_changed = True
        if ids_changed:
            self._save_excluded_ids(models, tools)
        if reasons_changed:
            self._save_excluded_reasons(m_reasons, t_reasons)

    def reset_to_defaults(self) -> None:
        self._executor.execute_batch(
            [
                ("DELETE FROM brain_models", ()),
                ("DELETE FROM brain_tools", ()),
                ("DELETE FROM brain_sequences", ()),
                (
                    "DELETE FROM brain_meta WHERE key IN ('seeded_v1', 'excluded_ids', 'excluded_reasons')",
                    (),
                ),
            ]
        )
        self.ensure_seeded()

    def list_models(self, enabled_only: bool = False) -> list[AIModelSpec]:
        res = self._executor.execute("SELECT payload_json FROM brain_models ORDER BY id ASC")
        items = [AIModelSpec.model_validate_json(str(r["payload_json"])) for r in res.rows]
        return [m for m in items if m.enabled] if enabled_only else items

    def get_model(self, model_id: str) -> AIModelSpec | None:
        clean_id = validate_identifier(model_id, "Modell-ID")
        res = self._executor.execute("SELECT payload_json FROM brain_models WHERE id = ?", (clean_id,))
        return AIModelSpec.model_validate_json(str(res.rows[0]["payload_json"])) if res.rows else None

    def upsert_model(self, spec: AIModelSpec) -> AIModelSpec:
        clean_id = validate_identifier(spec.id, "Modell-ID")
        normalized = spec.model_copy(update={"id": clean_id})
        self._executor.execute(
            """
            INSERT INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
            """,
            (clean_id, normalized.model_dump_json(), _utc_now_iso()),
        )
        return normalized

    def upsert_models_batch(self, specs: list[AIModelSpec]) -> None:
        if not specs:
            return
        now = _utc_now_iso()
        sql = """
            INSERT INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
        """
        stmts = []
        for spec in specs:
            clean_id = validate_identifier(spec.id, "Modell-ID")
            normalized = spec.model_copy(update={"id": clean_id})
            stmts.append((sql, (clean_id, normalized.model_dump_json(), now)))
        self._executor.execute_batch(stmts)

    def delete_model(self, model_id: str) -> bool:
        clean_id = validate_identifier(model_id, "Modell-ID")
        res = self._executor.execute("DELETE FROM brain_models WHERE id = ?", (clean_id,))
        return res.affected_rows > 0

    def list_tools(self, enabled_only: bool = False) -> list[ToolSpec]:
        res = self._executor.execute("SELECT payload_json FROM brain_tools ORDER BY id ASC")
        items = [ToolSpec.model_validate_json(str(r["payload_json"])) for r in res.rows]
        return [t for t in items if t.enabled] if enabled_only else items

    def get_tool(self, tool_id: str) -> ToolSpec | None:
        clean_id = validate_identifier(tool_id, "Tool-ID")
        res = self._executor.execute("SELECT payload_json FROM brain_tools WHERE id = ?", (clean_id,))
        return ToolSpec.model_validate_json(str(res.rows[0]["payload_json"])) if res.rows else None

    def upsert_tool(self, spec: ToolSpec) -> ToolSpec:
        clean_id = validate_identifier(spec.id, "Tool-ID")
        normalized = spec.model_copy(update={"id": clean_id})
        self._executor.execute(
            """
            INSERT INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
            """,
            (clean_id, normalized.model_dump_json(), _utc_now_iso()),
        )
        return normalized

    def upsert_tools_batch(self, specs: list[ToolSpec]) -> None:
        if not specs:
            return
        now = _utc_now_iso()
        sql = """
            INSERT INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
        """
        stmts = []
        for spec in specs:
            clean_id = validate_identifier(spec.id, "Tool-ID")
            normalized = spec.model_copy(update={"id": clean_id})
            stmts.append((sql, (clean_id, normalized.model_dump_json(), now)))
        self._executor.execute_batch(stmts)

    def delete_tool(self, tool_id: str) -> bool:
        clean_id = validate_identifier(tool_id, "Tool-ID")
        res = self._executor.execute("DELETE FROM brain_tools WHERE id = ?", (clean_id,))
        return res.affected_rows > 0

    def list_sequences(self) -> dict[BrainTier, TierSequenceConfig]:
        res = self._executor.execute("SELECT tier, payload_json FROM brain_sequences")
        defaults = build_default_sequences()
        for row in res.rows:
            tier = BrainTier(str(row["tier"]))
            defaults[tier] = TierSequenceConfig.model_validate_json(str(row["payload_json"]))
        return defaults

    def get_sequence(self, tier: BrainTier | str) -> TierSequenceConfig:
        target = BrainTier(tier)
        res = self._executor.execute("SELECT payload_json FROM brain_sequences WHERE tier = ?", (target.value,))
        if res.rows:
            return TierSequenceConfig.model_validate_json(str(res.rows[0]["payload_json"]))
        return build_default_sequences()[target]

    def upsert_sequence(self, config: TierSequenceConfig) -> TierSequenceConfig:
        now = _utc_now_iso()
        updated = config.model_copy(update={"updated_at": now})
        self._executor.execute(
            """
            INSERT INTO brain_sequences (tier, payload_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(tier) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
            """,
            (updated.tier.value, updated.model_dump_json(), now),
        )
        return updated

    def save_benchmark_run(self, report: BenchmarkReport) -> None:
        self._executor.execute(
            """
            INSERT INTO brain_benchmark_runs (run_id, started_at, payload_json) VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET started_at = excluded.started_at, payload_json = excluded.payload_json
            """,
            (report.run_id, report.started_at, report.model_dump_json()),
        )

    def get_latest_benchmark(self) -> BenchmarkReport | None:
        runs = self.list_benchmark_runs(limit=1)
        return runs[0] if runs else None

    def list_benchmark_runs(self, limit: int = 10) -> list[BenchmarkReport]:
        safe_limit = max(1, min(int(limit), 50))
        res = self._executor.execute(
            "SELECT payload_json FROM brain_benchmark_runs ORDER BY started_at DESC LIMIT ?",
            (safe_limit,),
        )
        return [BenchmarkReport.model_validate_json(str(r["payload_json"])) for r in res.rows]

    def upsert_encrypted_secret(self, env_name: str, encrypted_value: str) -> str:
        from ai_brain.secrets_vault import validate_secret_env_name

        clean_name = validate_secret_env_name(env_name)
        now = _utc_now_iso()
        self._executor.execute(
            """
            INSERT INTO brain_secrets (env_name, encrypted_value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(env_name) DO UPDATE SET
                encrypted_value = excluded.encrypted_value,
                updated_at = excluded.updated_at
            """,
            (clean_name, encrypted_value, now),
        )
        return now

    def delete_encrypted_secret(self, env_name: str) -> bool:
        from ai_brain.secrets_vault import validate_secret_env_name

        clean_name = validate_secret_env_name(env_name)
        res = self._executor.execute("DELETE FROM brain_secrets WHERE env_name = ?", (clean_name,))
        return res.affected_rows > 0

    def list_secret_timestamps(self) -> dict[str, str]:
        res = self._executor.execute("SELECT env_name, updated_at FROM brain_secrets ORDER BY env_name ASC")
        return {str(r["env_name"]): str(r["updated_at"]) for r in res.rows}

    def load_decrypted_secrets(self, passphrase: str = "") -> dict[str, str]:
        from ai_brain.secrets_vault import decrypt_secret_value

        res = self._executor.execute("SELECT env_name, encrypted_value FROM brain_secrets")
        decrypted: dict[str, str] = {}
        for row in res.rows:
            val = decrypt_secret_value(str(row["encrypted_value"]), passphrase=passphrase)
            if val:
                decrypted[str(row["env_name"])] = val
        return decrypted

    def get_turso_connection_credentials(self) -> tuple[str, str]:
        """Liefert `(turso_url, turso_token)` falls Turso Cloud aktiv ist, sonst `('', '')`."""
        getter = getattr(self._executor, "_connection_credentials", None)
        return getter() if callable(getter) else ("", "")

    def upsert_fleet_client(self, token_hash: str, *, enabled: bool = True) -> str:
        from ai_brain.access import FLEET_CLIENT_ID

        now = _utc_now_iso()
        self._executor.execute(
            """
            INSERT INTO brain_app_clients (app_id, token_hash, enabled, created_at, last_used_at)
            VALUES (?, ?, ?, ?, '')
            ON CONFLICT(app_id) DO UPDATE SET
                token_hash = excluded.token_hash,
                enabled = excluded.enabled,
                created_at = excluded.created_at
            """,
            (FLEET_CLIENT_ID, token_hash, 1 if enabled else 0, now),
        )
        return now

    def set_fleet_client_enabled(self, enabled: bool) -> bool:
        from ai_brain.access import FLEET_CLIENT_ID

        res = self._executor.execute(
            "UPDATE brain_app_clients SET enabled = ? WHERE app_id = ?",
            (1 if enabled else 0, FLEET_CLIENT_ID),
        )
        return res.affected_rows > 0

    def revoke_fleet_client(self) -> bool:
        from ai_brain.access import FLEET_CLIENT_ID

        res = self._executor.execute("DELETE FROM brain_app_clients WHERE app_id = ?", (FLEET_CLIENT_ID,))
        return res.affected_rows > 0

    def get_fleet_client_record(self) -> dict[str, Any] | None:
        from ai_brain.access import FLEET_CLIENT_ID

        res = self._executor.execute(
            "SELECT app_id, token_hash, enabled, created_at, last_used_at FROM brain_app_clients WHERE app_id = ?",
            (FLEET_CLIENT_ID,),
        )
        if not res.rows:
            return None
        r = res.rows[0]
        return {
            "app_id": str(r["app_id"]),
            "token_hash": str(r["token_hash"]),
            "enabled": bool(int(r["enabled"])),
            "created_at": str(r["created_at"]),
            "last_used_at": str(r["last_used_at"]),
        }

    def touch_fleet_client_last_used(self) -> None:
        from ai_brain.access import FLEET_CLIENT_ID

        self._executor.execute(
            "UPDATE brain_app_clients SET last_used_at = ? WHERE app_id = ?",
            (_utc_now_iso(), FLEET_CLIENT_ID),
        )

    def load_usage_quotas(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Lädt periodenbezogene Free-Kontingent-Zähler aus `brain_meta`."""
        return parse_usage_quotas(self.get_meta(USAGE_QUOTAS_META_KEY, "{}"))

    def reset_usage_quotas(self) -> None:
        """Setzt gespeicherte Verbrauchs-Zähler zurück."""
        self.set_meta(USAGE_QUOTAS_META_KEY, json.dumps({"tools": {}, "models": {}}))

    def load_app_policies(self) -> dict[str, AppBrainPolicy]:
        """Lädt gespeicherte Tier-/Budget-Optionen aller Apps aus `brain_meta`."""
        try:
            raw = json.loads(self.get_meta(APP_POLICIES_META_KEY, "{}") or "{}")
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        parsed: dict[str, AppBrainPolicy] = {}
        for key, val in raw.items():
            clean_key = str(key).strip().lower()
            if clean_key and isinstance(val, dict):
                try:
                    parsed[clean_key] = AppBrainPolicy.model_validate(val)
                except ValueError:
                    continue
        return parsed

    def get_app_policy(self, app_name: str) -> AppBrainPolicy:
        """Liefert die gespeicherte App-Policy (oder den kostenfreien Standard-Default)."""
        key = (app_name or "default").strip().lower()
        return self.load_app_policies().get(key) or default_app_policy(app_name)

    def save_app_policy(self, app_name: str, policy: AppBrainPolicy) -> AppBrainPolicy:
        """Speichert die Tier-/Budget-Optionen einer App dauerhaft in Turso/SQLite."""
        key = (app_name or "default").strip().lower()
        all_policies = self.load_app_policies()
        all_policies[key] = policy
        payload = {k: p.model_dump(mode="json") for k, p in sorted(all_policies.items())}
        self.set_meta(APP_POLICIES_META_KEY, json.dumps(payload))
        return policy

    def append_access_log(
        self, entry: BrainAccessLogEntry, *, max_entries: int = MAX_ACCESS_LOG_ENTRIES
    ) -> None:
        """Speichert einen Zugriffs-Log-Eintrag und begrenzt die Historie atomar auf `max_entries`."""
        limit = max(20, min(int(max_entries), 2000))
        stmts = [
            (
                """
                INSERT INTO brain_access_logs (
                    id, created_at, caller_app, environment, action, tier,
                    budget_mode, status, latency_ms, model_used, tools_used,
                    result_count, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.created_at,
                    entry.caller_app,
                    entry.environment,
                    entry.action,
                    entry.tier,
                    entry.budget_mode,
                    entry.status,
                    entry.latency_ms,
                    entry.model_used,
                    ", ".join(entry.tools_used),
                    entry.result_count,
                    entry.summary,
                ),
            ),
            (
                """
                DELETE FROM brain_access_logs
                WHERE id NOT IN (
                    SELECT id FROM brain_access_logs ORDER BY created_at DESC, id DESC LIMIT ?
                )
                """,
                (limit,),
            ),
        ]
        try:
            self._executor.execute_batch(stmts)
        except BrainStorageError as exc:
            if "no such table" not in str(exc).lower():
                raise
            self._init_schema()
            self._executor.execute_batch(stmts)

    def list_access_logs(
        self,
        *,
        limit: int = 200,
        caller_app: str = "",
        status: str = "",
    ) -> list[BrainAccessLogEntry]:
        """Liefert die neuesten Zugriffs-Logs (optional gefiltert nach App oder Status)."""
        safe_limit = max(1, min(int(limit), MAX_ACCESS_LOG_ENTRIES))
        clean_caller = caller_app.strip()
        clean_status = status.strip()
        sql = (
            "SELECT id, created_at, caller_app, environment, action, tier, "
            "budget_mode, status, latency_ms, model_used, tools_used, result_count, summary "
            "FROM brain_access_logs "
            "WHERE (? = '' OR caller_app = ?) AND (? = '' OR status = ?) "
            "ORDER BY created_at DESC, id DESC LIMIT ?"
        )
        params = (clean_caller, clean_caller, clean_status, clean_status, safe_limit)
        try:
            res = self._executor.execute(sql, params)
        except BrainStorageError as exc:
            if "no such table" not in str(exc).lower():
                raise
            self._init_schema()
            return []
        return [_row_to_access_log(r) for r in res.rows]

    def clear_access_logs(self) -> int:
        """Löscht alle gespeicherten Zugriffs-Logs aus der Brain-Datenbank."""
        try:
            res = self._executor.execute("DELETE FROM brain_access_logs")
            return res.affected_rows
        except BrainStorageError as exc:
            if "no such table" not in str(exc).lower():
                raise
            self._init_schema()
            return 0

    def load_dashboard_state(self) -> BrainDashboardState:
        """Lädt Modelle, Tools, Sequenzen, Benchmark, Secrets, Kontingente, Flotten-Key & Logs in einem Batch."""
        from ai_brain.access import FLEET_CLIENT_ID

        stmts = [
            ("SELECT payload_json FROM brain_models ORDER BY id ASC", ()),
            ("SELECT payload_json FROM brain_tools ORDER BY id ASC", ()),
            ("SELECT tier, payload_json FROM brain_sequences", ()),
            ("SELECT payload_json FROM brain_benchmark_runs ORDER BY started_at DESC LIMIT 1", ()),
            ("SELECT env_name, updated_at FROM brain_secrets ORDER BY env_name ASC", ()),
            ("SELECT key, value FROM brain_meta WHERE key IN (?, ?, ?)", ("excluded_ids", "excluded_reasons", USAGE_QUOTAS_META_KEY)),
            ("SELECT app_id, token_hash, enabled, created_at, last_used_at FROM brain_app_clients WHERE app_id = ?", (FLEET_CLIENT_ID,)),
            (
                "SELECT id, created_at, caller_app, environment, action, tier, "
                "budget_mode, status, latency_ms, model_used, tools_used, result_count, summary "
                "FROM brain_access_logs ORDER BY created_at DESC, id DESC LIMIT 200",
                (),
            ),
        ]
        try:
            res_m, res_t, res_s, res_b, res_sec, res_meta, res_fleet, res_logs = self._executor.execute_batch(stmts)
        except BrainStorageError as exc:
            if "no such table" not in str(exc).lower():
                raise
            if self._cache_key:
                with _INITIALIZED_LOCK:
                    _INITIALIZED_BACKENDS.discard(self._cache_key)
            self._init_schema()
            self.ensure_seeded()
            if self._cache_key:
                with _INITIALIZED_LOCK:
                    _INITIALIZED_BACKENDS.add(self._cache_key)
            res_m, res_t, res_s, res_b, res_sec, res_meta, res_fleet, res_logs = self._executor.execute_batch(stmts)

        models = [AIModelSpec.model_validate_json(str(r["payload_json"])) for r in res_m.rows]
        tools = [ToolSpec.model_validate_json(str(r["payload_json"])) for r in res_t.rows]
        seqs = build_default_sequences()
        for row in res_s.rows:
            seqs[BrainTier(str(row["tier"]))] = TierSequenceConfig.model_validate_json(str(row["payload_json"]))
        latest = BenchmarkReport.model_validate_json(str(res_b.rows[0]["payload_json"])) if res_b.rows else None
        sec_ts = {str(r["env_name"]): str(r["updated_at"]) for r in res_sec.rows}
        meta_map = {str(r["key"]): str(r["value"]) for r in res_meta.rows}
        excl_m, _ = _parse_excluded_payload(meta_map.get("excluded_ids", "{}"))
        reasons_m, _ = _parse_reasons_payload(meta_map.get("excluded_reasons", "{}"))
        quotas = parse_usage_quotas(meta_map.get(USAGE_QUOTAS_META_KEY, "{}"))
        fleet_client: dict[str, Any] | None = None
        if res_fleet.rows:
            r = res_fleet.rows[0]
            fleet_client = {
                "app_id": str(r["app_id"]),
                "token_hash": str(r["token_hash"]),
                "enabled": bool(int(r["enabled"])),
                "created_at": str(r["created_at"]),
                "last_used_at": str(r["last_used_at"]),
            }
        access_logs = [_row_to_access_log(r) for r in res_logs.rows]
        return BrainDashboardState(
            models=models,
            tools=tools,
            sequences=seqs,
            latest_benchmark=latest,
            secret_timestamps=sec_ts,
            excluded_model_ids=tuple(sorted(excl_m)),
            excluded_model_reasons={mid: reasons_m[mid] for mid in excl_m if mid in reasons_m},
            usage_quotas=quotas,
            fleet_client=fleet_client,
            access_logs=access_logs,
        )

    def close(self) -> None:
        self._executor.close()


def create_brain_repository(
    turso_url: str = "",
    turso_auth_token: str = "",
    sqlite_path: str | Path = "data/ai_brain.db",
    base_dir: Path | None = None,
    seed_defaults: bool = True,
) -> BrainRepository:
    """Erstellt ein Brain-Repository mit Turso Cloud oder lokalem SQLite-Fallback."""
    clean_url = (turso_url or "").strip()
    clean_tok = (turso_auth_token or "").strip()
    if clean_url and clean_tok:
        executor: BrainSqlExecutor = TursoBrainExecutor(clean_url, clean_tok)
    else:
        executor = SqliteBrainExecutor(sqlite_path, base_dir=base_dir)
    return BrainRepository(executor=executor, seed_defaults=seed_defaults)
