"""Persistenz-Repository für Modelle, Tools, Tier-Sequenzen und Testläufe."""

from __future__ import annotations

import datetime
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ki_brain.catalog import build_default_models, build_default_sequences, build_default_tools
from ki_brain.db_executors import BrainSqlExecutor, SqliteBrainExecutor, TursoBrainExecutor
from ki_brain.enums import BrainTier
from ki_brain.exceptions import BrainStorageError
from ki_brain.models import AIModelSpec, BenchmarkReport, TierSequenceConfig, ToolSpec
from ki_brain.pricing import infer_model_is_free
from ki_brain.quotas import USAGE_QUOTAS_META_KEY, parse_usage_quotas
from ki_brain.security import validate_identifier

_INITIALIZED_BACKENDS: set[str] = set()
_INITIALIZED_LOCK = threading.Lock()
_CATALOG_REV = "3"


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
    app_clients: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        meta_res = self._executor.execute(
            "SELECT key, value FROM brain_meta WHERE key IN ('seeded_v1', 'seeded_catalog_rev', 'known_builtin_ids', 'excluded_ids')"
        )
        meta = {str(r["key"]): str(r["value"]) for r in meta_res.rows}
        if meta.get("seeded_v1") == "1" and meta.get("seeded_catalog_rev") == _CATALOG_REV:
            return

        now = _utc_now_iso()
        defaults_models = build_default_models()
        defaults_tools = build_default_tools()
        stmts: list[tuple[str, tuple[str, ...]]] = []

        if meta.get("seeded_v1") != "1":
            for model in defaults_models:
                stmts.append(
                    ("INSERT OR IGNORE INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)", (model.id, model.model_dump_json(), now))
                )
            for tool in defaults_tools:
                stmts.append(
                    ("INSERT OR IGNORE INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)", (tool.id, tool.model_dump_json(), now))
                )
            for tier, seq in build_default_sequences().items():
                updated_seq = seq.model_copy(update={"updated_at": seq.updated_at or now})
                stmts.append(
                    ("INSERT OR IGNORE INTO brain_sequences (tier, payload_json, updated_at) VALUES (?, ?, ?)", (tier.value, updated_seq.model_dump_json(), now))
                )
            known_ids = {"models": [m.id for m in defaults_models], "tools": [t.id for t in defaults_tools]}
            stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("known_builtin_ids", json.dumps(known_ids))))
            stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("seeded_v1", "1")))
            stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("seeded_catalog_rev", _CATALOG_REV)))
            self._executor.execute_batch(stmts)
            return

        # Non-destruktives Katalog-Upgrade (inkl. Maps Grounding, Kontingent-Limits & Paid-Only-Klassifikation)
        try:
            known = json.loads(meta.get("known_builtin_ids") or "{}")
        except json.JSONDecodeError:
            known = {}
        try:
            excl = json.loads(meta.get("excluded_ids") or "{}")
        except json.JSONDecodeError:
            excl = {}
        known_m, known_t = set(known.get("models", [])), set(known.get("tools", []))
        excl_m = {str(x).strip().lower() for x in excl.get("models", [])}
        excl_t = {str(x).strip().lower() for x in excl.get("tools", [])}

        for model in defaults_models:
            if model.id not in known_m and model.id not in excl_m:
                stmts.append(("INSERT OR IGNORE INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)", (model.id, model.model_dump_json(), now)))
            known_m.add(model.id)
        for tool in defaults_tools:
            if tool.id not in known_t and tool.id not in excl_t:
                stmts.append(("INSERT OR IGNORE INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)", (tool.id, tool.model_dump_json(), now)))
            known_t.add(tool.id)

        self._append_rev3_metadata_upgrades(stmts, defaults_tools, now)
        stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("known_builtin_ids", json.dumps({"models": sorted(known_m), "tools": sorted(known_t)}))))
        stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("seeded_catalog_rev", _CATALOG_REV)))
        self._executor.execute_batch(stmts)

    def _append_rev3_metadata_upgrades(
        self, stmts: list[tuple[str, tuple[str, ...]]], defaults_tools: list[ToolSpec], now: str
    ) -> None:
        default_t_map = {t.id: t for t in defaults_tools}
        for m in self.list_models():
            if m.source != "custom" and not infer_model_is_free(
                m.model_id, m.name, provider=m.provider, input_cost_per_1m=m.input_cost_per_1m, output_cost_per_1m=m.output_cost_per_1m
            ) and (m.is_free or m.free_quota_period != "none"):
                up_m = m.model_copy(update={"is_free": False, "free_quota_period": "none"})
                stmts.append(("UPDATE brain_models SET payload_json = ?, updated_at = ? WHERE id = ?", (up_m.model_dump_json(), now, m.id)))
        for t in self.list_tools():
            dt = default_t_map.get(t.id)
            if dt and t.free_quota_limit == 0 and dt.free_quota_limit > 0:
                up_t = t.model_copy(update={
                    "free_quota_limit": dt.free_quota_limit, "free_quota_period": dt.free_quota_period,
                    "free_quota_unit": dt.free_quota_unit, "paid_cost_per_1k": dt.paid_cost_per_1k,
                })
                stmts.append(("UPDATE brain_tools SET payload_json = ?, updated_at = ? WHERE id = ?", (up_t.model_dump_json(), now, t.id)))
        paid_model_ids = {
            m.id for m in self.list_models()
            if not m.is_free or not infer_model_is_free(m.model_id, m.name, provider=m.provider, input_cost_per_1m=m.input_cost_per_1m, output_cost_per_1m=m.output_cost_per_1m)
        }
        for seq in self.list_sequences().values():
            if seq.manual_override:
                continue
            free_m_seq = [mid for mid in seq.model_sequence if mid not in paid_model_ids]
            paid_m_seq = seq.paid_model_sequence or list(seq.model_sequence)
            paid_s_seq = seq.paid_search_tool_sequence or list(seq.search_tool_sequence)
            paid_e_seq = seq.paid_extractor_tool_sequence or list(seq.extractor_tool_sequence)
            has_paid_deep = any(mid == "deep-research-preview-04-2026" for mid in paid_m_seq)
            up_s = seq.model_copy(update={
                "model_sequence": free_m_seq,
                "paid_model_sequence": paid_m_seq,
                "paid_search_tool_sequence": paid_s_seq,
                "paid_extractor_tool_sequence": paid_e_seq,
                "use_deep_agent": False,
                "paid_use_deep_agent": bool(seq.use_deep_agent or has_paid_deep) if seq.tier == BrainTier.DEEP else False,
            })
            stmts.append(("UPDATE brain_sequences SET payload_json = ?, updated_at = ? WHERE tier = ?", (up_s.model_dump_json(), now, seq.tier.value)))

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
                    "DELETE FROM brain_meta WHERE key IN ('seeded_v1', 'seeded_catalog_rev', 'known_builtin_ids', 'excluded_ids', 'excluded_reasons')",
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
        from ki_brain.secrets_vault import validate_secret_env_name

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
        from ki_brain.secrets_vault import validate_secret_env_name

        clean_name = validate_secret_env_name(env_name)
        res = self._executor.execute("DELETE FROM brain_secrets WHERE env_name = ?", (clean_name,))
        return res.affected_rows > 0

    def list_secret_timestamps(self) -> dict[str, str]:
        res = self._executor.execute("SELECT env_name, updated_at FROM brain_secrets ORDER BY env_name ASC")
        return {str(r["env_name"]): str(r["updated_at"]) for r in res.rows}

    def load_decrypted_secrets(self, passphrase: str = "") -> dict[str, str]:
        from ki_brain.secrets_vault import decrypt_secret_value

        res = self._executor.execute("SELECT env_name, encrypted_value FROM brain_secrets")
        decrypted: dict[str, str] = {}
        for row in res.rows:
            val = decrypt_secret_value(str(row["encrypted_value"]), passphrase=passphrase)
            if val:
                decrypted[str(row["env_name"])] = val
        return decrypted

    def upsert_app_client(self, app_id: str, token_hash: str, *, enabled: bool = True) -> str:
        from ki_brain.access import normalize_client_app_id

        clean_app = normalize_client_app_id(app_id)
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
            (clean_app, token_hash, 1 if enabled else 0, now),
        )
        return now

    def set_app_client_enabled(self, app_id: str, enabled: bool) -> bool:
        from ki_brain.access import normalize_client_app_id

        clean_app = normalize_client_app_id(app_id)
        res = self._executor.execute(
            "UPDATE brain_app_clients SET enabled = ? WHERE app_id = ?",
            (1 if enabled else 0, clean_app),
        )
        return res.affected_rows > 0

    def revoke_app_client(self, app_id: str) -> bool:
        from ki_brain.access import normalize_client_app_id

        clean_app = normalize_client_app_id(app_id)
        res = self._executor.execute("DELETE FROM brain_app_clients WHERE app_id = ?", (clean_app,))
        return res.affected_rows > 0

    def get_app_client_record(self, app_id: str) -> dict[str, Any] | None:
        from ki_brain.access import normalize_client_app_id

        clean_app = normalize_client_app_id(app_id)
        res = self._executor.execute(
            "SELECT app_id, token_hash, enabled, created_at, last_used_at FROM brain_app_clients WHERE app_id = ?",
            (clean_app,),
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

    def list_app_client_records(self) -> dict[str, dict[str, Any]]:
        res = self._executor.execute(
            "SELECT app_id, token_hash, enabled, created_at, last_used_at FROM brain_app_clients ORDER BY app_id ASC"
        )
        return {
            str(r["app_id"]): {
                "app_id": str(r["app_id"]),
                "token_hash": str(r["token_hash"]),
                "enabled": bool(int(r["enabled"])),
                "created_at": str(r["created_at"]),
                "last_used_at": str(r["last_used_at"]),
            }
            for r in res.rows
        }

    def touch_app_client_last_used(self, app_id: str) -> None:
        from ki_brain.access import normalize_client_app_id

        clean_app = normalize_client_app_id(app_id)
        self._executor.execute(
            "UPDATE brain_app_clients SET last_used_at = ? WHERE app_id = ?",
            (_utc_now_iso(), clean_app),
        )

    def load_usage_quotas(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Lädt periodenbezogene Free-Kontingent-Zähler aus `brain_meta`."""
        return parse_usage_quotas(self.get_meta(USAGE_QUOTAS_META_KEY, "{}"))

    def reset_usage_quotas(self) -> None:
        """Setzt gespeicherte Verbrauchs-Zähler zurück."""
        self.set_meta(USAGE_QUOTAS_META_KEY, json.dumps({"tools": {}, "models": {}}))

    def load_dashboard_state(self) -> BrainDashboardState:
        """Lädt Modelle, Tools, Sequenzen, Benchmark, Secrets, Kontingente & App-Clients in einem Batch."""
        stmts = [
            ("SELECT payload_json FROM brain_models ORDER BY id ASC", ()),
            ("SELECT payload_json FROM brain_tools ORDER BY id ASC", ()),
            ("SELECT tier, payload_json FROM brain_sequences", ()),
            ("SELECT payload_json FROM brain_benchmark_runs ORDER BY started_at DESC LIMIT 1", ()),
            ("SELECT env_name, updated_at FROM brain_secrets ORDER BY env_name ASC", ()),
            ("SELECT key, value FROM brain_meta WHERE key IN (?, ?, ?)", ("excluded_ids", "excluded_reasons", USAGE_QUOTAS_META_KEY)),
            ("SELECT app_id, token_hash, enabled, created_at, last_used_at FROM brain_app_clients ORDER BY app_id ASC", ()),
        ]
        try:
            res_m, res_t, res_s, res_b, res_sec, res_meta, res_clients = self._executor.execute_batch(stmts)
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
            res_m, res_t, res_s, res_b, res_sec, res_meta, res_clients = self._executor.execute_batch(stmts)

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
        app_clients = {
            str(r["app_id"]): {
                "app_id": str(r["app_id"]),
                "token_hash": str(r["token_hash"]),
                "enabled": bool(int(r["enabled"])),
                "created_at": str(r["created_at"]),
                "last_used_at": str(r["last_used_at"]),
            }
            for r in res_clients.rows
        }
        return BrainDashboardState(
            models=models,
            tools=tools,
            sequences=seqs,
            latest_benchmark=latest,
            secret_timestamps=sec_ts,
            excluded_model_ids=tuple(sorted(excl_m)),
            excluded_model_reasons={mid: reasons_m[mid] for mid in excl_m if mid in reasons_m},
            usage_quotas=quotas,
            app_clients=app_clients,
        )

    def close(self) -> None:
        self._executor.close()


def create_brain_repository(
    turso_url: str = "",
    turso_auth_token: str = "",
    sqlite_path: str | Path = "data/ki_brain.db",
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
