"""Persistenz-Repository für Modelle, Tools, Tier-Sequenzen und Testläufe."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from ki_brain.catalog import build_default_models, build_default_sequences, build_default_tools
from ki_brain.db_executors import BrainSqlExecutor, SqliteBrainExecutor, TursoBrainExecutor
from ki_brain.enums import BrainTier
from ki_brain.models import AIModelSpec, BenchmarkReport, TierSequenceConfig, ToolSpec
from ki_brain.security import validate_identifier

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
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class BrainRepository:
    """Einheitliches Repository (SQLite & Turso) für den dynamischen Brain-Zustand."""

    def __init__(self, executor: BrainSqlExecutor, seed_defaults: bool = True) -> None:
        self._executor = executor
        self._init_schema()
        if seed_defaults:
            self.ensure_seeded()

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
        defaults_models = build_default_models()
        defaults_tools = build_default_tools()
        stmts: list[tuple[str, tuple[str, ...]]] = []
        for model in defaults_models:
            stmts.append(
                (
                    "INSERT OR IGNORE INTO brain_models (id, payload_json, updated_at) VALUES (?, ?, ?)",
                    (model.id, model.model_dump_json(), now),
                )
            )
        for tool in defaults_tools:
            stmts.append(
                (
                    "INSERT OR IGNORE INTO brain_tools (id, payload_json, updated_at) VALUES (?, ?, ?)",
                    (tool.id, tool.model_dump_json(), now),
                )
            )
        for tier, seq in build_default_sequences().items():
            updated_seq = seq.model_copy(update={"updated_at": seq.updated_at or now})
            stmts.append(
                (
                    "INSERT OR IGNORE INTO brain_sequences (tier, payload_json, updated_at) VALUES (?, ?, ?)",
                    (tier.value, updated_seq.model_dump_json(), now),
                )
            )
        known_ids = {"models": [m.id for m in defaults_models], "tools": [t.id for t in defaults_tools]}
        stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("known_builtin_ids", json.dumps(known_ids))))
        stmts.append(("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", ("seeded_v1", "1")))
        self._executor.execute_batch(stmts)

    def list_excluded_ids(self) -> tuple[set[str], set[str]]:
        """Liefert persistent ausgeschlossene (gelöschte/defekte) Modell- und Tool-IDs."""
        raw = self.get_meta("excluded_ids", "{}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return set(), set()
        models = {str(x).strip() for x in data.get("models", []) if str(x).strip()}
        tools = {str(x).strip() for x in data.get("tools", []) if str(x).strip()}
        return models, tools

    def _save_excluded_ids(self, models: set[str], tools: set[str]) -> None:
        self.set_meta(
            "excluded_ids",
            json.dumps({"models": sorted(models), "tools": sorted(tools)}),
        )

    def add_excluded_ids(
        self,
        *,
        model_ids: list[str] | set[str] | tuple[str, ...] = (),
        tool_ids: list[str] | set[str] | tuple[str, ...] = (),
    ) -> None:
        """Markiert entfernte oder nicht funktionierende IDs dauerhaft gegen Auto-Sync-Wiederaufnahme."""
        clean_m = {str(x).strip().lower() for x in model_ids if str(x).strip()}
        clean_t = {str(x).strip().lower() for x in tool_ids if str(x).strip()}
        if not clean_m and not clean_t:
            return
        models, tools = self.list_excluded_ids()
        updated_m = models | clean_m
        updated_t = tools | clean_t
        if updated_m != models or updated_t != tools:
            self._save_excluded_ids(updated_m, updated_t)

    def remove_excluded_id(self, *, model_id: str = "", tool_id: str = "") -> None:
        """Hebt den Auto-Sync-Ausschluss bei expliziter manueller Neuanlage wieder auf."""
        models, tools = self.list_excluded_ids()
        changed = False
        clean_m = model_id.strip().lower()
        clean_t = tool_id.strip().lower()
        if clean_m and clean_m in models:
            models.discard(clean_m)
            changed = True
        if clean_t and clean_t in tools:
            tools.discard(clean_t)
            changed = True
        if changed:
            self._save_excluded_ids(models, tools)

    def reset_to_defaults(self) -> None:
        self._executor.execute_batch(
            [
                ("DELETE FROM brain_models", ()),
                ("DELETE FROM brain_tools", ()),
                ("DELETE FROM brain_sequences", ()),
                ("DELETE FROM brain_meta WHERE key IN ('seeded_v1', 'known_builtin_ids', 'excluded_ids')", ()),
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
