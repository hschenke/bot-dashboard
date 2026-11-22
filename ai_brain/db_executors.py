"""SQLite-WAL & Turso-HTTPS (Hrana v2) Datenbank-Executoren für ai_brain."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from ai_brain.exceptions import BrainStorageError, BrainValidationError
from ai_brain.security import resolve_safe_sqlite_path, validate_safe_external_url

_TURSO_BATCH_SIZE = 20
_TURSO_TIMEOUT_SECONDS = 12.0


@dataclass(frozen=True)
class BrainStatementResult:
    """Normiertes SQL-Abfrageergebnis für SQLite und Turso."""

    rows: list[dict[str, Any]]
    affected_rows: int = 0


class BrainSqlExecutor(Protocol):
    """Gemeinsame Datenbank-Schnittstelle für lokales SQLite und Turso Cloud."""

    @property
    def backend_name(self) -> str: ...

    def execute(self, sql: str, params: Sequence[Any] = ()) -> BrainStatementResult: ...

    def execute_batch(
        self,
        statements: Sequence[tuple[str, Sequence[Any]]],
    ) -> list[BrainStatementResult]: ...

    def close(self) -> None: ...


class SqliteBrainExecutor:
    """Thread-sicherer lokaler SQLite-Executor (WAL-Modus, 100 % `?`-Platzhalter)."""

    def __init__(self, db_path: str | Path = "data/ai_brain.db", base_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._is_memory = str(db_path).strip() == ":memory:"
        if self._is_memory:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._label = "SQLite (:memory:)"
        else:
            target = self._resolve_target(db_path, base_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(target), check_same_thread=False, timeout=10.0)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=10000;")
            self._label = f"SQLite ({target.name})"
        self._conn.row_factory = sqlite3.Row

    @staticmethod
    def _resolve_target(db_path: str | Path, base_dir: Path | None) -> Path:
        if isinstance(db_path, Path):
            if ".." in db_path.parts:
                raise BrainValidationError(f"Unsicherer SQLite-Pfad: '{db_path}'.")
            return db_path.resolve()
        anchor = (base_dir or Path.cwd()).resolve()
        candidate = Path(db_path.strip())
        if candidate.is_absolute():
            if ".." in candidate.parts:
                raise BrainValidationError(f"Unsicherer SQLite-Pfad: '{db_path}'.")
            return candidate.resolve()
        return resolve_safe_sqlite_path(anchor, db_path)

    @property
    def backend_name(self) -> str:
        return self._label

    def execute(self, sql: str, params: Sequence[Any] = ()) -> BrainStatementResult:
        return self.execute_batch([(sql, params)])[0]

    def execute_batch(
        self,
        statements: Sequence[tuple[str, Sequence[Any]]],
    ) -> list[BrainStatementResult]:
        if not statements:
            return []
        results: list[BrainStatementResult] = []
        with self._lock:
            try:
                for sql, params in statements:
                    cur = self._conn.execute(sql, tuple(params))
                    rows = [dict(row) for row in cur.fetchall()] if cur.description else []
                    affected = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
                    results.append(BrainStatementResult(rows=rows, affected_rows=affected))
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise BrainStorageError(f"SQLite-Fehler im AI-Brain: {exc}") from exc
        return results

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _normalize_turso_pipeline_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if cleaned.startswith("libsql://"):
        cleaned = "https://" + cleaned.removeprefix("libsql://")
    elif not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"
    safe = validate_safe_external_url(cleaned)
    parsed = urlparse(safe)
    return f"{parsed.scheme}://{parsed.netloc}/v2/pipeline"


def _encode_arg(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "value": "1" if value else "0"}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    return {"type": "text", "value": str(value)}


def _decode_cell(cell: dict[str, Any]) -> Any:
    ctype, raw = cell.get("type"), cell.get("value")
    if ctype == "null" or raw is None:
        return None
    if ctype == "integer":
        return int(raw)
    if ctype == "float":
        return float(raw)
    return str(raw)


class TursoBrainExecutor:
    """HTTPS Hrana v2 Executor für gemeinsame Nutzung des Brains über Turso Cloud."""

    def __init__(self, database_url: str, auth_token: str) -> None:
        token = auth_token.strip()
        if not token:
            raise BrainStorageError("Kein Turso Auth-Token für das AI-Brain angegeben.")
        self._pipeline_url = _normalize_turso_pipeline_url(database_url)
        parsed = urlparse(self._pipeline_url)
        self._host = parsed.netloc
        self._database_url = f"{parsed.scheme}://{parsed.netloc}"
        self._auth_token = token
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "AIWebsearchBrain/1.0",
            }
        )

    @property
    def backend_name(self) -> str:
        return f"Turso Cloud ({self._host.split('.')[0]})"

    def _connection_credentials(self) -> tuple[str, str]:
        """Interne Turso-Verbindungsdaten für Admin-Ausstellung von `AI_BRAIN_KEY`."""
        return self._database_url, self._auth_token

    def execute(self, sql: str, params: Sequence[Any] = ()) -> BrainStatementResult:
        return self.execute_batch([(sql, params)])[0]

    def execute_batch(
        self,
        statements: Sequence[tuple[str, Sequence[Any]]],
    ) -> list[BrainStatementResult]:
        if not statements:
            return []
        results: list[BrainStatementResult] = []
        for idx in range(0, len(statements), _TURSO_BATCH_SIZE):
            chunk = statements[idx : idx + _TURSO_BATCH_SIZE]
            results.extend(self._run_chunk(chunk))
        return results

    def _run_chunk(self, chunk: Sequence[tuple[str, Sequence[Any]]]) -> list[BrainStatementResult]:
        reqs: list[dict[str, Any]] = [
            {"type": "execute", "stmt": {"sql": sql, "args": [_encode_arg(p) for p in params]}}
            for sql, params in chunk
        ]
        reqs.append({"type": "close"})
        for attempt in range(3):
            try:
                with self._lock:
                    resp = self._session.post(
                        self._pipeline_url, json={"requests": reqs}, timeout=_TURSO_TIMEOUT_SECONDS
                    )
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    raise BrainStorageError(f"Turso HTTP-Fehler {resp.status_code}.")
                return self._parse_results(resp.json())
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise BrainStorageError(f"Turso Netzwerkfehler: {exc}") from exc
        raise BrainStorageError("Turso Pipeline-Aufruf fehlgeschlagen.")

    @staticmethod
    def _parse_results(payload: dict[str, Any]) -> list[BrainStatementResult]:
        parsed: list[BrainStatementResult] = []
        for item in payload.get("results", []):
            inner = item.get("response", {})
            if inner.get("type") == "close":
                continue
            if item.get("type") == "error":
                raise BrainStorageError(f"Turso SQL-Fehler: {item.get('error', {}).get('message', '')}")
            block = inner.get("result", {})
            cols = [str(c.get("name", "")) for c in block.get("cols", [])]
            rows = [{cols[i]: _decode_cell(cell) for i, cell in enumerate(row)} for row in block.get("rows", [])]
            affected = int(block.get("affected_row_count", 0) or 0)
            parsed.append(BrainStatementResult(rows=rows, affected_rows=affected))
        return parsed

    def close(self) -> None:
        with self._lock:
            self._session.close()
