"""Lokaler SQLite-Executor (WAL-Modus & optionaler Read-Only-Modus für Fremd-DBs)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.config.constants import SQLITE_TIMEOUT_SECONDS
from src.domain.exceptions import StorageConnectionError, StorageError
from src.storage.executor import StatementResult


class SqliteExecutor:
    """Thread-sicherer SQLite-Executor mit parametrisierten `?`-Queries."""

    def __init__(
        self,
        db_path: Path | str,
        read_only: bool = False,
        timeout_seconds: float = SQLITE_TIMEOUT_SECONDS,
    ) -> None:
        self._read_only = read_only
        self._lock = threading.Lock()

        if str(db_path) == ":memory:":
            self._label = "SQLite (:memory:)"
            self._conn = sqlite3.connect(":memory:", timeout=timeout_seconds, check_same_thread=False)
        else:
            path_obj = Path(db_path).resolve()
            self._label = f"SQLite ({path_obj.name})"
            if read_only:
                if not path_obj.is_file():
                    raise StorageConnectionError(f"Lokale SQLite-Datei nicht vorhanden: {path_obj}")
                uri = f"{path_obj.as_uri()}?mode=ro"
                self._conn = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=timeout_seconds,
                    check_same_thread=False,
                )
            else:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(
                    str(path_obj),
                    timeout=timeout_seconds,
                    check_same_thread=False,
                )

        self._conn.row_factory = sqlite3.Row
        if not read_only and str(db_path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")

    @property
    def backend_name(self) -> str:
        return self._label

    def execute(self, sql: str, params: Sequence[Any] = ()) -> StatementResult:
        return self.execute_batch([(sql, params)])[0]

    def execute_batch(
        self,
        statements: Sequence[tuple[str, Sequence[Any]]],
    ) -> list[StatementResult]:
        if not statements:
            return []
        results: list[StatementResult] = []
        try:
            with self._lock, self._conn:
                for sql, params in statements:
                    cursor = self._conn.execute(sql, tuple(params))
                    fetched = cursor.fetchall() if cursor.description else []
                    rows = [dict(row) for row in fetched]
                    affected = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount >= 0 else 0
                    results.append(StatementResult(rows=rows, affected_rows=affected))
            return results
        except sqlite3.Error as exc:
            raise StorageError(f"SQLite-Ausführungsfehler: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()
