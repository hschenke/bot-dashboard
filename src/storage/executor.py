"""Einheitliches SQL-Ausführungsprotokoll für SQLite und Turso Cloud LibSQL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class StatementResult:
    """Normalisiertes Ergebnis einer parametrisierten SQL-Anweisung."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    affected_rows: int = 0


class SqlExecutor(Protocol):
    """Abstrakte Datenbankschnittstelle für identische Queries auf SQLite & Turso."""

    @property
    def backend_name(self) -> str: ...

    def execute(self, sql: str, params: Sequence[Any] = ()) -> StatementResult: ...

    def execute_batch(
        self,
        statements: Sequence[tuple[str, Sequence[Any]]],
    ) -> list[StatementResult]: ...

    def close(self) -> None: ...
