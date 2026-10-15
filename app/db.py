"""SQLite connections and schema management.

Table schemas are derived from domain dataclasses, so the database shape always follows the
entity definitions in code. A field named ``id`` becomes an auto-incrementing primary key whose
values are never reused, other fields map by type, and ``X | None`` fields are nullable.
"""

import dataclasses
import sqlite3
import types
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

import aiosqlite

from app.models import Task

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_SQLITE_TYPES: dict[type, str] = {bool: "INTEGER", int: "INTEGER", str: "TEXT", float: "REAL"}

BUSY_TIMEOUT_MS = 5000


class SchemaMismatchError(RuntimeError):
    """The database table does not match the schema derived from the domain model."""


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    python_type: type
    sql_type: str
    nullable: bool
    primary_key: bool

    def definition(self) -> str:
        if self.primary_key:
            return f"{self.name} {self.sql_type} PRIMARY KEY AUTOINCREMENT"
        parts = [self.name, self.sql_type]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.python_type is bool:
            parts.append(f"CHECK ({self.name} IN (0, 1))")
        return " ".join(parts)

    def to_python(self, value: Any) -> Any:
        if value is None or self.python_type is not bool:
            return value
        return bool(value)


def _resolve_type(annotation: Any) -> tuple[type, bool]:
    """Return the concrete type and whether it is nullable (``X | None``)."""
    if get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) != 1 or len(get_args(annotation)) != 2:
            raise TypeError(f"Unsupported union type for a column: {annotation!r}")
        return members[0], True
    return annotation, False


@dataclass(frozen=True, slots=True)
class TableSchema:
    table: str
    columns: tuple[Column, ...]
    indexes: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def for_dataclass(
        cls,
        entity: "type[DataclassInstance]",
        table: str,
        indexes: Sequence[Sequence[str]] = (),
    ) -> "TableSchema":
        hints = get_type_hints(entity)
        columns: list[Column] = []
        for field in dataclasses.fields(entity):
            python_type, nullable = _resolve_type(hints[field.name])
            sql_type = _SQLITE_TYPES.get(python_type)
            if sql_type is None:
                raise TypeError(
                    f"No SQLite column type for {entity.__name__}.{field.name}: {python_type!r}"
                )
            primary_key = field.name == "id"
            if primary_key and (python_type is not int or nullable):
                raise TypeError(f"{entity.__name__}.id must be a non-optional int")
            columns.append(Column(field.name, python_type, sql_type, nullable, primary_key))
        if not any(column.primary_key for column in columns):
            raise TypeError(f"{entity.__name__} must have an 'id' field")
        names = {column.name for column in columns}
        for index in indexes:
            unknown = [name for name in index if name not in names]
            if not index or unknown:
                raise TypeError(f"Index {tuple(index)} references unknown fields: {unknown}")
        return cls(
            table=table,
            columns=tuple(columns),
            indexes=tuple(tuple(index) for index in indexes),
        )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def writable_column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.primary_key)

    def create_table_sql(self) -> str:
        definitions = ",\n    ".join(column.definition() for column in self.columns)
        return f"CREATE TABLE IF NOT EXISTS {self.table} (\n    {definitions}\n)"

    def create_index_sql(self) -> list[str]:
        return [
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_{'_'.join(index)} "
            f"ON {self.table} ({', '.join(index)})"
            for index in self.indexes
        ]

    def row_to_values(self, row: sqlite3.Row) -> dict[str, Any]:
        return {column.name: column.to_python(row[column.name]) for column in self.columns}

    async def ensure(self, connection: aiosqlite.Connection) -> None:
        """Create the table if missing, verify its columns, then create any missing indexes."""
        await connection.execute(self.create_table_sql())
        await self.verify(connection)
        for sql in self.create_index_sql():
            await connection.execute(sql)

    async def verify(self, connection: aiosqlite.Connection) -> None:
        rows = await connection.execute_fetchall(f"PRAGMA table_info({self.table})")
        # table_info rows: (cid, name, type, notnull, dflt_value, pk)
        actual = {row[1]: (str(row[2]).upper(), bool(row[3]), bool(row[5])) for row in rows}
        expected = {
            column.name: (
                column.sql_type,
                not column.nullable and not column.primary_key,
                column.primary_key,
            )
            for column in self.columns
        }

        problems: list[str] = []
        for name in expected.keys() - actual.keys():
            problems.append(f"missing column {name!r}")
        for name in actual.keys() - expected.keys():
            problems.append(f"unexpected column {name!r}")
        for name in expected.keys() & actual.keys():
            if expected[name] != actual[name]:
                problems.append(
                    f"column {name!r} is (type, not_null, primary_key)={actual[name]}, "
                    f"expected {expected[name]}"
                )
        if problems:
            raise SchemaMismatchError(
                f"Table {self.table!r} does not match the domain model: "
                + "; ".join(sorted(problems))
            )


TASKS_SCHEMA = TableSchema.for_dataclass(
    Task,
    table="tasks",
    # Every query is scoped to one user and ordered by id, optionally filtered by completion.
    indexes=[("user_id", "id"), ("user_id", "completed", "id")],
)


@asynccontextmanager
async def connect(path: Path) -> AsyncGenerator[aiosqlite.Connection]:
    """Open a connection in autocommit mode, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path, isolation_level=None) as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        yield connection


async def initialize(connection: aiosqlite.Connection) -> None:
    await TASKS_SCHEMA.ensure(connection)


@asynccontextmanager
async def transaction(connection: aiosqlite.Connection) -> AsyncGenerator[None]:
    """Group statements on an autocommit connection into one atomic transaction."""
    await connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        await connection.execute("ROLLBACK")
        raise
    await connection.execute("COMMIT")


async def reset(connection: aiosqlite.Connection) -> None:
    """Delete all tasks and restart ID numbering at 1."""
    await connection.execute(f"DELETE FROM {TASKS_SCHEMA.table}")
    await connection.execute("DELETE FROM sqlite_sequence WHERE name = ?", (TASKS_SCHEMA.table,))


async def drop(connection: aiosqlite.Connection) -> None:
    """Drop the tasks table, its indexes, and its ID sequence, whatever shape the table has."""
    await connection.execute(f"DROP TABLE IF EXISTS {TASKS_SCHEMA.table}")
