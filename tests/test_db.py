"""Tests for schema generation from domain dataclasses and schema verification."""

import dataclasses
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import pytest

from app import db
from app.db import TASKS_SCHEMA, SchemaMismatchError, TableSchema
from app.models import Task

pytestmark = pytest.mark.anyio


@dataclass
class ExtendedTask:
    id: int
    user_id: int
    description: str
    completed: bool
    version: int
    notes: str | None
    priority: int


PRE_VERSIONING_TABLE_SQL = (
    "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, "
    "description TEXT NOT NULL, completed INTEGER NOT NULL CHECK (completed IN (0, 1)))"
)


@pytest.fixture
async def connection(db_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    async with db.connect(db_path) as conn:
        yield conn


async def table_columns(connection: aiosqlite.Connection, table: str) -> list[str]:
    rows = await connection.execute_fetchall(f"PRAGMA table_info({table})")
    return [row[1] for row in rows]


class TestSchemaGeneration:
    def test_task_columns_match_task_fields_in_order(self) -> None:
        assert TASKS_SCHEMA.column_names == tuple(f.name for f in dataclasses.fields(Task))

    def test_task_table_sql(self) -> None:
        assert TASKS_SCHEMA.create_table_sql() == (
            "CREATE TABLE IF NOT EXISTS tasks (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    user_id INTEGER NOT NULL,\n"
            "    description TEXT NOT NULL,\n"
            "    completed INTEGER NOT NULL CHECK (completed IN (0, 1)),\n"
            "    version INTEGER NOT NULL\n"
            ")"
        )

    def test_schema_follows_dataclass_definition(self) -> None:
        schema = TableSchema.for_dataclass(ExtendedTask, table="extended")

        assert schema.column_names == (
            "id",
            "user_id",
            "description",
            "completed",
            "version",
            "notes",
            "priority",
        )
        assert schema.writable_column_names == schema.column_names[1:]
        assert "notes TEXT," in schema.create_table_sql()
        assert "priority INTEGER NOT NULL" in schema.create_table_sql()

    def test_rejects_unsupported_field_type(self) -> None:
        @dataclass
        class WithList:
            id: int
            tags: list[str]

        with pytest.raises(TypeError, match=r"WithList\.tags"):
            TableSchema.for_dataclass(WithList, table="t")

    def test_rejects_dataclass_without_id(self) -> None:
        @dataclass
        class NoId:
            name: str

        with pytest.raises(TypeError, match="'id'"):
            TableSchema.for_dataclass(NoId, table="t")

    def test_rejects_optional_id(self) -> None:
        @dataclass
        class OptionalId:
            id: int | None

        with pytest.raises(TypeError, match="non-optional int"):
            TableSchema.for_dataclass(OptionalId, table="t")


class TestSchemaInitialization:
    async def test_connect_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "tasks.db"

        async with db.connect(path):
            pass

        assert path.exists()

    async def test_initialize_creates_task_table(self, connection: aiosqlite.Connection) -> None:
        await db.initialize(connection)

        assert await table_columns(connection, "tasks") == list(TASKS_SCHEMA.column_names)

    async def test_initialize_is_idempotent(self, connection: aiosqlite.Connection) -> None:
        await db.initialize(connection)
        await connection.execute(
            "INSERT INTO tasks (user_id, description, completed, version) VALUES (1, 'x', 0, 1)"
        )

        await db.initialize(connection)

        rows = await connection.execute_fetchall("SELECT description FROM tasks")
        assert [row[0] for row in rows] == ["x"]

    async def test_verify_detects_missing_column(self, connection: aiosqlite.Connection) -> None:
        await TableSchema.for_dataclass(Task, table="extended").ensure(connection)

        with pytest.raises(SchemaMismatchError) as exc_info:
            await TableSchema.for_dataclass(ExtendedTask, table="extended").ensure(connection)

        assert "missing column 'notes'" in str(exc_info.value)
        assert "missing column 'priority'" in str(exc_info.value)

    async def test_verify_detects_unexpected_column(self, connection: aiosqlite.Connection) -> None:
        await TableSchema.for_dataclass(ExtendedTask, table="tasks").ensure(connection)

        with pytest.raises(SchemaMismatchError, match="unexpected column 'notes'"):
            await db.initialize(connection)

    async def test_verify_detects_changed_column_type(
        self, connection: aiosqlite.Connection
    ) -> None:
        await connection.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER NOT NULL, description TEXT NOT NULL, completed TEXT NOT NULL, "
            "version INTEGER NOT NULL)"
        )

        with pytest.raises(SchemaMismatchError, match="column 'completed'"):
            await db.initialize(connection)

    async def test_verify_detects_changed_nullability(
        self, connection: aiosqlite.Connection
    ) -> None:
        await connection.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER NOT NULL, description TEXT, completed INTEGER NOT NULL, "
            "version INTEGER NOT NULL)"
        )

        with pytest.raises(SchemaMismatchError, match="column 'description'"):
            await db.initialize(connection)

    async def test_pre_ownership_table_is_rejected(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "description TEXT NOT NULL, completed INTEGER NOT NULL)"
        )

        with pytest.raises(SchemaMismatchError, match="missing column 'user_id'"):
            await db.initialize(connection)

    async def test_pre_versioning_table_is_rejected(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(PRE_VERSIONING_TABLE_SQL)

        with pytest.raises(SchemaMismatchError, match="missing column 'version'"):
            await db.initialize(connection)

    async def test_pre_versioning_table_can_be_upgraded_in_place(
        self, connection: aiosqlite.Connection
    ) -> None:
        await connection.execute(PRE_VERSIONING_TABLE_SQL)
        await connection.execute(
            "INSERT INTO tasks (user_id, description, completed) VALUES (1, 'x', 0)"
        )

        await connection.execute("ALTER TABLE tasks ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        await db.initialize(connection)

        rows = await connection.execute_fetchall("SELECT description, version FROM tasks")
        assert [tuple(row) for row in rows] == [("x", 1)]

    async def test_drop_allows_recreating_a_mismatched_table(
        self, connection: aiosqlite.Connection
    ) -> None:
        await connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT)")

        await db.drop(connection)
        await db.initialize(connection)

        assert await table_columns(connection, "tasks") == list(TASKS_SCHEMA.column_names)


class TestIndexes:
    async def test_initialize_creates_declared_indexes(
        self, connection: aiosqlite.Connection
    ) -> None:
        await db.initialize(connection)

        rows = await connection.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'tasks' "
            "AND name LIKE 'ix_%' ORDER BY name"
        )
        assert [row[0] for row in rows] == [
            "ix_tasks_user_id_completed_id",
            "ix_tasks_user_id_id",
        ]

    async def test_user_scoped_listing_uses_an_index(
        self, connection: aiosqlite.Connection
    ) -> None:
        await db.initialize(connection)

        plan = await connection.execute_fetchall(
            "EXPLAIN QUERY PLAN SELECT * FROM tasks WHERE user_id = 1 AND completed = 1 "
            "AND id > 5 ORDER BY id LIMIT 10"
        )

        assert "USING INDEX ix_tasks_user_id_completed_id" in " ".join(row[3] for row in plan)

    def test_rejects_index_on_unknown_field(self) -> None:
        with pytest.raises(TypeError, match="owner"):
            TableSchema.for_dataclass(Task, table="t", indexes=[("owner", "id")])


class TestConstraints:
    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param(
                "INSERT INTO tasks (user_id, description, completed, version) "
                "VALUES (1, NULL, 0, 1)",
                id="null-description",
            ),
            pytest.param(
                "INSERT INTO tasks (user_id, description, completed, version) "
                "VALUES (NULL, 'x', 0, 1)",
                id="null-user",
            ),
            pytest.param(
                "INSERT INTO tasks (user_id, description, completed, version) "
                "VALUES (1, 'x', 2, 1)",
                id="bool",
            ),
            pytest.param(
                "INSERT INTO tasks (user_id, description, completed) VALUES (1, 'x', 0)",
                id="missing-version",
            ),
        ],
    )
    async def test_rejects_invalid_rows(self, connection: aiosqlite.Connection, sql: str) -> None:
        await db.initialize(connection)

        with pytest.raises(sqlite3.IntegrityError):
            await connection.execute(sql)
