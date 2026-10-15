import sqlite3
from typing import Any, Unpack, get_type_hints

import aiosqlite

from app.db import TASKS_SCHEMA
from app.models import INITIAL_VERSION, Task, TaskChanges, TaskQuery, VersionConflictError


class SQLiteTaskRepository:
    """Task store backed by SQLite.

    Expects an autocommit connection whose schema has been initialized. Every operation is a
    single SQL statement, so each one is atomic without explicit transactions.
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._schema = TASKS_SCHEMA
        self._table = self._schema.table
        self._returning = ", ".join(self._schema.column_names)
        # Column names are interpolated into SQL, so writes are limited to these known names.
        self._insertable = frozenset(self._schema.writable_column_names)
        self._updatable = frozenset(get_type_hints(TaskChanges))
        assert self._updatable <= self._insertable, "TaskChanges must name task columns"

    async def create(self, *, user_id: int, description: str, completed: bool) -> Task:
        values: dict[str, Any] = {
            "user_id": user_id,
            "description": description,
            "completed": completed,
            "version": INITIAL_VERSION,
        }
        self._check_columns(values, self._insertable)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        row = await self._fetch_one_or_none(
            f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders}) "
            f"RETURNING {self._returning}",
            tuple(values.values()),
        )
        assert row is not None, "INSERT ... RETURNING always yields a row"
        return self._to_task(row)

    async def get(self, task_id: int, *, user_id: int) -> Task | None:
        row = await self._fetch_one_or_none(
            f"SELECT {self._returning} FROM {self._table} WHERE id = ? AND user_id = ?",
            (task_id, user_id),
        )
        return None if row is None else self._to_task(row)

    async def find(self, query: TaskQuery) -> list[Task]:
        conditions = ["user_id = ?"]
        parameters: list[Any] = [query.user_id]
        if query.completed is not None:
            conditions.append("completed = ?")
            parameters.append(query.completed)
        if query.after is not None:
            conditions.append("id > ?")
            parameters.append(query.after)
        if query.before is not None:
            conditions.append("id < ?")
            parameters.append(query.before)

        sql = f"SELECT {self._returning} FROM {self._table} WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id DESC" if query.descending else " ORDER BY id ASC"
        if query.limit is not None:
            sql += " LIMIT ?"
            parameters.append(query.limit)

        rows = await self._connection.execute_fetchall(sql, parameters)
        return [self._to_task(row) for row in rows]

    async def update(
        self, task_id: int, *, user_id: int, expected_version: int, **changes: Unpack[TaskChanges]
    ) -> Task | None:
        values: dict[str, Any] = dict(changes)
        self._check_columns(values, self._updatable)
        assignments = "".join(f"{name} = ?, " for name in values) + "version = version + 1"
        row = await self._fetch_one_or_none(
            f"UPDATE {self._table} SET {assignments} "
            f"WHERE id = ? AND user_id = ? AND version = ? RETURNING {self._returning}",
            (*values.values(), task_id, user_id, expected_version),
        )
        if row is not None:
            return self._to_task(row)
        await self._raise_if_version_conflict(task_id, user_id, expected_version)
        return None

    async def delete(self, task_id: int, *, user_id: int, expected_version: int) -> bool:
        row = await self._fetch_one_or_none(
            f"DELETE FROM {self._table} WHERE id = ? AND user_id = ? AND version = ? RETURNING id",
            (task_id, user_id, expected_version),
        )
        if row is not None:
            return True
        await self._raise_if_version_conflict(task_id, user_id, expected_version)
        return False

    async def _raise_if_version_conflict(
        self, task_id: int, user_id: int, expected_version: int
    ) -> None:
        """Explain why a conditional write matched no row: a stale version, or no such task.

        The write itself was atomic; this follow-up read only chooses which error to report.
        """
        current = await self.get(task_id, user_id=user_id)
        if current is not None:
            raise VersionConflictError(task_id, expected_version, current.version)

    @staticmethod
    def _check_columns(values: dict[str, Any], allowed: frozenset[str]) -> None:
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"Not writable columns: {sorted(unknown)}")

    async def _fetch_one_or_none(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> sqlite3.Row | None:
        rows = list(await self._connection.execute_fetchall(sql, parameters))
        return rows[0] if rows else None

    def _to_task(self, row: sqlite3.Row) -> Task:
        return Task(**self._schema.row_to_values(row))
