import dataclasses
from typing import Unpack, get_type_hints

from app.models import INITIAL_VERSION, Task, TaskChanges, TaskQuery, VersionConflictError

_UPDATABLE = frozenset(get_type_hints(TaskChanges))


class InMemoryTaskRepository:
    """Process-local task store. Data is lost when the process exits.

    IDs are sequential starting at 1, shared across users, and never reused, even after
    deletion. No method awaits, so each operation (including its version check) runs
    atomically with respect to other coroutines.
    """

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    async def create(self, *, user_id: int, description: str, completed: bool) -> Task:
        task = Task(
            id=self._next_id,
            user_id=user_id,
            description=description,
            completed=completed,
            version=INITIAL_VERSION,
        )
        self._tasks[task.id] = task
        self._next_id += 1
        return task

    async def get(self, task_id: int, *, user_id: int) -> Task | None:
        return self._owned(task_id, user_id)

    async def find(self, query: TaskQuery) -> list[Task]:
        matches = [
            task
            for task in self._tasks.values()
            if task.user_id == query.user_id
            and (query.completed is None or task.completed == query.completed)
            and (query.after is None or task.id > query.after)
            and (query.before is None or task.id < query.before)
        ]
        matches.sort(key=lambda task: task.id, reverse=query.descending)
        return matches if query.limit is None else matches[: query.limit]

    async def update(
        self, task_id: int, *, user_id: int, expected_version: int, **changes: Unpack[TaskChanges]
    ) -> Task | None:
        unknown = changes.keys() - _UPDATABLE
        if unknown:
            raise ValueError(f"Not writable columns: {sorted(unknown)}")
        task = self._owned(task_id, user_id)
        if task is None:
            return None
        _check_version(task, expected_version)
        updated = dataclasses.replace(task, **changes, version=task.version + 1)
        self._tasks[task_id] = updated
        return updated

    async def delete(self, task_id: int, *, user_id: int, expected_version: int) -> bool:
        task = self._owned(task_id, user_id)
        if task is None:
            return False
        _check_version(task, expected_version)
        del self._tasks[task_id]
        return True

    def _owned(self, task_id: int, user_id: int) -> Task | None:
        task = self._tasks.get(task_id)
        return task if task is not None and task.user_id == user_id else None


def _check_version(task: Task, expected_version: int) -> None:
    if task.version != expected_version:
        raise VersionConflictError(task.id, expected_version, task.version)
