from typing import Protocol, Unpack

from app.models import Task, TaskChanges, TaskQuery


class TaskRepository(Protocol):
    """Task storage. Implementations assign unique IDs that are never reused.

    Every read and write is scoped to one owner: a task owned by someone else behaves exactly
    like a task that does not exist.

    Writes to existing tasks are conditional on ``expected_version``: if the stored task has a
    different version, nothing changes and ``VersionConflictError`` is raised. The check and the
    write happen atomically, so two writes based on the same version cannot both succeed.
    """

    async def create(self, *, user_id: int, description: str, completed: bool) -> Task:
        """Persist a new task at ``INITIAL_VERSION`` and return it with its assigned ID."""
        ...

    async def get(self, task_id: int, *, user_id: int) -> Task | None:
        """Return the user's task with the given ID, or None if they have no such task."""
        ...

    async def find(self, query: TaskQuery) -> list[Task]:
        """Return the user's tasks matching the query, in the order and up to the limit given."""
        ...

    async def update(
        self, task_id: int, *, user_id: int, expected_version: int, **changes: Unpack[TaskChanges]
    ) -> Task | None:
        """Apply changes and increment the version, returning the updated task.

        Returns None if the user has no such task.
        """
        ...

    async def delete(self, task_id: int, *, user_id: int, expected_version: int) -> bool:
        """Delete the user's task and return True, or return False if not found."""
        ...
