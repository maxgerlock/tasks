"""Core domain entities and errors, independent of HTTP and storage concerns."""

from dataclasses import dataclass
from typing import TypedDict

INITIAL_VERSION = 1


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    user_id: int
    description: str
    completed: bool
    # Starts at INITIAL_VERSION and increases by one with every successful update.
    version: int


class TaskChanges(TypedDict, total=False):
    """Fields that may change on an existing task. Absent keys are left unchanged.

    ``id``, ``user_id`` and ``version`` are deliberately absent: identity and ownership are
    fixed, and the version is maintained by the repository.
    """

    description: str
    completed: bool


@dataclass(frozen=True, slots=True)
class TaskQuery:
    """Selects one user's tasks for listing. Cursors are exclusive ID bounds.

    Results are ordered by ID descending when ``before`` is set, otherwise ascending.
    ``limit=None`` means no limit.
    """

    user_id: int
    completed: bool | None = None
    after: int | None = None
    before: int | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.after is not None and self.before is not None:
            raise ValueError("Only one of 'after' and 'before' may be set")
        if self.limit is not None and self.limit < 0:
            raise ValueError("'limit' must not be negative")

    @property
    def descending(self) -> bool:
        return self.before is not None


@dataclass(frozen=True, slots=True)
class TaskPage:
    items: list[Task]
    next_cursor: int | None
    has_more: bool


class TaskNotFoundError(Exception):
    def __init__(self, task_id: int) -> None:
        super().__init__(f"Task {task_id} not found")
        self.task_id = task_id


class VersionConflictError(Exception):
    """The task changed since the caller read it, so their write was not applied."""

    def __init__(self, task_id: int, expected_version: int, current_version: int) -> None:
        super().__init__(f"Task {task_id} is at version {current_version}, not {expected_version}")
        self.task_id = task_id
        self.expected_version = expected_version
        self.current_version = current_version
