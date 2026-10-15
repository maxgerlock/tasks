"""Business logic for tasks."""

from app.models import Task, TaskChanges, TaskNotFoundError, TaskPage, TaskQuery
from app.repository import TaskRepository
from app.schemas import TaskCreate, TaskListParams, TaskUpdate


class TaskService:
    """Task operations on behalf of one user, who can only see and change their own tasks."""

    def __init__(self, repository: TaskRepository, user_id: int) -> None:
        self._repository = repository
        self._user_id = user_id

    async def create_task(self, data: TaskCreate) -> Task:
        return await self._repository.create(
            user_id=self._user_id, description=data.description, completed=False
        )

    async def get_task(self, task_id: int) -> Task:
        task = await self._repository.get(task_id, user_id=self._user_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    async def list_tasks(self, params: TaskListParams) -> TaskPage:
        # Fetch one extra row to learn whether another page exists without a separate count.
        tasks = await self._repository.find(
            TaskQuery(
                user_id=self._user_id,
                completed=params.completed,
                after=params.after,
                before=params.before,
                limit=params.limit + 1,
            )
        )
        items = tasks[: params.limit]
        has_more = len(tasks) > params.limit
        return TaskPage(
            items=items,
            next_cursor=items[-1].id if has_more else None,
            has_more=has_more,
        )

    async def update_task(self, task_id: int, data: TaskUpdate) -> Task:
        changes: TaskChanges = {}
        if data.description is not None:
            changes["description"] = data.description
        if data.completed is not None:
            changes["completed"] = data.completed

        task = await self._repository.update(
            task_id, user_id=self._user_id, expected_version=data.version, **changes
        )
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    async def delete_task(self, task_id: int, version: int) -> None:
        deleted = await self._repository.delete(
            task_id, user_id=self._user_id, expected_version=version
        )
        if not deleted:
            raise TaskNotFoundError(task_id)
