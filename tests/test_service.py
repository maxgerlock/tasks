import pytest

from app.models import Task, TaskNotFoundError, TaskPage
from app.repository import TaskRepository
from app.schemas import TaskCreate, TaskListParams, TaskUpdate
from app.service import TaskService
from tests.conftest import OTHER_USER_ID, USER_ID

pytestmark = pytest.mark.anyio


async def test_create_task_starts_incomplete(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    assert task.description == "Buy groceries"
    assert task.completed is False


async def test_create_task_persists_to_repository(
    service: TaskService, repository: TaskRepository
) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    assert await repository.get(task.id, user_id=USER_ID) == task


async def test_get_task_returns_existing_task(service: TaskService) -> None:
    created = await service.create_task(TaskCreate(description="Buy groceries"))

    assert await service.get_task(created.id) == created


async def test_get_task_raises_for_missing_task(service: TaskService) -> None:
    with pytest.raises(TaskNotFoundError) as exc_info:
        await service.get_task(42)

    assert exc_info.value.task_id == 42


async def test_list_tasks_returns_all_tasks_when_they_fit(service: TaskService) -> None:
    first = await service.create_task(TaskCreate(description="first"))
    second = await service.create_task(TaskCreate(description="second"))

    page = await service.list_tasks(TaskListParams())

    assert page == TaskPage(items=[first, second], next_cursor=None, has_more=False)


async def test_list_tasks_is_empty_initially(service: TaskService) -> None:
    page = await service.list_tasks(TaskListParams())

    assert page == TaskPage(items=[], next_cursor=None, has_more=False)


async def create_many(service: TaskService, count: int) -> None:
    for i in range(count):
        await service.create_task(TaskCreate(description=f"task {i}"))


@pytest.mark.parametrize(
    ("params", "expected_ids", "next_cursor"),
    [
        pytest.param(TaskListParams(limit=2), [1, 2], 2, id="first-page"),
        pytest.param(TaskListParams(after=2, limit=2), [3, 4], 4, id="middle-page"),
        pytest.param(TaskListParams(after=3, limit=2), [4, 5], None, id="exact-last-page"),
        pytest.param(TaskListParams(after=4, limit=2), [5], None, id="short-last-page"),
        pytest.param(TaskListParams(before=6, limit=2), [5, 4], 4, id="before-first-page"),
        pytest.param(TaskListParams(before=3, limit=2), [2, 1], None, id="before-last-page"),
        pytest.param(TaskListParams(limit=5), [1, 2, 3, 4, 5], None, id="limit-equals-total"),
    ],
)
async def test_list_tasks_pages(
    service: TaskService,
    params: TaskListParams,
    expected_ids: list[int],
    next_cursor: int | None,
) -> None:
    await create_many(service, 5)

    page = await service.list_tasks(params)

    assert [task.id for task in page.items] == expected_ids
    assert page.next_cursor == next_cursor
    assert page.has_more is (next_cursor is not None)


async def test_list_tasks_filters_by_completion(service: TaskService) -> None:
    await create_many(service, 4)
    await service.update_task(2, TaskUpdate(version=1, completed=True))
    await service.update_task(3, TaskUpdate(version=1, completed=True))

    completed = await service.list_tasks(TaskListParams(completed=True))
    incomplete = await service.list_tasks(TaskListParams(completed=False))

    assert [task.id for task in completed.items] == [2, 3]
    assert [task.id for task in incomplete.items] == [1, 4]


async def test_update_task_changes_description_only(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    updated = await service.update_task(task.id, TaskUpdate(version=1, description="Buy coffee"))

    assert updated == Task(
        id=task.id, user_id=USER_ID, description="Buy coffee", completed=False, version=2
    )


async def test_update_task_changes_completed_only(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    updated = await service.update_task(task.id, TaskUpdate(version=1, completed=True))

    assert updated == Task(
        id=task.id, user_id=USER_ID, description="Buy groceries", completed=True, version=2
    )


async def test_update_task_changes_both_fields(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    updated = await service.update_task(
        task.id, TaskUpdate(version=1, description="Buy coffee", completed=True)
    )

    assert updated == Task(
        id=task.id, user_id=USER_ID, description="Buy coffee", completed=True, version=2
    )


async def test_update_task_can_mark_incomplete(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))
    await service.update_task(task.id, TaskUpdate(version=1, completed=True))

    updated = await service.update_task(task.id, TaskUpdate(version=2, completed=False))

    assert updated.completed is False


async def test_update_task_is_persisted(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    updated = await service.update_task(task.id, TaskUpdate(version=1, completed=True))

    assert await service.get_task(task.id) == updated


async def test_update_task_raises_for_missing_task(service: TaskService) -> None:
    with pytest.raises(TaskNotFoundError) as exc_info:
        await service.update_task(42, TaskUpdate(version=1, completed=True))

    assert exc_info.value.task_id == 42


async def test_delete_task_removes_task(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    await service.delete_task(task.id, version=1)

    with pytest.raises(TaskNotFoundError):
        await service.get_task(task.id)
    assert (await service.list_tasks(TaskListParams())).items == []


async def test_delete_task_raises_for_missing_task(service: TaskService) -> None:
    with pytest.raises(TaskNotFoundError) as exc_info:
        await service.delete_task(42, version=1)

    assert exc_info.value.task_id == 42


async def test_create_task_assigns_current_user(service: TaskService) -> None:
    task = await service.create_task(TaskCreate(description="Buy groceries"))

    assert task.user_id == USER_ID


async def test_users_are_isolated(service: TaskService, repository: TaskRepository) -> None:
    other = TaskService(repository, OTHER_USER_ID)
    mine = await service.create_task(TaskCreate(description="mine"))
    theirs = await other.create_task(TaskCreate(description="theirs"))

    assert (await service.list_tasks(TaskListParams())).items == [mine]
    assert (await other.list_tasks(TaskListParams())).items == [theirs]
    with pytest.raises(TaskNotFoundError):
        await other.get_task(mine.id)
    with pytest.raises(TaskNotFoundError):
        await other.update_task(mine.id, TaskUpdate(version=1, completed=True))
    with pytest.raises(TaskNotFoundError):
        await other.delete_task(mine.id, version=1)
    assert await service.get_task(mine.id) == mine
