"""Contract tests run against every TaskRepository implementation."""

import asyncio
import dataclasses
from typing import Any, cast

import pytest

from app.models import Task, TaskQuery, VersionConflictError
from app.repository import TaskRepository
from tests.conftest import OTHER_USER_ID, USER_ID

pytestmark = pytest.mark.anyio


async def create(
    repository: TaskRepository,
    description: str = "Buy groceries",
    *,
    completed: bool = False,
    user_id: int = USER_ID,
) -> Task:
    return await repository.create(user_id=user_id, description=description, completed=completed)


async def all_tasks(repository: TaskRepository, user_id: int = USER_ID) -> list[Task]:
    return await repository.find(TaskQuery(user_id=user_id))


async def test_create_returns_task_with_given_fields(repository: TaskRepository) -> None:
    task = await create(repository)

    assert task == Task(
        id=1, user_id=USER_ID, description="Buy groceries", completed=False, version=1
    )


async def test_create_round_trips_completed_true(repository: TaskRepository) -> None:
    task = await create(repository, "Done already", completed=True)

    assert await repository.get(task.id, user_id=USER_ID) == Task(
        id=1, user_id=USER_ID, description="Done already", completed=True, version=1
    )


async def test_create_assigns_sequential_ids_starting_at_one(repository: TaskRepository) -> None:
    ids = [(await create(repository, f"task {i}")).id for i in range(3)]

    assert ids == [1, 2, 3]


async def test_get_returns_created_task(repository: TaskRepository) -> None:
    created = await create(repository)

    assert await repository.get(created.id, user_id=USER_ID) == created


async def test_get_returns_none_for_missing_task(repository: TaskRepository) -> None:
    assert await repository.get(1, user_id=USER_ID) is None


async def test_find_is_empty_initially(repository: TaskRepository) -> None:
    assert await all_tasks(repository) == []


async def test_find_without_criteria_returns_all_tasks_in_id_order(
    repository: TaskRepository,
) -> None:
    created = [await create(repository, f"task {i}") for i in range(3)]

    assert await all_tasks(repository) == created


async def test_find_returns_a_copy(repository: TaskRepository) -> None:
    await create(repository)

    (await all_tasks(repository)).clear()

    assert len(await all_tasks(repository)) == 1


async def test_preserves_unicode_and_quotes(repository: TaskRepository) -> None:
    description = "Café ☕ — \"quoted\" 'single'; DROP TABLE tasks; --"

    task = await create(repository, description)

    assert await repository.get(task.id, user_id=USER_ID) == task
    assert task.description == description


async def test_concurrent_creates_assign_unique_ids(repository: TaskRepository) -> None:
    count = 200

    tasks = await asyncio.gather(*(create(repository, f"task {i}") for i in range(count)))

    assert sorted(task.id for task in tasks) == list(range(1, count + 1))
    assert len(await all_tasks(repository)) == count


async def test_update_changes_only_supplied_fields(repository: TaskRepository) -> None:
    task = await create(repository)

    updated = await repository.update(task.id, user_id=USER_ID, expected_version=1, completed=True)

    assert updated == Task(
        id=task.id, user_id=USER_ID, description="Buy groceries", completed=True, version=2
    )


async def test_update_can_change_multiple_fields(repository: TaskRepository) -> None:
    task = await create(repository)

    updated = await repository.update(
        task.id, user_id=USER_ID, expected_version=1, description="Buy coffee", completed=True
    )

    assert updated == Task(
        id=task.id, user_id=USER_ID, description="Buy coffee", completed=True, version=2
    )


async def test_update_with_no_changes_still_increments_version(
    repository: TaskRepository,
) -> None:
    task = await create(repository)

    updated = await repository.update(task.id, user_id=USER_ID, expected_version=1)

    assert updated == dataclasses.replace(task, version=2)


async def test_update_is_persisted(repository: TaskRepository) -> None:
    task = await create(repository)

    updated = await repository.update(
        task.id, user_id=USER_ID, expected_version=1, description="Buy coffee"
    )

    assert await repository.get(task.id, user_id=USER_ID) == updated


async def test_update_does_not_affect_other_tasks(repository: TaskRepository) -> None:
    first = await create(repository, "first")
    second = await create(repository, "second")

    await repository.update(first.id, user_id=USER_ID, expected_version=1, completed=True)

    assert await repository.get(second.id, user_id=USER_ID) == second


async def test_update_returns_none_for_missing_task(repository: TaskRepository) -> None:
    assert await repository.update(1, user_id=USER_ID, expected_version=1, completed=True) is None


@pytest.mark.parametrize("field", ["id", "version"])
async def test_update_rejects_repository_managed_fields(
    repository: TaskRepository, field: str
) -> None:
    task = await create(repository)

    with pytest.raises(ValueError, match=field):
        await repository.update(
            task.id, user_id=USER_ID, expected_version=1, **cast(Any, {field: 99})
        )

    assert await repository.get(task.id, user_id=USER_ID) == task


async def test_update_cannot_smuggle_a_new_owner(repository: TaskRepository) -> None:
    task = await create(repository)
    smuggled = cast(Any, {"user_id": OTHER_USER_ID})

    with pytest.raises(TypeError, match="user_id"):
        await repository.update(task.id, user_id=USER_ID, expected_version=1, **smuggled)

    assert await repository.get(task.id, user_id=USER_ID) == task


async def test_delete_removes_task(repository: TaskRepository) -> None:
    task = await create(repository)

    assert await repository.delete(task.id, user_id=USER_ID, expected_version=1) is True
    assert await repository.get(task.id, user_id=USER_ID) is None
    assert await all_tasks(repository) == []


async def test_delete_returns_false_for_missing_task(repository: TaskRepository) -> None:
    assert await repository.delete(1, user_id=USER_ID, expected_version=1) is False


async def test_delete_twice_returns_false_the_second_time(repository: TaskRepository) -> None:
    task = await create(repository)
    await repository.delete(task.id, user_id=USER_ID, expected_version=1)

    assert await repository.delete(task.id, user_id=USER_ID, expected_version=1) is False


async def test_update_after_delete_returns_none(repository: TaskRepository) -> None:
    task = await create(repository)
    await repository.delete(task.id, user_id=USER_ID, expected_version=1)

    assert (
        await repository.update(task.id, user_id=USER_ID, expected_version=1, completed=True)
        is None
    )


async def test_ids_are_not_reused_after_delete(repository: TaskRepository) -> None:
    first = await create(repository, "first")
    second = await create(repository, "second")
    await repository.delete(second.id, user_id=USER_ID, expected_version=1)
    await repository.delete(first.id, user_id=USER_ID, expected_version=1)

    assert (await create(repository, "third")).id == 3


class TestVersioning:
    async def test_new_tasks_start_at_version_one(self, repository: TaskRepository) -> None:
        assert (await create(repository)).version == 1

    async def test_each_update_increments_version(self, repository: TaskRepository) -> None:
        task = await create(repository)

        v2 = await repository.update(task.id, user_id=USER_ID, expected_version=1, completed=True)
        v3 = await repository.update(
            task.id, user_id=USER_ID, expected_version=2, description="Buy coffee"
        )

        assert v2 is not None and v2.version == 2
        assert v3 == Task(
            id=task.id, user_id=USER_ID, description="Buy coffee", completed=True, version=3
        )
        assert await repository.get(task.id, user_id=USER_ID) == v3

    @pytest.mark.parametrize("expected_version", [1, 3], ids=["stale", "ahead"])
    async def test_update_with_wrong_version_conflicts_and_changes_nothing(
        self, repository: TaskRepository, expected_version: int
    ) -> None:
        task = await create(repository)
        current = await repository.update(
            task.id, user_id=USER_ID, expected_version=1, completed=True
        )

        with pytest.raises(VersionConflictError) as exc_info:
            await repository.update(
                task.id, user_id=USER_ID, expected_version=expected_version, description="lost"
            )

        assert (exc_info.value.expected_version, exc_info.value.current_version) == (
            expected_version,
            2,
        )
        assert await repository.get(task.id, user_id=USER_ID) == current

    async def test_delete_with_stale_version_conflicts_and_keeps_task(
        self, repository: TaskRepository
    ) -> None:
        task = await create(repository)
        current = await repository.update(
            task.id, user_id=USER_ID, expected_version=1, completed=True
        )

        with pytest.raises(VersionConflictError):
            await repository.delete(task.id, user_id=USER_ID, expected_version=1)

        assert await repository.get(task.id, user_id=USER_ID) == current

    async def test_delete_with_current_version_succeeds(self, repository: TaskRepository) -> None:
        task = await create(repository)
        await repository.update(task.id, user_id=USER_ID, expected_version=1, completed=True)

        assert await repository.delete(task.id, user_id=USER_ID, expected_version=2) is True

    async def test_missing_task_is_not_a_conflict(self, repository: TaskRepository) -> None:
        assert (
            await repository.update(9, user_id=USER_ID, expected_version=5, completed=True) is None
        )
        assert await repository.delete(9, user_id=USER_ID, expected_version=5) is False

    async def test_other_users_task_is_not_found_regardless_of_version(
        self, repository: TaskRepository
    ) -> None:
        theirs = await create(repository, user_id=OTHER_USER_ID)

        for version in (1, 2):
            assert (
                await repository.update(
                    theirs.id, user_id=USER_ID, expected_version=version, completed=True
                )
                is None
            )
            assert (
                await repository.delete(theirs.id, user_id=USER_ID, expected_version=version)
                is False
            )
        assert await repository.get(theirs.id, user_id=OTHER_USER_ID) == theirs

    async def test_concurrent_updates_from_same_version_have_one_winner(
        self, repository: TaskRepository
    ) -> None:
        task = await create(repository, "original")

        results = await asyncio.gather(
            *(
                repository.update(
                    task.id, user_id=USER_ID, expected_version=1, description=f"writer {i}"
                )
                for i in range(50)
            ),
            return_exceptions=True,
        )

        winners = [result for result in results if isinstance(result, Task)]
        conflicts = [result for result in results if isinstance(result, VersionConflictError)]
        assert len(winners) == 1
        assert len(conflicts) == 49
        assert await repository.get(task.id, user_id=USER_ID) == winners[0]
        assert winners[0].version == 2

    async def test_concurrent_update_and_delete_from_same_version_have_one_winner(
        self, repository: TaskRepository
    ) -> None:
        task = await create(repository)

        updated, deleted = await asyncio.gather(
            repository.update(task.id, user_id=USER_ID, expected_version=1, completed=True),
            repository.delete(task.id, user_id=USER_ID, expected_version=1),
            return_exceptions=True,
        )

        outcomes = {
            "updated": isinstance(updated, Task),
            "deleted": deleted is True,
        }
        assert sum(outcomes.values()) == 1
        assert isinstance(updated, VersionConflictError) or isinstance(
            deleted, VersionConflictError
        )

    async def test_read_modify_write_with_retries_loses_no_updates(
        self, repository: TaskRepository
    ) -> None:
        task = await create(repository, "")
        writers = 20

        async def append(marker: str) -> None:
            while True:
                current = await repository.get(task.id, user_id=USER_ID)
                assert current is not None
                try:
                    await repository.update(
                        task.id,
                        user_id=USER_ID,
                        expected_version=current.version,
                        description=current.description + marker,
                    )
                    return
                except VersionConflictError:
                    continue

        await asyncio.gather(*(append(f"[{i}]") for i in range(writers)))

        final = await repository.get(task.id, user_id=USER_ID)
        assert final is not None
        assert final.version == writers + 1
        assert sorted(final.description.strip("[]").split("][")) == sorted(
            str(i) for i in range(writers)
        )


class TestOwnership:
    async def test_ids_are_unique_across_users(self, repository: TaskRepository) -> None:
        mine = await create(repository, user_id=USER_ID)
        theirs = await create(repository, user_id=OTHER_USER_ID)

        assert (mine.id, theirs.id) == (1, 2)
        assert theirs.user_id == OTHER_USER_ID

    async def test_get_hides_other_users_task(self, repository: TaskRepository) -> None:
        theirs = await create(repository, user_id=OTHER_USER_ID)

        assert await repository.get(theirs.id, user_id=USER_ID) is None
        assert await repository.get(theirs.id, user_id=OTHER_USER_ID) == theirs

    async def test_update_cannot_touch_other_users_task(self, repository: TaskRepository) -> None:
        theirs = await create(repository, user_id=OTHER_USER_ID)

        assert (
            await repository.update(theirs.id, user_id=USER_ID, expected_version=1, completed=True)
            is None
        )
        assert await repository.get(theirs.id, user_id=OTHER_USER_ID) == theirs

    async def test_delete_cannot_touch_other_users_task(self, repository: TaskRepository) -> None:
        theirs = await create(repository, user_id=OTHER_USER_ID)

        assert await repository.delete(theirs.id, user_id=USER_ID, expected_version=1) is False
        assert await repository.get(theirs.id, user_id=OTHER_USER_ID) == theirs

    async def test_find_returns_only_own_tasks(self, repository: TaskRepository) -> None:
        for i in range(6):
            await create(repository, f"task {i}", user_id=USER_ID if i % 2 else OTHER_USER_ID)

        assert [task.id for task in await all_tasks(repository, USER_ID)] == [2, 4, 6]
        assert [task.id for task in await all_tasks(repository, OTHER_USER_ID)] == [1, 3, 5]

    async def test_find_limit_counts_only_own_tasks(self, repository: TaskRepository) -> None:
        for i in range(6):
            await create(repository, f"task {i}", user_id=USER_ID if i % 2 else OTHER_USER_ID)

        page = await repository.find(TaskQuery(user_id=USER_ID, after=2, limit=2))

        assert [task.id for task in page] == [4, 6]


class TestFind:
    """User 1 owns tasks 1-10 (even IDs completed); user 2 owns task 11."""

    @pytest.fixture(autouse=True)
    async def tasks(self, repository: TaskRepository) -> None:
        for i in range(1, 11):
            await create(repository, f"task {i}", completed=i % 2 == 0)
        await create(repository, "someone else's", completed=True, user_id=OTHER_USER_ID)

    @pytest.mark.parametrize(
        ("query", "expected_ids"),
        [
            pytest.param(TaskQuery(USER_ID, completed=True), [2, 4, 6, 8, 10], id="completed"),
            pytest.param(TaskQuery(USER_ID, completed=False), [1, 3, 5, 7, 9], id="incomplete"),
            pytest.param(TaskQuery(USER_ID, after=7), [8, 9, 10], id="after-exclusive-asc"),
            pytest.param(TaskQuery(USER_ID, before=4), [3, 2, 1], id="before-exclusive-desc"),
            pytest.param(TaskQuery(USER_ID, limit=3), [1, 2, 3], id="limit"),
            pytest.param(TaskQuery(USER_ID, limit=0), [], id="limit-zero"),
            pytest.param(TaskQuery(USER_ID, after=2, limit=2), [3, 4], id="after-limit"),
            pytest.param(TaskQuery(USER_ID, before=9, limit=3), [8, 7, 6], id="before-limit"),
            pytest.param(
                TaskQuery(USER_ID, completed=True, after=3, limit=2), [4, 6], id="filter-after"
            ),
            pytest.param(
                TaskQuery(USER_ID, completed=False, before=8, limit=2), [7, 5], id="filter-before"
            ),
            pytest.param(TaskQuery(USER_ID, after=10), [], id="after-last"),
            pytest.param(TaskQuery(USER_ID, before=1), [], id="before-first"),
            pytest.param(TaskQuery(USER_ID, after=0, limit=1), [1], id="after-zero"),
            pytest.param(TaskQuery(USER_ID, before=100, limit=2), [10, 9], id="before-past-end"),
        ],
    )
    async def test_find(
        self, repository: TaskRepository, query: TaskQuery, expected_ids: list[int]
    ) -> None:
        assert [task.id for task in await repository.find(query)] == expected_ids

    async def test_cursor_need_not_be_an_existing_id(self, repository: TaskRepository) -> None:
        await repository.delete(5, user_id=USER_ID, expected_version=1)

        after = await repository.find(TaskQuery(USER_ID, after=5, limit=2))
        before = await repository.find(TaskQuery(USER_ID, before=5, limit=2))

        assert [task.id for task in after] == [6, 7]
        assert [task.id for task in before] == [4, 3]


def test_query_rejects_both_cursors() -> None:
    with pytest.raises(ValueError, match="after"):
        TaskQuery(USER_ID, after=1, before=5)


def test_query_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        TaskQuery(USER_ID, limit=-1)
