"""GET /tasks filtering and cursor pagination."""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.repository import TaskRepository
from app.schemas import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from tests.conftest import USER_ID

pytestmark = pytest.mark.anyio


async def add_tasks(app: FastAPI, count: int) -> None:
    """Create tasks 1..count directly in an empty store; even IDs are completed."""
    repository: TaskRepository = app.state.task_repository
    for task_id in range(1, count + 1):
        task = await repository.create(
            user_id=USER_ID, description=f"task {task_id}", completed=task_id % 2 == 0
        )
        assert task.id == task_id


async def get_page(client: AsyncClient, **params: str | int) -> dict[str, Any]:
    response = await client.get("/tasks", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def ids(page: dict[str, Any]) -> list[int]:
    return [task["id"] for task in page["items"]]


async def collect(
    client: AsyncClient, direction: str, start: int, **params: str | int
) -> list[int]:
    """Follow next_cursor until exhausted and return every ID seen, in order."""
    seen: list[int] = []
    cursor = start
    while True:
        page = await get_page(client, **{direction: cursor}, **params)
        seen.extend(ids(page))
        if not page["has_more"]:
            assert page["next_cursor"] is None
            return seen
        cursor = page["next_cursor"]


class TestDefaults:
    async def test_default_limit_applies_without_parameters(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, DEFAULT_PAGE_LIMIT + 5)

        page = await get_page(client)

        assert ids(page) == list(range(1, DEFAULT_PAGE_LIMIT + 1))
        assert page["next_cursor"] == DEFAULT_PAGE_LIMIT
        assert page["has_more"] is True

    async def test_exactly_one_full_page_has_no_more(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, DEFAULT_PAGE_LIMIT)

        page = await get_page(client)

        assert len(page["items"]) == DEFAULT_PAGE_LIMIT
        assert page["next_cursor"] is None
        assert page["has_more"] is False


class TestAfter:
    async def test_is_exclusive_and_ascending(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 10)

        page = await get_page(client, after=5, limit=3)

        assert page == {
            "items": [
                {"id": 6, "description": "task 6", "completed": True, "version": 1},
                {"id": 7, "description": "task 7", "completed": False, "version": 1},
                {"id": 8, "description": "task 8", "completed": True, "version": 1},
            ],
            "next_cursor": 8,
            "has_more": True,
        }

    async def test_last_page(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 10)

        page = await get_page(client, after=8, limit=5)

        assert ids(page) == [9, 10]
        assert page["next_cursor"] is None
        assert page["has_more"] is False

    async def test_zero_starts_from_the_beginning(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 3)

        assert ids(await get_page(client, after=0)) == [1, 2, 3]

    async def test_past_the_end_is_empty(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 3)

        page = await get_page(client, after=3)

        assert page == {"items": [], "next_cursor": None, "has_more": False}


class TestBefore:
    async def test_is_exclusive_and_descending(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 10)

        page = await get_page(client, before=10, limit=3)

        assert ids(page) == [9, 8, 7]
        assert page["next_cursor"] == 7
        assert page["has_more"] is True

    async def test_last_page(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 10)

        page = await get_page(client, before=4, limit=10)

        assert ids(page) == [3, 2, 1]
        assert page["next_cursor"] is None
        assert page["has_more"] is False

    async def test_beyond_newest_starts_from_newest(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, 10)

        assert ids(await get_page(client, before=1000, limit=3)) == [10, 9, 8]

    @pytest.mark.parametrize("before", [0, 1])
    async def test_at_or_below_first_id_is_empty(
        self, app: FastAPI, client: AsyncClient, before: int
    ) -> None:
        await add_tasks(app, 3)

        page = await get_page(client, before=before)

        assert page == {"items": [], "next_cursor": None, "has_more": False}


class TestWalkingPages:
    async def test_forward_walk_visits_every_task_once(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, 55)

        assert await collect(client, "after", 0, limit=20) == list(range(1, 56))

    async def test_backward_walk_visits_every_task_once(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, 55)

        assert await collect(client, "before", 56, limit=20) == list(range(55, 0, -1))

    async def test_walk_with_filter(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 25)

        completed = await collect(client, "after", 0, limit=4, completed="true")

        assert completed == list(range(2, 26, 2))

    async def test_cursor_survives_concurrent_changes(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        await add_tasks(app, 10)
        first = await get_page(client, limit=4)

        await client.delete("/tasks/4", params={"version": 1})
        await client.delete("/tasks/6", params={"version": 1})
        await client.post("/tasks", json={"description": "new"})

        rest = await collect(client, "after", first["next_cursor"], limit=4)

        assert ids(first) == [1, 2, 3, 4]
        assert rest == [5, 7, 8, 9, 10, 11]


class TestLimit:
    async def test_limit_one(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 3)

        page = await get_page(client, limit=1)

        assert ids(page) == [1]
        assert page["next_cursor"] == 1

    async def test_maximum_limit(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, MAX_PAGE_LIMIT + 10)

        page = await get_page(client, limit=MAX_PAGE_LIMIT)

        assert len(page["items"]) == MAX_PAGE_LIMIT
        assert page["next_cursor"] == MAX_PAGE_LIMIT
        assert page["has_more"] is True

    @pytest.mark.parametrize("limit", ["0", "-1", str(MAX_PAGE_LIMIT + 1), "abc", "1.5", ""])
    async def test_rejects_invalid_limit(self, client: AsyncClient, limit: str) -> None:
        response = await client.get("/tasks", params={"limit": limit})

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", "limit"]


class TestCompletedFilter:
    @pytest.mark.parametrize(
        ("value", "expected_ids"),
        [("true", [2, 4, 6]), ("false", [1, 3, 5])],
    )
    async def test_filters(
        self, app: FastAPI, client: AsyncClient, value: str, expected_ids: list[int]
    ) -> None:
        await add_tasks(app, 6)

        page = await get_page(client, completed=value)

        assert ids(page) == expected_ids
        assert all(task["completed"] is (value == "true") for task in page["items"])

    async def test_omitted_returns_both(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 6)

        assert ids(await get_page(client)) == [1, 2, 3, 4, 5, 6]

    async def test_has_more_accounts_for_filter(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 11)

        page = await get_page(client, completed="true", limit=5)

        assert ids(page) == [2, 4, 6, 8, 10]
        assert page["has_more"] is False
        assert page["next_cursor"] is None

    async def test_combines_with_before(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 10)

        page = await get_page(client, completed="false", before=8, limit=2)

        assert ids(page) == [7, 5]
        assert page["next_cursor"] == 5
        assert page["has_more"] is True

    async def test_reflects_updates(self, app: FastAPI, client: AsyncClient) -> None:
        await add_tasks(app, 3)

        await client.patch("/tasks/1", json={"version": 1, "completed": True})

        assert ids(await get_page(client, completed="true")) == [1, 2]

    @pytest.mark.parametrize("value", ["1", "0", "yes", "no", "on", "True", "FALSE", "", "null"])
    async def test_rejects_non_boolean_values(self, client: AsyncClient, value: str) -> None:
        response = await client.get("/tasks", params={"completed": value})

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", "completed"]


class TestQueryValidation:
    async def test_rejects_both_cursors(self, client: AsyncClient) -> None:
        response = await client.get("/tasks", params={"after": 1, "before": 5})

        assert response.status_code == 422
        assert "Only one of 'after' and 'before'" in response.text

    @pytest.mark.parametrize("cursor", ["after", "before"])
    @pytest.mark.parametrize("value", ["-1", "abc", "1.5", ""])
    async def test_rejects_invalid_cursor(
        self, client: AsyncClient, cursor: str, value: str
    ) -> None:
        response = await client.get("/tasks", params={cursor: value})

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", cursor]

    async def test_rejects_unknown_parameter(self, client: AsyncClient) -> None:
        response = await client.get("/tasks", params={"complete": "true"})

        assert response.status_code == 422
        [error] = response.json()["detail"]
        assert error["type"] == "extra_forbidden"
        assert error["loc"] == ["query", "complete"]
