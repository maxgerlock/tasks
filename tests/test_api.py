from typing import Any

import pytest
from httpx import AsyncClient

from app.schemas import DESCRIPTION_MAX_LENGTH

pytestmark = pytest.mark.anyio


async def create_task(client: AsyncClient, description: str) -> dict[str, Any]:
    response = await client.post("/tasks", json={"description": description})
    assert response.status_code == 201
    return response.json()


async def list_items(client: AsyncClient, **params: str | int) -> list[dict[str, Any]]:
    response = await client.get("/tasks", params=params)
    assert response.status_code == 200
    return response.json()["items"]


class TestCreateTask:
    async def test_returns_created_task(self, client: AsyncClient) -> None:
        response = await client.post("/tasks", json={"description": "Buy groceries"})

        assert response.status_code == 201
        assert response.json() == {
            "id": 1,
            "description": "Buy groceries",
            "completed": False,
            "version": 1,
        }

    async def test_sets_location_header(self, client: AsyncClient) -> None:
        response = await client.post("/tasks", json={"description": "Buy groceries"})

        assert response.headers["location"] == f"/tasks/{response.json()['id']}"

    async def test_assigns_unique_sequential_ids(self, client: AsyncClient) -> None:
        ids = [(await create_task(client, f"task {i}"))["id"] for i in range(3)]

        assert ids == [1, 2, 3]

    async def test_strips_surrounding_whitespace(self, client: AsyncClient) -> None:
        task = await create_task(client, "  Buy groceries \n")

        assert task["description"] == "Buy groceries"

    async def test_accepts_description_at_max_length(self, client: AsyncClient) -> None:
        description = "x" * DESCRIPTION_MAX_LENGTH

        assert (await create_task(client, description))["description"] == description

    async def test_max_length_applies_after_stripping(self, client: AsyncClient) -> None:
        description = "x" * DESCRIPTION_MAX_LENGTH

        assert (await create_task(client, f"  {description}  "))["description"] == description

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"description": ""}, id="empty"),
            pytest.param({"description": "   \t\n"}, id="whitespace-only"),
            pytest.param({"description": "x" * (DESCRIPTION_MAX_LENGTH + 1)}, id="too-long"),
            pytest.param({}, id="missing"),
            pytest.param({"description": None}, id="null"),
            pytest.param({"description": 123}, id="not-a-string"),
            pytest.param({"description": "x", "completed": True}, id="extra-completed"),
            pytest.param({"description": "x", "id": 99}, id="extra-id"),
            pytest.param({"description": "x", "version": 1}, id="extra-version"),
            pytest.param({"description": "x", "version": 5}, id="extra-version-other"),
            pytest.param(["Buy groceries"], id="not-an-object"),
        ],
    )
    async def test_rejects_invalid_body(self, client: AsyncClient, body: Any) -> None:
        response = await client.post("/tasks", json=body)

        assert response.status_code == 422
        assert "detail" in response.json()

    async def test_error_identifies_version_as_not_permitted(self, client: AsyncClient) -> None:
        response = await client.post("/tasks", json={"description": "x", "version": 7})

        [error] = response.json()["detail"]
        assert error["type"] == "extra_forbidden"
        assert error["loc"] == ["body", "version"]
        assert await list_items(client) == []

    async def test_rejects_malformed_json(self, client: AsyncClient) -> None:
        response = await client.post(
            "/tasks", content=b"{not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    async def test_rejected_request_does_not_create_task(self, client: AsyncClient) -> None:
        await client.post("/tasks", json={"description": ""})

        assert await list_items(client) == []
        assert (await create_task(client, "first valid"))["id"] == 1


class TestGetTask:
    async def test_returns_existing_task(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.get("/tasks/1")

        assert response.status_code == 200
        assert response.json() == {
            "id": 1,
            "description": "Buy groceries",
            "completed": False,
            "version": 1,
        }

    @pytest.mark.parametrize("task_id", [1, 999])
    async def test_returns_404_for_missing_task(self, client: AsyncClient, task_id: int) -> None:
        response = await client.get(f"/tasks/{task_id}")

        assert response.status_code == 404
        assert response.json() == {"detail": "Task not found"}

    @pytest.mark.parametrize("task_id", ["abc", "1.5", "0", "-1"])
    async def test_returns_422_for_invalid_id(self, client: AsyncClient, task_id: str) -> None:
        response = await client.get(f"/tasks/{task_id}")

        assert response.status_code == 422
        assert "detail" in response.json()


class TestListTasks:
    async def test_returns_empty_page_initially(self, client: AsyncClient) -> None:
        response = await client.get("/tasks")

        assert response.status_code == 200
        assert response.json() == {"items": [], "next_cursor": None, "has_more": False}

    async def test_returns_existing_tasks(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.get("/tasks")

        assert response.status_code == 200
        assert response.json() == {
            "items": [{"id": 1, "description": "Buy groceries", "completed": False, "version": 1}],
            "next_cursor": None,
            "has_more": False,
        }

    async def test_returns_all_tasks_in_id_order(self, client: AsyncClient) -> None:
        created = [await create_task(client, f"task {i}") for i in range(3)]

        assert await list_items(client) == created

    async def test_order_is_stable_across_updates(self, client: AsyncClient) -> None:
        for i in range(3):
            await create_task(client, f"task {i}")

        await client.patch(
            "/tasks/1", json={"version": 1, "description": "zzz renamed", "completed": True}
        )

        assert [task["id"] for task in await list_items(client)] == [1, 2, 3]


class TestUpdateTask:
    async def test_updates_description_only(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch("/tasks/1", json={"version": 1, "description": "Buy coffee"})

        assert response.status_code == 200
        assert response.json() == {
            "id": 1,
            "description": "Buy coffee",
            "completed": False,
            "version": 2,
        }

    async def test_updates_completed_only(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch("/tasks/1", json={"version": 1, "completed": True})

        assert response.status_code == 200
        assert response.json() == {
            "id": 1,
            "description": "Buy groceries",
            "completed": True,
            "version": 2,
        }

    async def test_updates_both_fields(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch(
            "/tasks/1",
            json={"version": 1, "description": "Buy groceries and coffee", "completed": True},
        )

        assert response.status_code == 200
        assert response.json() == {
            "id": 1,
            "description": "Buy groceries and coffee",
            "completed": True,
            "version": 2,
        }

    async def test_can_mark_task_incomplete_again(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")
        await client.patch("/tasks/1", json={"version": 1, "completed": True})

        response = await client.patch("/tasks/1", json={"version": 2, "completed": False})

        assert response.json()["completed"] is False
        assert response.json()["version"] == 3

    async def test_strips_surrounding_whitespace(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch(
            "/tasks/1", json={"version": 1, "description": "  Buy coffee \n"}
        )

        assert response.json()["description"] == "Buy coffee"

    async def test_update_is_visible_to_get_and_list(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        updated = (await client.patch("/tasks/1", json={"version": 1, "completed": True})).json()

        assert (await client.get("/tasks/1")).json() == updated
        assert await list_items(client) == [updated]

    async def test_does_not_affect_other_tasks(self, client: AsyncClient) -> None:
        await create_task(client, "first")
        second = await create_task(client, "second")

        await client.patch("/tasks/1", json={"version": 1, "completed": True})

        assert (await client.get("/tasks/2")).json() == second

    @pytest.mark.parametrize("task_id", [1, 999])
    async def test_returns_404_for_missing_task(self, client: AsyncClient, task_id: int) -> None:
        response = await client.patch(f"/tasks/{task_id}", json={"version": 1, "completed": True})

        assert response.status_code == 404
        assert response.json() == {"detail": "Task not found"}

    @pytest.mark.parametrize("task_id", ["abc", "0", "-1"])
    async def test_returns_422_for_invalid_id(self, client: AsyncClient, task_id: str) -> None:
        response = await client.patch(f"/tasks/{task_id}", json={"version": 1, "completed": True})

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"version": 1}, id="no-changes"),
            pytest.param({}, id="empty-object"),
            pytest.param({"version": 1, "description": ""}, id="empty-description"),
            pytest.param({"version": 1, "description": "   \t\n"}, id="whitespace-description"),
            pytest.param(
                {"version": 1, "description": "x" * (DESCRIPTION_MAX_LENGTH + 1)}, id="too-long"
            ),
            pytest.param({"version": 1, "description": None}, id="null-description"),
            pytest.param({"version": 1, "description": 123}, id="non-string-description"),
            pytest.param({"version": 1, "completed": None}, id="null-completed"),
            pytest.param({"version": 1, "completed": "true"}, id="string-completed"),
            pytest.param({"version": 1, "completed": 1}, id="integer-completed"),
            pytest.param({"version": 1, "id": 2}, id="id-only"),
            pytest.param({"version": 1, "completed": True, "id": 2}, id="id-with-valid-field"),
            pytest.param({"version": 1, "completed": True, "priority": "high"}, id="unknown-field"),
            pytest.param([{"version": 1, "completed": True}], id="not-an-object"),
        ],
    )
    async def test_rejects_invalid_body(self, client: AsyncClient, body: Any) -> None:
        original = await create_task(client, "Buy groceries")

        response = await client.patch("/tasks/1", json=body)

        assert response.status_code == 422
        assert "detail" in response.json()
        assert (await client.get("/tasks/1")).json() == original

    async def test_error_identifies_unknown_field(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch(
            "/tasks/1", json={"version": 1, "description": "Updated task", "priority": "high"}
        )

        assert response.status_code == 422
        [error] = response.json()["detail"]
        assert error["type"] == "extra_forbidden"
        assert error["loc"] == ["body", "priority"]

    async def test_rejects_malformed_json(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.patch(
            "/tasks/1", content=b"{not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    async def test_invalid_body_is_rejected_before_existence_check(
        self, client: AsyncClient
    ) -> None:
        response = await client.patch("/tasks/999", json={"version": 1})

        assert response.status_code == 422


class TestDeleteTask:
    async def test_returns_204_with_empty_body(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        response = await client.delete("/tasks/1", params={"version": 1})

        assert response.status_code == 204
        assert response.content == b""

    async def test_deleted_task_behaves_as_nonexistent(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")

        await client.delete("/tasks/1", params={"version": 1})

        get_response = await client.get("/tasks/1")
        assert get_response.status_code == 404
        assert get_response.json() == {"detail": "Task not found"}
        assert await list_items(client) == []
        patch_response = await client.patch("/tasks/1", json={"version": 1, "completed": True})
        assert patch_response.status_code == 404

    async def test_deleting_twice_returns_404(self, client: AsyncClient) -> None:
        await create_task(client, "Buy groceries")
        await client.delete("/tasks/1", params={"version": 1})

        response = await client.delete("/tasks/1", params={"version": 1})

        assert response.status_code == 404
        assert response.json() == {"detail": "Task not found"}

    @pytest.mark.parametrize("task_id", [1, 999])
    async def test_returns_404_for_missing_task(self, client: AsyncClient, task_id: int) -> None:
        response = await client.delete(f"/tasks/{task_id}", params={"version": 1})

        assert response.status_code == 404
        assert response.json() == {"detail": "Task not found"}

    @pytest.mark.parametrize("task_id", ["abc", "0", "-1"])
    async def test_returns_422_for_invalid_id(self, client: AsyncClient, task_id: str) -> None:
        response = await client.delete(f"/tasks/{task_id}", params={"version": 1})

        assert response.status_code == 422

    async def test_does_not_affect_other_tasks(self, client: AsyncClient) -> None:
        first = await create_task(client, "first")
        await create_task(client, "second")
        third = await create_task(client, "third")

        await client.delete("/tasks/2", params={"version": 1})

        assert await list_items(client) == [first, third]

    async def test_ids_are_not_reused_after_delete(self, client: AsyncClient) -> None:
        await create_task(client, "first")
        await client.delete("/tasks/1", params={"version": 1})

        assert (await create_task(client, "second"))["id"] == 2
