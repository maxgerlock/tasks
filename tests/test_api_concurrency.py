"""HTTP-level tests for optimistic concurrency control via task versions."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient, Response

from app.config import Settings, StorageBackend
from app.main import create_app
from tests.conftest import OTHER_USER_ID, as_user, serve

pytestmark = pytest.mark.anyio

CONFLICT = {"detail": "Task has been modified"}
NOT_FOUND = {"detail": "Task not found"}


async def create_task(client: AsyncClient, description: str = "Buy groceries") -> dict[str, Any]:
    response = await client.post("/tasks", json={"description": description})
    assert response.status_code == 201
    return response.json()


async def get_task(client: AsyncClient, task_id: int) -> dict[str, Any]:
    response = await client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    return response.json()


async def patch(client: AsyncClient, task_id: int, **body: Any) -> Response:
    return await client.patch(f"/tasks/{task_id}", json=body)


async def delete(client: AsyncClient, task_id: int, version: Any) -> Response:
    return await client.delete(f"/tasks/{task_id}", params={"version": version})


class TestVersionIsExposed:
    async def test_every_task_representation_includes_version(self, client: AsyncClient) -> None:
        created = await create_task(client)
        fetched = await get_task(client, created["id"])
        listed = (await client.get("/tasks")).json()["items"][0]

        assert created["version"] == fetched["version"] == listed["version"] == 1

    async def test_version_increments_by_one_per_successful_update(
        self, client: AsyncClient
    ) -> None:
        task = await create_task(client)

        versions = [task["version"]]
        for completed in (True, False, True):
            response = await patch(client, task["id"], version=versions[-1], completed=completed)
            assert response.status_code == 200
            versions.append(response.json()["version"])

        assert versions == [1, 2, 3, 4]
        assert (await get_task(client, task["id"]))["version"] == 4

    async def test_rejected_requests_do_not_change_version(self, client: AsyncClient) -> None:
        task = await create_task(client)

        await patch(client, task["id"], version=1, description="")
        await patch(client, task["id"], version=7, completed=True)
        await delete(client, task["id"], version=7)

        assert (await get_task(client, task["id"]))["version"] == 1

    async def test_versions_are_tracked_per_task(self, client: AsyncClient) -> None:
        first = await create_task(client, "first")
        second = await create_task(client, "second")

        await patch(client, first["id"], version=1, completed=True)

        assert (await get_task(client, second["id"]))["version"] == 1
        assert (await patch(client, second["id"], version=1, completed=True)).status_code == 200


class TestStaleUpdate:
    async def test_client_with_stale_version_gets_409(self, client: AsyncClient) -> None:
        # Two clients read the same state; the first write wins, the second is stale.
        task = await create_task(client)
        client_a_view = await get_task(client, task["id"])
        client_b_view = await get_task(client, task["id"])

        response_a = await patch(
            client, task["id"], version=client_a_view["version"], description="Buy coffee"
        )
        response_b = await patch(
            client, task["id"], version=client_b_view["version"], completed=True
        )

        assert response_a.status_code == 200
        assert response_b.status_code == 409
        assert response_b.json() == CONFLICT
        assert await get_task(client, task["id"]) == response_a.json()

    async def test_client_can_retry_after_refetching(self, client: AsyncClient) -> None:
        task = await create_task(client)
        await patch(client, task["id"], version=1, description="Buy coffee")
        assert (await patch(client, task["id"], version=1, completed=True)).status_code == 409

        latest = await get_task(client, task["id"])
        response = await patch(client, task["id"], version=latest["version"], completed=True)

        assert response.status_code == 200
        assert response.json() == {
            "id": task["id"],
            "description": "Buy coffee",
            "completed": True,
            "version": 3,
        }

    async def test_version_ahead_of_current_is_a_conflict(self, client: AsyncClient) -> None:
        task = await create_task(client)

        response = await patch(client, task["id"], version=2, completed=True)

        assert response.status_code == 409
        assert response.json() == CONFLICT

    async def test_conflict_applies_even_when_change_matches_current_state(
        self, client: AsyncClient
    ) -> None:
        task = await create_task(client)
        updated = (await patch(client, task["id"], version=1, completed=True)).json()

        response = await patch(client, task["id"], version=1, completed=True)

        assert response.status_code == 409
        assert await get_task(client, task["id"]) == updated

    async def test_update_after_delete_is_404_not_409(self, client: AsyncClient) -> None:
        task = await create_task(client)
        await delete(client, task["id"], version=1)

        response = await patch(client, task["id"], version=1, completed=True)

        assert response.status_code == 404
        assert response.json() == NOT_FOUND


class TestStaleDelete:
    async def test_delete_with_stale_version_gets_409_and_keeps_task(
        self, client: AsyncClient
    ) -> None:
        task = await create_task(client)
        updated = (await patch(client, task["id"], version=1, completed=True)).json()

        response = await delete(client, task["id"], version=1)

        assert response.status_code == 409
        assert response.json() == CONFLICT
        assert await get_task(client, task["id"]) == updated

    async def test_delete_with_current_version_succeeds(self, client: AsyncClient) -> None:
        task = await create_task(client)
        await patch(client, task["id"], version=1, completed=True)

        response = await delete(client, task["id"], version=2)

        assert response.status_code == 204
        assert (await client.get(f"/tasks/{task['id']}")).status_code == 404

    async def test_update_after_stale_delete_still_works(self, client: AsyncClient) -> None:
        task = await create_task(client)
        await patch(client, task["id"], version=1, completed=True)
        await delete(client, task["id"], version=1)

        response = await patch(client, task["id"], version=2, description="still here")

        assert response.status_code == 200


class TestVersionValidation:
    @pytest.mark.parametrize(
        "version",
        [
            pytest.param(None, id="null"),
            pytest.param("1", id="string"),
            pytest.param(1.5, id="float"),
            pytest.param(True, id="bool"),
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param([1], id="list"),
        ],
    )
    async def test_patch_rejects_invalid_version(self, client: AsyncClient, version: Any) -> None:
        task = await create_task(client)

        response = await patch(client, task["id"], version=version, completed=True)

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "version"]

    async def test_patch_requires_version(self, client: AsyncClient) -> None:
        task = await create_task(client)

        response = await patch(client, task["id"], completed=True)

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert (error["type"], error["loc"]) == ("missing", ["body", "version"])
        assert (await get_task(client, task["id"]))["completed"] is False

    async def test_delete_requires_version(self, client: AsyncClient) -> None:
        task = await create_task(client)

        response = await client.delete(f"/tasks/{task['id']}")

        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert (error["type"], error["loc"]) == ("missing", ["query", "version"])
        assert (await client.get(f"/tasks/{task['id']}")).status_code == 200

    @pytest.mark.parametrize("version", ["0", "-1", "abc", "1.5", ""])
    async def test_delete_rejects_invalid_version(self, client: AsyncClient, version: str) -> None:
        task = await create_task(client)

        response = await delete(client, task["id"], version=version)

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", "version"]
        assert (await client.get(f"/tasks/{task['id']}")).status_code == 200

    async def test_invalid_version_is_rejected_before_existence_check(
        self, client: AsyncClient
    ) -> None:
        assert (await patch(client, 999, version=0, completed=True)).status_code == 422
        assert (await delete(client, 999, version=0)).status_code == 422


class TestOwnershipTakesPrecedence:
    @pytest.mark.parametrize("version", [1, 2], ids=["current", "stale"])
    async def test_other_users_task_is_404_whatever_the_version(
        self, client: AsyncClient, version: int
    ) -> None:
        theirs = (
            await client.post(
                "/tasks", json={"description": "theirs"}, headers=as_user(OTHER_USER_ID)
            )
        ).json()

        patch_response = await client.patch(
            f"/tasks/{theirs['id']}", json={"version": version, "completed": True}
        )
        delete_response = await delete(client, theirs["id"], version=version)

        assert (patch_response.status_code, patch_response.json()) == (404, NOT_FOUND)
        assert (delete_response.status_code, delete_response.json()) == (404, NOT_FOUND)
        owner_view = await client.get(f"/tasks/{theirs['id']}", headers=as_user(OTHER_USER_ID))
        assert owner_view.json() == theirs


class TestConcurrentRequests:
    async def test_concurrent_updates_from_same_state_have_one_winner(
        self, client: AsyncClient
    ) -> None:
        task = await create_task(client)

        responses = await asyncio.gather(
            *(patch(client, task["id"], version=1, description=f"writer {i}") for i in range(20))
        )

        statuses = sorted(response.status_code for response in responses)
        assert statuses == [200] + [409] * 19
        winner = next(response.json() for response in responses if response.status_code == 200)
        assert winner["version"] == 2
        assert await get_task(client, task["id"]) == winner

    async def test_concurrent_update_and_delete_from_same_state_have_one_winner(
        self, client: AsyncClient
    ) -> None:
        task = await create_task(client)

        update, removal = await asyncio.gather(
            patch(client, task["id"], version=1, completed=True),
            delete(client, task["id"], version=1),
        )

        assert sorted([update.status_code, removal.status_code]) in ([200, 409], [204, 409])

    async def test_separate_app_instances_cannot_both_update(self, db_path: Path) -> None:
        settings = Settings(storage=StorageBackend.SQLITE, db_path=db_path)
        async with serve(create_app(settings)) as setup:
            task = await create_task(setup)

        async with (
            serve(create_app(settings)) as first,
            serve(create_app(settings)) as second,
        ):
            responses = await asyncio.gather(
                patch(first, task["id"], version=1, description="from first"),
                patch(second, task["id"], version=1, description="from second"),
            )

            assert sorted(response.status_code for response in responses) == [200, 409]
            winner = next(r.json() for r in responses if r.status_code == 200)
            assert await get_task(first, task["id"]) == winner
            assert await get_task(second, task["id"]) == winner
