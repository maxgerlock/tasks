"""Caller identification via X-User-ID and per-user task isolation."""

from typing import Any

import pytest
from httpx import AsyncClient

from app.auth import USER_ID_HEADER
from tests.conftest import OTHER_USER_ID, USER_ID, as_user

pytestmark = pytest.mark.anyio

NOT_FOUND = {"detail": "Task not found"}

# Each endpoint as (method, path, JSON body), using valid inputs.
ENDPOINTS = [
    pytest.param("POST", "/tasks", {"description": "x"}, id="create"),
    pytest.param("GET", "/tasks", None, id="list"),
    pytest.param("GET", "/tasks/1", None, id="get"),
    pytest.param("PATCH", "/tasks/1", {"version": 1, "completed": True}, id="update"),
    pytest.param("DELETE", "/tasks/1?version=1", None, id="delete"),
]


async def create_task(client: AsyncClient, description: str, user_id: int) -> dict[str, Any]:
    response = await client.post(
        "/tasks", json={"description": description}, headers=as_user(user_id)
    )
    assert response.status_code == 201
    return response.json()


class TestIdentification:
    @pytest.mark.parametrize(("method", "path", "body"), ENDPOINTS)
    async def test_missing_header_is_401(
        self, client: AsyncClient, method: str, path: str, body: Any
    ) -> None:
        client.headers.pop(USER_ID_HEADER)

        response = await client.request(method, path, json=body)

        assert response.status_code == 401
        assert response.json() == {"detail": f"Missing {USER_ID_HEADER} header"}
        assert response.headers["www-authenticate"] == USER_ID_HEADER

    @pytest.mark.parametrize(("method", "path", "body"), ENDPOINTS)
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("", id="empty"),
            pytest.param("abc", id="letters"),
            pytest.param("0", id="zero"),
            pytest.param("-1", id="negative"),
            pytest.param("1.5", id="decimal"),
            pytest.param("1e3", id="exponent"),
            pytest.param("+1", id="plus-sign"),
            pytest.param("01", id="leading-zero"),
            pytest.param("1 2", id="inner-space"),
            pytest.param("9" * 20, id="too-large"),
            pytest.param(str(2**63), id="just-over-int64"),
        ],
    )
    async def test_malformed_header_is_401(
        self, client: AsyncClient, method: str, path: str, body: Any, value: str
    ) -> None:
        response = await client.request(method, path, json=body, headers={USER_ID_HEADER: value})

        assert response.status_code == 401
        assert response.json() == {"detail": f"{USER_ID_HEADER} must be a positive integer"}
        assert response.headers["www-authenticate"] == USER_ID_HEADER

    async def test_non_ascii_digits_are_malformed(self, client: AsyncClient) -> None:
        arabic_indic_one = "\u0661"

        response = await client.get("/tasks", headers=[(b"X-User-ID", arabic_indic_one.encode())])

        assert response.status_code == 401

    async def test_multiple_headers_are_rejected(self, client: AsyncClient) -> None:
        client.headers.pop(USER_ID_HEADER)

        response = await client.get(
            "/tasks", headers=[(USER_ID_HEADER, "1"), (USER_ID_HEADER, "2")]
        )

        assert response.status_code == 401
        assert response.json() == {"detail": f"Multiple {USER_ID_HEADER} headers"}

    async def test_header_name_is_case_insensitive(self, client: AsyncClient) -> None:
        client.headers.pop(USER_ID_HEADER)

        response = await client.get("/tasks", headers={"x-user-id": "1"})

        assert response.status_code == 200

    async def test_largest_int64_is_accepted(self, client: AsyncClient) -> None:
        user_id = 2**63 - 1

        task = await create_task(client, "big user", user_id)

        response = await client.get(f"/tasks/{task['id']}", headers=as_user(user_id))
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            pytest.param("POST", "/tasks", {"description": ""}, id="invalid-body"),
            pytest.param("GET", "/tasks/abc", None, id="invalid-path"),
            pytest.param("GET", "/tasks?limit=0", None, id="invalid-query"),
            pytest.param("PATCH", "/tasks/1", {}, id="invalid-update"),
            pytest.param("DELETE", "/tasks/1", None, id="delete-without-version"),
        ],
    )
    async def test_identity_is_checked_before_request_validation(
        self, client: AsyncClient, method: str, path: str, body: Any
    ) -> None:
        client.headers.pop(USER_ID_HEADER)

        response = await client.request(method, path, json=body)

        assert response.status_code == 401

    async def test_rejected_request_creates_nothing(self, client: AsyncClient) -> None:
        await client.post("/tasks", json={"description": "x"}, headers={USER_ID_HEADER: "abc"})

        assert (await client.get("/tasks")).json()["items"] == []


class TestCreate:
    async def test_task_belongs_to_caller(self, client: AsyncClient) -> None:
        task = await create_task(client, "mine", USER_ID)

        assert (await client.get(f"/tasks/{task['id']}", headers=as_user(USER_ID))).json() == task
        response = await client.get(f"/tasks/{task['id']}", headers=as_user(OTHER_USER_ID))
        assert response.status_code == 404

    async def test_response_does_not_expose_owner(self, client: AsyncClient) -> None:
        task = await create_task(client, "mine", USER_ID)

        assert set(task) == {"id", "description", "completed", "version"}

    async def test_cannot_create_task_for_someone_else(self, client: AsyncClient) -> None:
        response = await client.post(
            "/tasks", json={"description": "sneaky", "user_id": OTHER_USER_ID}
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "user_id"]
        other = await client.get("/tasks", headers=as_user(OTHER_USER_ID))
        assert other.json()["items"] == []

    async def test_ids_are_unique_across_users(self, client: AsyncClient) -> None:
        first = await create_task(client, "a", USER_ID)
        second = await create_task(client, "b", OTHER_USER_ID)
        third = await create_task(client, "c", USER_ID)

        assert [first["id"], second["id"], third["id"]] == [1, 2, 3]


class TestOtherUsersTasks:
    """User 2 owns task 1; every request below is made as user 1."""

    @pytest.fixture
    async def theirs(self, client: AsyncClient) -> dict[str, Any]:
        return await create_task(client, "theirs", OTHER_USER_ID)

    async def test_get_is_404(self, client: AsyncClient, theirs: dict[str, Any]) -> None:
        response = await client.get(f"/tasks/{theirs['id']}")

        assert response.status_code == 404
        assert response.json() == NOT_FOUND

    async def test_update_is_404_and_changes_nothing(
        self, client: AsyncClient, theirs: dict[str, Any]
    ) -> None:
        response = await client.patch(
            f"/tasks/{theirs['id']}",
            json={"version": 1, "description": "hijacked", "completed": True},
        )

        assert response.status_code == 404
        assert response.json() == NOT_FOUND
        owner_view = await client.get(f"/tasks/{theirs['id']}", headers=as_user(OTHER_USER_ID))
        assert owner_view.json() == theirs

    async def test_delete_is_404_and_deletes_nothing(
        self, client: AsyncClient, theirs: dict[str, Any]
    ) -> None:
        response = await client.delete(f"/tasks/{theirs['id']}", params={"version": 1})

        assert response.status_code == 404
        assert response.json() == NOT_FOUND
        owner_view = await client.get(f"/tasks/{theirs['id']}", headers=as_user(OTHER_USER_ID))
        assert owner_view.status_code == 200

    async def test_404_is_indistinguishable_from_nonexistent(
        self, client: AsyncClient, theirs: dict[str, Any]
    ) -> None:
        for method, body in [
            ("GET", None),
            ("PATCH", {"version": 1, "completed": True}),
            ("DELETE", None),
        ]:
            params = {"version": 1} if method == "DELETE" else None
            foreign = await client.request(
                method, f"/tasks/{theirs['id']}", json=body, params=params
            )
            missing = await client.request(method, "/tasks/999", json=body, params=params)

            assert (foreign.status_code, foreign.json()) == (missing.status_code, missing.json())
            assert foreign.headers.keys() == missing.headers.keys()

    async def test_invalid_update_is_still_422(
        self, client: AsyncClient, theirs: dict[str, Any]
    ) -> None:
        response = await client.patch(f"/tasks/{theirs['id']}", json={"version": 1})

        assert response.status_code == 422

    async def test_owner_cannot_be_changed(
        self, client: AsyncClient, theirs: dict[str, Any]
    ) -> None:
        response = await client.patch(
            f"/tasks/{theirs['id']}",
            json={"version": 1, "user_id": USER_ID},
            headers=as_user(OTHER_USER_ID),
        )

        assert response.status_code == 422
        assert (await client.get(f"/tasks/{theirs['id']}")).status_code == 404


class TestListing:
    @pytest.fixture(autouse=True)
    async def interleaved(self, client: AsyncClient) -> None:
        """Tasks 1-12 alternate owners: user 1 owns odd IDs, user 2 owns even IDs.

        Each user's tasks 1-4 (in creation order) are completed.
        """
        for i in range(12):
            owner = USER_ID if i % 2 == 0 else OTHER_USER_ID
            task = await create_task(client, f"task {i + 1}", owner)
            if i < 8:
                await client.patch(
                    f"/tasks/{task['id']}",
                    json={"version": 1, "completed": True},
                    headers=as_user(owner),
                )

    async def test_lists_only_own_tasks(self, client: AsyncClient) -> None:
        mine = (await client.get("/tasks")).json()
        theirs = (await client.get("/tasks", headers=as_user(OTHER_USER_ID))).json()

        assert [task["id"] for task in mine["items"]] == [1, 3, 5, 7, 9, 11]
        assert [task["id"] for task in theirs["items"]] == [2, 4, 6, 8, 10, 12]

    async def test_user_without_tasks_gets_empty_page(self, client: AsyncClient) -> None:
        response = await client.get("/tasks", headers=as_user(99))

        assert response.json() == {"items": [], "next_cursor": None, "has_more": False}

    async def test_pagination_counts_only_own_tasks(self, client: AsyncClient) -> None:
        page = (await client.get("/tasks", params={"limit": 2})).json()

        assert [task["id"] for task in page["items"]] == [1, 3]
        assert page["next_cursor"] == 3
        assert page["has_more"] is True

    async def test_has_more_ignores_other_users_later_tasks(self, client: AsyncClient) -> None:
        page = (await client.get("/tasks", params={"after": 9})).json()

        assert [task["id"] for task in page["items"]] == [11]
        assert page["has_more"] is False
        assert page["next_cursor"] is None

    async def test_walking_pages_visits_only_own_tasks(self, client: AsyncClient) -> None:
        seen: list[int] = []
        params: dict[str, int] = {"limit": 4}
        while True:
            page = (await client.get("/tasks", params=params)).json()
            seen.extend(task["id"] for task in page["items"])
            if not page["has_more"]:
                break
            params["after"] = page["next_cursor"]

        assert seen == [1, 3, 5, 7, 9, 11]

    async def test_before_counts_only_own_tasks(self, client: AsyncClient) -> None:
        page = (await client.get("/tasks", params={"before": 12, "limit": 2})).json()

        assert [task["id"] for task in page["items"]] == [11, 9]
        assert page["next_cursor"] == 9

    async def test_cursor_may_be_another_users_task_id(self, client: AsyncClient) -> None:
        page = (await client.get("/tasks", params={"after": 6, "limit": 2})).json()

        assert [task["id"] for task in page["items"]] == [7, 9]

    @pytest.mark.parametrize(
        ("completed", "user_id", "expected_ids"),
        [
            ("true", USER_ID, [1, 3, 5, 7]),
            ("false", USER_ID, [9, 11]),
            ("true", OTHER_USER_ID, [2, 4, 6, 8]),
            ("false", OTHER_USER_ID, [10, 12]),
        ],
    )
    async def test_completed_filter_is_scoped_to_user(
        self, client: AsyncClient, completed: str, user_id: int, expected_ids: list[int]
    ) -> None:
        response = await client.get(
            "/tasks", params={"completed": completed}, headers=as_user(user_id)
        )

        assert [task["id"] for task in response.json()["items"]] == expected_ids
