"""End-to-end persistence tests: each `serve` block is one application run."""

import sqlite3
from pathlib import Path

import pytest

from app.config import Settings, StorageBackend
from app.db import SchemaMismatchError
from app.main import create_app
from tests.conftest import serve

pytestmark = pytest.mark.anyio


@pytest.fixture
def sqlite_settings(db_path: Path) -> Settings:
    return Settings(storage=StorageBackend.SQLITE, db_path=db_path)


async def test_starts_against_empty_store(sqlite_settings: Settings) -> None:
    assert not sqlite_settings.db_path.parent.exists()

    async with serve(create_app(sqlite_settings)) as client:
        response = await client.get("/tasks")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert sqlite_settings.db_path.exists()


async def test_tasks_survive_restart(sqlite_settings: Settings) -> None:
    async with serve(create_app(sqlite_settings)) as client:
        first = (await client.post("/tasks", json={"description": "Buy groceries"})).json()
        second = (await client.post("/tasks", json={"description": "Walk dog"})).json()

    async with serve(create_app(sqlite_settings)) as client:
        assert (await client.get("/tasks")).json()["items"] == [first, second]
        assert (await client.get(f"/tasks/{first['id']}")).json() == first


async def test_updates_survive_restart(sqlite_settings: Settings) -> None:
    async with serve(create_app(sqlite_settings)) as client:
        await client.post("/tasks", json={"description": "Buy groceries"})
        await client.patch(
            "/tasks/1", json={"version": 1, "description": "Buy coffee", "completed": True}
        )

    async with serve(create_app(sqlite_settings)) as client:
        task = (await client.get("/tasks/1")).json()

    assert task == {"id": 1, "description": "Buy coffee", "completed": True, "version": 2}
    assert task["completed"] is True


async def test_deletes_survive_restart(sqlite_settings: Settings) -> None:
    async with serve(create_app(sqlite_settings)) as client:
        await client.post("/tasks", json={"description": "Buy groceries"})
        await client.delete("/tasks/1", params={"version": 1})

    async with serve(create_app(sqlite_settings)) as client:
        assert (await client.get("/tasks/1")).status_code == 404
        assert (await client.get("/tasks")).json()["items"] == []


async def test_ids_continue_after_restart(sqlite_settings: Settings) -> None:
    async with serve(create_app(sqlite_settings)) as client:
        await client.post("/tasks", json={"description": "first"})
        await client.post("/tasks", json={"description": "second"})

    async with serve(create_app(sqlite_settings)) as client:
        response = await client.post("/tasks", json={"description": "third"})

    assert response.json()["id"] == 3


async def test_ids_of_deleted_tasks_are_not_reused_after_restart(
    sqlite_settings: Settings,
) -> None:
    async with serve(create_app(sqlite_settings)) as client:
        await client.post("/tasks", json={"description": "first"})
        await client.post("/tasks", json={"description": "second"})
        await client.delete("/tasks/2", params={"version": 1})
        await client.delete("/tasks/1", params={"version": 1})

    async with serve(create_app(sqlite_settings)) as client:
        response = await client.post("/tasks", json={"description": "third"})

    assert response.json()["id"] == 3


async def test_ownership_survives_restart(sqlite_settings: Settings) -> None:
    async with serve(create_app(sqlite_settings), user_id=1) as client:
        mine = (await client.post("/tasks", json={"description": "mine"})).json()
    async with serve(create_app(sqlite_settings), user_id=2) as client:
        theirs = (await client.post("/tasks", json={"description": "theirs"})).json()

    async with serve(create_app(sqlite_settings), user_id=1) as client:
        assert (await client.get("/tasks")).json()["items"] == [mine]
        assert (await client.get(f"/tasks/{theirs['id']}")).status_code == 404
    async with serve(create_app(sqlite_settings), user_id=2) as client:
        assert (await client.get("/tasks")).json()["items"] == [theirs]


async def test_concurrently_running_apps_share_the_store(sqlite_settings: Settings) -> None:
    async with (
        serve(create_app(sqlite_settings)) as first,
        serve(create_app(sqlite_settings)) as second,
    ):
        a = (await first.post("/tasks", json={"description": "from first"})).json()
        b = (await second.post("/tasks", json={"description": "from second"})).json()

        assert a["id"] != b["id"]
        assert (await first.get("/tasks")).json()["items"] == [a, b]
        assert (await second.get("/tasks")).json()["items"] == [a, b]


async def test_startup_fails_on_schema_mismatch(sqlite_settings: Settings) -> None:
    sqlite_settings.db_path.parent.mkdir(parents=True)
    with sqlite3.connect(sqlite_settings.db_path) as conn:
        conn.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT)")

    with pytest.raises(SchemaMismatchError, match="missing column 'description'"):
        async with serve(create_app(sqlite_settings)):
            pass


async def test_app_reads_storage_settings_from_environment(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKS_STORAGE", "sqlite")
    monkeypatch.setenv("TASKS_DB_PATH", str(db_path))

    async with serve(create_app()) as client:
        await client.post("/tasks", json={"description": "Buy groceries"})

    async with serve(create_app()) as client:
        assert len((await client.get("/tasks")).json()["items"]) == 1
    assert db_path.exists()


async def test_memory_backend_does_not_persist_across_restart(db_path: Path) -> None:
    settings = Settings(storage=StorageBackend.MEMORY, db_path=db_path)

    async with serve(create_app(settings)) as client:
        await client.post("/tasks", json={"description": "Buy groceries"})

    async with serve(create_app(settings)) as client:
        assert (await client.get("/tasks")).json()["items"] == []
    assert not db_path.exists()
