import random
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.config import Settings, StorageBackend
from app.db import SchemaMismatchError
from app.main import create_app
from app.seed import DEFAULT_COUNT, main, seed
from tests.conftest import serve


@pytest.fixture(autouse=True)
def sqlite_env(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKS_STORAGE", "sqlite")
    monkeypatch.setenv("TASKS_DB_PATH", str(db_path))


def read_rows(db_path: Path) -> list[tuple[int, str, int]]:
    with closing(sqlite3.connect(db_path)) as connection:
        return connection.execute(
            "SELECT id, description, completed FROM tasks ORDER BY id"
        ).fetchall()


def test_seeds_empty_store(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--count", "30"])

    rows = read_rows(db_path)
    assert [row[0] for row in rows] == list(range(1, 31))
    assert all(row[1] for row in rows)
    assert "Seeded 30 tasks" in capsys.readouterr().out


def test_default_count_includes_completed_and_incomplete(db_path: Path) -> None:
    main([])

    rows = read_rows(db_path)
    assert len(rows) == DEFAULT_COUNT
    assert {row[2] for row in rows} == {0, 1}


def test_appends_to_existing_tasks(db_path: Path) -> None:
    main(["--count", "5"])
    main(["--count", "5"])

    assert [row[0] for row in read_rows(db_path)] == list(range(1, 11))


def test_reset_replaces_tasks_and_restarts_ids(db_path: Path) -> None:
    main(["--count", "5"])

    main(["--count", "3", "--reset"])

    assert [row[0] for row in read_rows(db_path)] == [1, 2, 3]


def test_random_seed_is_reproducible(db_path: Path) -> None:
    main(["--count", "20", "--reset", "--random-seed", "7"])
    first = read_rows(db_path)

    main(["--count", "20", "--reset", "--random-seed", "7"])

    assert read_rows(db_path) == first


def test_refuses_memory_backend(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKS_STORAGE", "memory")

    with pytest.raises(SystemExit) as exc_info:
        main(["--count", "5"])

    assert exc_info.value.code == 2
    assert not db_path.exists()


@pytest.mark.parametrize("count", ["0", "-3", "many"])
def test_rejects_invalid_count(db_path: Path, count: str) -> None:
    with pytest.raises(SystemExit):
        main(["--count", count])

    assert not db_path.exists()


def read_owners(db_path: Path) -> list[int]:
    with closing(sqlite3.connect(db_path)) as connection:
        return [row[0] for row in connection.execute("SELECT user_id FROM tasks ORDER BY id")]


def test_default_users_are_assigned_round_robin(db_path: Path) -> None:
    main(["--count", "7"])

    assert read_owners(db_path) == [1, 2, 3, 1, 2, 3, 1]


def test_users_option(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--count", "4", "--users", "7, 42"])

    assert read_owners(db_path) == [7, 42, 7, 42]
    assert "user 42: 2 tasks" in capsys.readouterr().out


def test_users_option_ignores_duplicates(db_path: Path) -> None:
    main(["--count", "3", "--users", "5,5,6"])

    assert read_owners(db_path) == [5, 6, 5]


@pytest.mark.parametrize("users", ["", "0", "1,abc", "1,-2", "1,,2"])
def test_rejects_invalid_users(db_path: Path, users: str) -> None:
    with pytest.raises(SystemExit):
        main(["--users", users])

    assert not db_path.exists()


def test_recreate_replaces_an_outdated_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "description TEXT NOT NULL, completed INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO tasks (description, completed) VALUES ('old', 0)")
        connection.commit()

    main(["--count", "3", "--recreate"])

    assert [row[0] for row in read_rows(db_path)] == [1, 2, 3]
    assert read_owners(db_path) == [1, 2, 3]


def test_outdated_table_without_recreate_fails_and_keeps_data(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT)")
        connection.execute("INSERT INTO tasks (title) VALUES ('old')")
        connection.commit()

    with pytest.raises(SchemaMismatchError):
        main(["--count", "3"])

    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("SELECT title FROM tasks").fetchall() == [("old",)]


@pytest.mark.anyio
async def test_seeded_tasks_are_served_to_their_owners(db_path: Path) -> None:
    settings = Settings(storage=StorageBackend.SQLITE, db_path=db_path)
    await seed(settings, count=45, user_ids=[1, 2, 3], rng=random.Random(0))

    async with serve(create_app(settings), user_id=2) as client:
        page = (await client.get("/tasks", params={"limit": 100})).json()

    assert [task["id"] for task in page["items"]] == list(range(2, 46, 3))
