from pathlib import Path

import pytest

from app.config import DEFAULT_DB_PATH, Settings, StorageBackend


def test_defaults_to_sqlite_at_default_path() -> None:
    assert Settings.from_env({}) == Settings(storage=StorageBackend.SQLITE, db_path=DEFAULT_DB_PATH)


@pytest.mark.parametrize("value", ["memory", "MEMORY", " Memory "])
def test_reads_storage_backend_case_insensitively(value: str) -> None:
    assert Settings.from_env({"TASKS_STORAGE": value}).storage is StorageBackend.MEMORY


def test_reads_db_path() -> None:
    settings = Settings.from_env({"TASKS_DB_PATH": "/tmp/elsewhere/tasks.db"})

    assert settings.db_path == Path("/tmp/elsewhere/tasks.db")


def test_rejects_unknown_storage_backend() -> None:
    with pytest.raises(ValueError, match="TASKS_STORAGE 'postgres'"):
        Settings.from_env({"TASKS_STORAGE": "postgres"})
