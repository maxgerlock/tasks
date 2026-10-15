"""Application settings loaded from environment variables."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self


class StorageBackend(StrEnum):
    MEMORY = "memory"
    SQLITE = "sqlite"


DEFAULT_STORAGE = StorageBackend.SQLITE
DEFAULT_DB_PATH = Path("data/tasks.db")


@dataclass(frozen=True, slots=True)
class Settings:
    storage: StorageBackend = DEFAULT_STORAGE
    db_path: Path = DEFAULT_DB_PATH

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        raw_storage = environ.get("TASKS_STORAGE", DEFAULT_STORAGE)
        try:
            storage = StorageBackend(raw_storage.strip().lower())
        except ValueError:
            allowed = ", ".join(backend.value for backend in StorageBackend)
            raise ValueError(
                f"Invalid TASKS_STORAGE {raw_storage!r}; expected one of: {allowed}"
            ) from None
        db_path = Path(environ.get("TASKS_DB_PATH", DEFAULT_DB_PATH))
        return cls(storage=storage, db_path=db_path)
