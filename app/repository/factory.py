from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app import db
from app.config import Settings, StorageBackend
from app.repository.base import TaskRepository
from app.repository.memory import InMemoryTaskRepository
from app.repository.sqlite import SQLiteTaskRepository


@asynccontextmanager
async def open_task_repository(settings: Settings) -> AsyncGenerator[TaskRepository]:
    """Provide the configured repository, initializing and releasing any backing resources."""
    match settings.storage:
        case StorageBackend.MEMORY:
            yield InMemoryTaskRepository()
        case StorageBackend.SQLITE:
            async with db.connect(settings.db_path) as connection:
                await db.initialize(connection)
                yield SQLiteTaskRepository(connection)
