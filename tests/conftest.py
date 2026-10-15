from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.auth import USER_ID_HEADER
from app.config import Settings, StorageBackend
from app.main import create_app
from app.repository import TaskRepository, open_task_repository
from app.service import TaskService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "tasks.db"


@pytest.fixture(params=list(StorageBackend), ids=str)
def settings(request: pytest.FixtureRequest, db_path: Path) -> Settings:
    return Settings(storage=request.param, db_path=db_path)


@pytest.fixture
async def repository(settings: Settings) -> AsyncIterator[TaskRepository]:
    async with open_task_repository(settings) as repo:
        yield repo


USER_ID = 1
OTHER_USER_ID = 2


@pytest.fixture
def service(repository: TaskRepository) -> TaskService:
    return TaskService(repository, USER_ID)


def as_user(user_id: int) -> dict[str, str]:
    return {USER_ID_HEADER: str(user_id)}


@asynccontextmanager
async def serve(app: FastAPI, user_id: int | None = USER_ID) -> AsyncGenerator[httpx.AsyncClient]:
    """Run the app's startup/shutdown lifecycle around an HTTP client bound to it.

    The client identifies as ``user_id`` on every request unless that is None.
    """
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=as_user(user_id) if user_id is not None else None,
        ) as client,
    ):
        yield client


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with serve(app) as test_client:
        yield test_client
