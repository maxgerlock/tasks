"""FastAPI application factory and route definitions."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse

from app.auth import USER_ID_HEADER
from app.config import Settings
from app.dependencies import TaskServiceDep
from app.models import TaskNotFoundError, VersionConflictError
from app.repository import TaskRepository, open_task_repository
from app.schemas import TaskCreate, TaskListParams, TaskPageRead, TaskRead, TaskUpdate

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": f"Missing or malformed {USER_ID_HEADER} header"
        }
    },
)

TaskId = Annotated[int, Path(ge=1)]
NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "Task not found"}
}
WRITE_RESPONSES: dict[int | str, dict[str, Any]] = {
    **NOT_FOUND_RESPONSE,
    status.HTTP_409_CONFLICT: {"description": "The task's version has changed since it was read"},
}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_task(data: TaskCreate, response: Response, service: TaskServiceDep) -> TaskRead:
    task = await service.create_task(data)
    response.headers["Location"] = f"/tasks/{task.id}"
    return TaskRead.model_validate(task)


@router.get("/{task_id}", responses=NOT_FOUND_RESPONSE)
async def get_task(task_id: TaskId, service: TaskServiceDep) -> TaskRead:
    return TaskRead.model_validate(await service.get_task(task_id))


@router.get("")
async def list_tasks(
    params: Annotated[TaskListParams, Query()], service: TaskServiceDep
) -> TaskPageRead:
    return TaskPageRead.model_validate(await service.list_tasks(params))


@router.patch("/{task_id}", responses=WRITE_RESPONSES)
async def update_task(task_id: TaskId, data: TaskUpdate, service: TaskServiceDep) -> TaskRead:
    return TaskRead.model_validate(await service.update_task(task_id, data))


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT, responses=WRITE_RESPONSES)
async def delete_task(
    task_id: TaskId,
    version: Annotated[int, Query(ge=1, description="The task version the client last read")],
    service: TaskServiceDep,
) -> None:
    await service.delete_task(task_id, version)


async def task_not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": "Task not found"})


async def version_conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT, content={"detail": "Task has been modified"}
    )


def create_app(
    settings: Settings | None = None, *, repository: TaskRepository | None = None
) -> FastAPI:
    """Build the app. A supplied repository is used as-is; otherwise one is opened from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if repository is not None:
            app.state.task_repository = repository
            yield
            return
        async with open_task_repository(settings or Settings.from_env()) as opened:
            app.state.task_repository = opened
            yield

    app = FastAPI(title="Tasks API", version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(TaskNotFoundError, task_not_found_handler)
    app.add_exception_handler(VersionConflictError, version_conflict_handler)
    app.include_router(router)
    return app


app = create_app()
