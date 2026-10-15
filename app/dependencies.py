"""FastAPI dependency providers."""

from typing import Annotated

from fastapi import Depends, Request

from app.auth import get_current_user_id
from app.repository import TaskRepository
from app.service import TaskService


def get_task_repository(request: Request) -> TaskRepository:
    repository: TaskRepository = request.app.state.task_repository
    return repository


CurrentUserId = Annotated[int, Depends(get_current_user_id)]


def get_task_service(
    repository: Annotated[TaskRepository, Depends(get_task_repository)],
    user_id: CurrentUserId,
) -> TaskService:
    return TaskService(repository, user_id)


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
