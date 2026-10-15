"""Pydantic request/response models for the HTTP API."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

DESCRIPTION_MAX_LENGTH = 500
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100

Description = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=DESCRIPTION_MAX_LENGTH),
]


def _parse_query_bool(value: object) -> object:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("Must be 'true' or 'false'")


QueryBool = Annotated[bool, BeforeValidator(_parse_query_bool)]
Cursor = Annotated[int, Field(ge=0)]
Version = Annotated[StrictInt, Field(ge=1)]


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Description


class TaskUpdate(BaseModel):
    """Partial update, conditional on ``version`` matching the task's current version.

    Omitted fields are unchanged; explicit nulls are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    version: Version
    description: Description | None = None
    completed: StrictBool | None = None

    @field_validator("description", "completed", mode="before")
    @classmethod
    def reject_null(cls, value: object) -> object:
        # Defaults bypass validation, so this only fires for an explicitly supplied null.
        if value is None:
            raise ValueError("Field may not be null")
        return value

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> Self:
        if not self.model_fields_set - {"version"}:
            raise ValueError("At least one field to change must be provided")
        return self


class TaskListParams(BaseModel):
    """Query parameters for listing tasks.

    ``after`` pages forward in ascending ID order; ``before`` pages backward in descending
    order. Both are exclusive bounds and at most one may be given.
    """

    model_config = ConfigDict(extra="forbid")

    completed: QueryBool | None = None
    after: Cursor | None = None
    before: Cursor | None = None
    limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT)

    @model_validator(mode="after")
    def at_most_one_cursor(self) -> Self:
        if self.after is not None and self.before is not None:
            raise ValueError("Only one of 'after' and 'before' may be provided")
        return self


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    description: str
    completed: bool
    version: int


class TaskPageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: list[TaskRead]
    next_cursor: int | None
    has_more: bool
