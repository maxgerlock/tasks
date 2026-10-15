"""Caller identification.

For now the caller states their user ID in a request header; nothing verifies the claim.
A real authentication scheme can replace ``get_current_user_id`` without touching routes.
"""

import re
from typing import Annotated

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

USER_ID_HEADER = "X-User-ID"
# SQLite INTEGER columns hold signed 64-bit values.
MAX_USER_ID = 2**63 - 1
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]{0,18}")

# Registers the header as a security scheme so /docs offers an "Authorize" button for it.
_user_id_header = APIKeyHeader(
    name=USER_ID_HEADER,
    auto_error=False,
    description="ID of the user making the request: a positive integer.",
)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": USER_ID_HEADER},
    )


def parse_user_id(value: str) -> int:
    """Parse a canonical positive decimal integer, raising ValueError for anything else."""
    if not _POSITIVE_INTEGER.fullmatch(value) or int(value) > MAX_USER_ID:
        raise ValueError(f"{USER_ID_HEADER} must be a positive integer")
    return int(value)


async def get_current_user_id(
    request: Request, _documented: Annotated[str | None, Security(_user_id_header)]
) -> int:
    # Read the raw header: APIKeyHeader reports an empty value as missing and hides duplicates.
    values = request.headers.getlist(USER_ID_HEADER)
    if not values:
        raise _unauthorized(f"Missing {USER_ID_HEADER} header")
    if len(values) > 1:
        raise _unauthorized(f"Multiple {USER_ID_HEADER} headers")
    try:
        return parse_user_id(values[0])
    except ValueError as exc:
        raise _unauthorized(str(exc)) from None
