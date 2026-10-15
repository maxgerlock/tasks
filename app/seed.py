"""Seed the configured SQLite database with sample tasks.

Usage: uv run seed [--count N] [--users 1,2,3] [--reset | --recreate] [--random-seed N]

The database location comes from TASKS_DB_PATH, the same setting the server uses.
"""

import argparse
import asyncio
import random
from collections import Counter
from collections.abc import Sequence

from app import db
from app.auth import parse_user_id
from app.config import Settings, StorageBackend
from app.models import Task
from app.repository import SQLiteTaskRepository

DEFAULT_COUNT = 250
DEFAULT_USER_IDS = (1, 2, 3)
COMPLETED_RATIO = 0.4

_VERBS = ["Buy", "Call", "Clean", "Email", "Fix", "Plan", "Read", "Review", "Schedule", "Write"]
_OBJECTS = [
    "groceries",
    "the dentist",
    "the garage",
    "quarterly report",
    "bike tire",
    "weekend trip",
    "book club chapter",
    "pull request",
    "team offsite",
    "thank-you notes",
]


def generate_description(rng: random.Random) -> str:
    return f"{rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"


async def seed(
    settings: Settings,
    *,
    count: int,
    user_ids: Sequence[int] = DEFAULT_USER_IDS,
    reset: bool = False,
    recreate: bool = False,
    rng: random.Random,
) -> list[Task]:
    """Create ``count`` tasks, assigning owners from ``user_ids`` in round-robin order."""
    async with db.connect(settings.db_path) as connection, db.transaction(connection):
        if recreate:
            await db.drop(connection)
        await db.initialize(connection)
        if reset:
            await db.reset(connection)
        repository = SQLiteTaskRepository(connection)
        return [
            await repository.create(
                user_id=user_ids[i % len(user_ids)],
                description=generate_description(rng),
                completed=rng.random() < COMPLETED_RATIO,
            )
            for i in range(count)
        ]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _user_ids(value: str) -> list[int]:
    try:
        ids = [parse_user_id(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of positive integers"
        ) from None
    return list(dict.fromkeys(ids))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed the tasks database with sample tasks.")
    parser.add_argument(
        "--count",
        type=_positive_int,
        default=DEFAULT_COUNT,
        help=f"number of tasks to create (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--users",
        type=_user_ids,
        default=list(DEFAULT_USER_IDS),
        help="comma-separated user IDs that own the tasks, assigned round-robin "
        f"(default: {','.join(map(str, DEFAULT_USER_IDS))})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete all existing tasks and restart IDs at 1 before seeding",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the tasks table first; use after the Task model changes",
    )
    parser.add_argument(
        "--random-seed", type=int, default=None, help="seed for reproducible sample data"
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    if settings.storage is not StorageBackend.SQLITE:
        parser.error("seeding requires TASKS_STORAGE=sqlite; in-memory data would be discarded")

    tasks = asyncio.run(
        seed(
            settings,
            count=args.count,
            user_ids=args.users,
            reset=args.reset,
            recreate=args.recreate,
            rng=random.Random(args.random_seed),
        )
    )
    completed = sum(task.completed for task in tasks)
    per_user = Counter(task.user_id for task in tasks)
    print(
        f"Seeded {len(tasks)} tasks ({completed} completed) into {settings.db_path}, "
        f"IDs {tasks[0].id}-{tasks[-1].id}"
    )
    for user_id, owned in sorted(per_user.items()):
        print(f"  user {user_id}: {owned} tasks")


if __name__ == "__main__":
    main()
