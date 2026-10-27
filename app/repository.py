from datetime import datetime, timezone

from app.database import connection
from app.schemas import TaskCreate, TaskUpdate


EDITABLE_FIELDS = {
    "title", "description", "status", "priority", "due_date"
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def create_task(payload: TaskCreate):
    values = payload.model_dump(mode="json")
    now = timestamp()

    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tasks (
                title, description, status, priority, due_date,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["title"],
                values["description"],
                values["status"],
                values["priority"],
                values["due_date"],
                now,
                now,
            ),
        )

        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

        return dict(row)


def list_tasks():
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY id DESC"
        ).fetchall()

        return [dict(row) for row in rows]


def get_task(task_id: int):
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

        return dict(row) if row else None


def update_task(task_id: int, payload: TaskUpdate):
    changes = payload.model_dump(mode="json", exclude_unset=True)

    # Column names cannot be SQL parameters, so explicitly allowlist them.
    if not changes or not set(changes).issubset(EDITABLE_FIELDS):
        raise ValueError("Invalid update fields.")

    with connection() as conn:
        # Keep the read and write within one write transaction.
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

        if existing is None:
            return None

        # Avoid rewriting values that are already set.
        actual_changes = {
            name: value
            for name, value in changes.items()
            if existing[name] != value
        }

        if not actual_changes:
            return dict(existing)

        assignments = ", ".join(
            f"{name} = ?" for name in actual_changes
        )

        conn.execute(
            f"""
            UPDATE tasks
            SET {assignments},
                updated_at = ?,
                version = version + 1
            WHERE id = ?
            """,
            (*actual_changes.values(), timestamp(), task_id),
        )

        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

        return dict(row)


def delete_task(task_id: int):
    with connection() as conn:
        cursor = conn.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task_id,),
        )

        return cursor.rowcount == 1