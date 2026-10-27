import json
from datetime import datetime, timezone

from app.database import connection
from app.schemas import TaskCreate, TaskUpdate


class ExecutionConflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def execute_changes(request_id: str, actions: list[dict]):
    if not request_id:
        raise ValueError("Missing request ID.")

    if not 1 <= len(actions) <= 5:
        raise ValueError("Provide between one and five actions.")

    actions_json = json.dumps(actions, sort_keys=True)

    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")

        previous = conn.execute(
            "SELECT * FROM executions WHERE request_id = ?",
            (request_id,),
        ).fetchone()

        if previous:
            if previous["actions_json"] != actions_json:
                raise ExecutionConflict(
                    "This request ID was already used for different actions."
                )

            return json.loads(previous["result_json"])

        # Revalidate at execution time, including today's date.
        validated = []
        initial_versions = {}

        for action in actions:
            operation = action["operation"]

            if operation == "create_task":
                values = TaskCreate.model_validate(
                    action["values"]
                ).model_dump(mode="json")

            elif operation == "update_task":
                values = TaskUpdate.model_validate(
                    action["values"]
                ).model_dump(mode="json", exclude_unset=True)

            elif operation == "delete_task":
                values = {}

            else:
                raise ValueError(
                    "This execution step supports task changes only. "
                    "Listing and search execution will be added next."
                )

            if operation in {"update_task", "delete_task"}:
                task_id = action["task_id"]
                expected = action["expected_version"]

                if task_id in initial_versions:
                    if initial_versions[task_id] != expected:
                        raise ExecutionConflict(
                            f"Inconsistent versions for task #{task_id}."
                        )

                initial_versions[task_id] = expected

            validated.append({**action, "values": values})

        # Check the original versions once before applying the ordered batch.
        for task_id, expected in initial_versions.items():
            row = conn.execute(
                "SELECT version FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()

            if row is None:
                raise ExecutionConflict(
                    f"Task #{task_id} no longer exists."
                )

            if row["version"] != expected:
                raise ExecutionConflict(
                    f"Task #{task_id} changed after selection. "
                    "Submit the instruction again to use its current state."
                )

        results = []

        for action in validated:
            operation = action["operation"]
            values = action["values"]
            timestamp = now()

            if operation == "create_task":
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
                        timestamp,
                        timestamp,
                    ),
                )

                task_id = cursor.lastrowid
                row = conn.execute(
                    "SELECT * FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()

                results.append({
                    "operation": operation,
                    "task": dict(row),
                    "message": f"Created task #{task_id}: {row['title']}.",
                })
                continue

            task_id = action["task_id"]
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

            if row is None:
                raise ExecutionConflict(
                    f"Task #{task_id} was removed earlier in this batch."
                )

            if operation == "delete_task":
                conn.execute(
                    "DELETE FROM tasks WHERE id = ?", (task_id,)
                )
                results.append({
                    "operation": operation,
                    "task_id": task_id,
                    "message": f"Deleted task #{task_id}: {row['title']}.",
                })
                continue

            changes = {
                key: value
                for key, value in values.items()
                if row[key] != value
            }

            if changes:
                # Keys come exclusively from the validated TaskUpdate model.
                assignments = ", ".join(
                    f"{key} = ?" for key in changes
                )

                conn.execute(
                    f"""
                    UPDATE tasks
                    SET {assignments},
                        updated_at = ?,
                        version = version + 1
                    WHERE id = ?
                    """,
                    (*changes.values(), timestamp, task_id),
                )

                row = conn.execute(
                    "SELECT * FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()

                message = f"Updated task #{task_id}: {row['title']}."
            else:
                message = f"Task #{task_id} already has those values."

            results.append({
                "operation": operation,
                "task": dict(row),
                "message": message,
            })

        result = {"results": results}

        conn.execute(
            """
            INSERT INTO executions (request_id, actions_json, result_json)
            VALUES (?, ?, ?)
            """,
            (request_id, actions_json, json.dumps(result)),
        )

        return result