import json
from datetime import datetime, timezone

from app.action_validation import validate_action
from app.database import connection
from app.read_service import (
    format_search_results,
    format_task_list,
    list_filtered_tasks,
    semantic_search,
)
from app.schemas import TaskCreate, TaskUpdate


class ExecutionConflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_prepared_action(action: dict) -> dict:
    # Strip internal resolution fields before validating the public proposal.
    public_fields = {
        "operation": action["operation"],
        "task_id": action.get("task_id"),
        "target_query": action.get("target_query"),
        "values": action.get("values", {}),
        "filters": action.get("filters", {}),
    }

    validated = validate_action(public_fields)

    return {
        **action,
        **validated,
    }


def execute_read_only(actions: list[dict]):
    results = []

    for action in actions:
        operation = action["operation"]

        if operation == "list_tasks":
            with connection() as conn:
                tasks = list_filtered_tasks(conn, action["filters"])

            results.append({
                "operation": operation,
                "tasks": tasks,
                "message": format_task_list(tasks),
            })

        elif operation == "search_tasks":
            matches = semantic_search(
                action["target_query"],
                action["filters"],
            )

            results.append({
                "operation": operation,
                "matches": matches,
                "message": format_search_results(matches),
            })

    return {"results": results}


def execute_changes(request_id: str, actions: list[dict]):
    if not request_id:
        raise ValueError("Missing request ID.")

    if not 1 <= len(actions) <= 5:
        raise ValueError("Provide between one and five actions.")

    actions_json = json.dumps(actions, sort_keys=True)

    # Return a prior committed mutation result before revalidating dates.
    with connection() as conn:
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

    validated = [
        validate_prepared_action(action)
        for action in actions
    ]

    mutation_operations = {
        "create_task", "update_task", "delete_task"
    }

    has_mutations = any(
        action["operation"] in mutation_operations
        for action in validated
    )

    has_search = any(
        action["operation"] == "search_tasks"
        for action in validated
    )

    if not has_mutations:
        return execute_read_only(validated)

    # Semantic search uses a separate store. Until mixed-store ordering is
    # implemented, reject the whole mixed batch rather than partially execute.
    if has_search:
        raise ValueError(
            "Please send semantic search and task changes as separate "
            "messages for now. No changes were made."
        )

    with connection() as conn:
        conn.execute("BEGIN IMMEDIATE")

        # Recheck after acquiring the write lock.
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

        initial_versions = {}

        for action in validated:
            if action["operation"] in {"update_task", "delete_task"}:
                task_id = action["task_id"]
                expected = action.get("expected_version")

                if type(expected) is not int or expected < 1:
                    raise ValueError(
                        f"Missing or invalid version for task #{task_id}."
                    )

                if task_id in initial_versions:
                    if initial_versions[task_id] != expected:
                        raise ExecutionConflict(
                            f"Inconsistent versions for task #{task_id}."
                        )

                initial_versions[task_id] = expected

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
                    "Submit the instruction again."
                )

        results = []

        for action in validated:
            operation = action["operation"]
            timestamp = now()

            if operation == "list_tasks":
                tasks = list_filtered_tasks(conn, action["filters"])

                results.append({
                    "operation": operation,
                    "tasks": tasks,
                    "message": format_task_list(tasks),
                })
                continue

            if operation == "create_task":
                values = TaskCreate.model_validate(
                    action["values"]
                ).model_dump(mode="json")

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
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()

                results.append({
                    "operation": operation,
                    "task": dict(row),
                    "message": (
                        f"Created task #{task_id}: {row['title']}."
                    ),
                })
                continue

            task_id = action["task_id"]

            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()

            if row is None:
                raise ExecutionConflict(
                    f"Task #{task_id} was removed earlier in this batch."
                )

            if operation == "delete_task":
                conn.execute(
                    "DELETE FROM tasks WHERE id = ?",
                    (task_id,),
                )

                results.append({
                    "operation": operation,
                    "task_id": task_id,
                    "message": (
                        f"Deleted task #{task_id}: {row['title']}."
                    ),
                })
                continue

            values = TaskUpdate.model_validate(
                action["values"]
            ).model_dump(mode="json", exclude_unset=True)

            changes = {
                key: value
                for key, value in values.items()
                if row[key] != value
            }

            if changes:
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
                    "SELECT * FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()

                message = (
                    f"Updated task #{task_id}: {row['title']}."
                )
            else:
                message = (
                    f"Task #{task_id} already has those values."
                )

            results.append({
                "operation": operation,
                "task": dict(row),
                "message": message,
            })

        result = {"results": results}

        conn.execute(
            """
            INSERT INTO executions (
                request_id, actions_json, result_json
            )
            VALUES (?, ?, ?)
            """,
            (
                request_id,
                actions_json,
                json.dumps(result),
            ),
        )

        return result