from datetime import date

from app.agent_schemas import ActionProposal
from app.schemas import TaskCreate, TaskUpdate


STATUSES = {"to_do", "in_progress", "done", "unfinished"}
PRIORITIES = {"low", "medium", "high"}


def validate_action(raw: dict) -> dict:
    action = ActionProposal.model_validate(raw)
    operation = action.operation

    if operation not in {"list_tasks", "search_tasks"} and action.filters:
        raise ValueError("Filters are only supported for listing and search.")

    if operation in {"list_tasks", "search_tasks", "delete_task"}:
        if action.values:
            raise ValueError(f"{operation} cannot contain task changes.")

    if operation == "create_task":
        if action.task_id is not None or action.target_query:
            raise ValueError("Creating a task cannot target an existing task.")

        values = TaskCreate.model_validate(action.values).model_dump(
            mode="json"
        )

    elif operation == "update_task":
        values = TaskUpdate.model_validate(action.values).model_dump(
            mode="json",
            exclude_unset=True,
        )

    else:
        values = {}

    if operation in {"update_task", "delete_task"}:
        if action.task_id is None and not (
            action.target_query and action.target_query.strip()
        ):
            raise ValueError("Specify the task ID or title.")

    if operation in {"list_tasks", "search_tasks"}:
        if action.task_id is not None:
            raise ValueError("Use filters or a search query for listing/search.")

        if operation == "search_tasks" and not (
            action.target_query and action.target_query.strip()
        ):
            raise ValueError("Provide a search query.")

        if operation == "list_tasks" and action.target_query:
            raise ValueError("Use search_tasks for a meaning-based query.")

    filters = action.filters

    if "status" in filters and filters["status"] not in STATUSES:
        raise ValueError("Invalid status filter.")

    if "priority" in filters and filters["priority"] not in PRIORITIES:
        raise ValueError("Invalid priority filter.")

    if "due_date" in filters:
        value = filters["due_date"]
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("Date filters must use YYYY-MM-DD.")

        if parsed.isoformat() != value:
            raise ValueError("Date filters must use YYYY-MM-DD.")

    return {
        **action.model_dump(mode="json"),
        "values": values,
    }