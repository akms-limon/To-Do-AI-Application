from datetime import date

from app.agent_schemas import ActionProposal
from app.schemas import TaskCreate, TaskUpdate

STATUSES = {"to_do", "in_progress", "done", "unfinished"}
PRIORITIES = {"low", "medium", "high"}


def validate_action(raw: dict) -> dict:
    action = ActionProposal.model_validate(raw)

    # Both operations are read-only. A supplied text query requires search,
    # even when the model labels the request as a listing. Preserve filters
    # and run every remaining validation check after this narrow correction.
    if action.operation in {"list_tasks", "search_tasks"}:
        query = (action.target_query or "").strip() or None
        operation = action.operation
        if operation == "list_tasks" and query is not None:
            operation = "search_tasks"
        action = action.model_copy(
            update={"operation": operation, "target_query": query}
        )

    operation = action.operation
    filters = action.filters
    mutation = operation in {"update_task", "delete_task"}
    if not mutation and (action.scope != "single" or action.reference):
        raise ValueError("Scope/reference only apply to updates and deletions.")
    if action.task_id is not None and action.target_query:
        raise ValueError("Use either an explicit task ID or a target query, not both.")
    if action.scope != "single" and action.task_id is not None:
        raise ValueError("A group scope cannot also specify one task ID.")
    if action.scope == "all" and (filters or action.target_query):
        raise ValueError("Use matching scope for filtered or descriptive groups.")
    if action.scope == "matching" and not (filters or action.target_query):
        raise ValueError("Matching scope requires filters or a description.")
    if action.scope == "references" and not action.reference:
        raise ValueError("Identify last_tasks or last_results for contextual targets.")
    if action.reference and action.scope != "references":
        raise ValueError("Reference requires references scope.")

    if operation == "create_task" and filters:
        raise ValueError("Creation cannot contain target filters.")

    if operation in {"list_tasks", "search_tasks", "delete_task"}:
        if action.values:
            raise ValueError(f"{operation} cannot contain task changes.")

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

    if operation == "create_task":
        if action.task_id is not None or action.target_query:
            raise ValueError("Creating a task cannot target an existing task.")

        values = TaskCreate.model_validate(action.values).model_dump(mode="json")

    elif operation == "update_task":
        values = TaskUpdate.model_validate(action.values).model_dump(
            mode="json", exclude_unset=True
        )

    else:
        values = {}

    if operation in {"update_task", "delete_task"}:
        has_query = bool(action.target_query and action.target_query.strip())

        if (
            action.scope == "single"
            and action.task_id is None
            and not has_query
            and not filters
        ):
            raise ValueError(
                "Specify a task ID, title, description, or target filters."
            )

    if operation in {"list_tasks", "search_tasks"}:
        if action.task_id is not None:
            raise ValueError("Use filters or a query for listing/search.")

        if operation == "search_tasks" and not (
            action.target_query and action.target_query.strip()
        ):
            raise ValueError("Provide a search query.")

        if operation == "list_tasks" and action.target_query:
            raise ValueError("Use search_tasks for a meaning-based query.")

    return {
        **action.model_dump(mode="json"),
        "values": values,
    }
