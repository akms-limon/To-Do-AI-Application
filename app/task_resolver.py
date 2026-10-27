from langgraph.types import interrupt

from app.repository import get_task, list_tasks
from app.search_service import search_tasks


class RequestCancelled(Exception):
    pass


def choose_candidate(candidates: list[dict]):
    options = {
        task["id"]: task["title"]
        for task in candidates
    }

    lines = [
        "Choose a task by its ID, or type cancel.",
        "These are candidates; none has been selected automatically.",
    ]

    for task_id, title in options.items():
        lines.append(f"#{task_id}: {title}")

    question = "\n".join(lines)

    while True:
        answer = interrupt({
            "kind": "task_selection",
            "question": question,
        })

        if not isinstance(answer, str):
            question = "Reply with one of the listed task IDs, or cancel."
            continue

        answer = answer.strip()

        if answer.lower() in {"cancel", "stop", "none"}:
            raise RequestCancelled()

        candidate_id = answer.removeprefix("#")

        if not candidate_id.isascii() or not candidate_id.isdigit():
            question = "Enter a numeric task ID from the list, or cancel."
            continue

        task_id = int(candidate_id)

        if task_id not in options:
            question = "That ID was not offered. Choose a listed ID, or cancel."
            continue

        return task_id


def resolve_target(action: dict):
    if action["task_id"] is not None:
        task = get_task(action["task_id"])

        if task is None:
            raise ValueError(f"Task #{action['task_id']} does not exist.")

        return task

    query = action["target_query"].strip()

    exact_matches = [
        task
        for task in list_tasks()
        if task["title"].strip().casefold() == query.casefold()
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if exact_matches:
        candidates = exact_matches
    else:
        candidates = [
            match["task"]
            for match in search_tasks(query, limit=5)
        ]

    if not candidates:
        raise ValueError(
            "No current indexed candidate was found. "
            "Specify an existing task ID or exact title."
        )

    selected_id = choose_candidate(candidates)

    # Re-read after clarification: the task might have changed.
    task = get_task(selected_id)

    if task is None:
        raise ValueError(
            "The selected task was deleted. Specify another task."
        )

    return task