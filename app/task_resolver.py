import re
import unicodedata

from langgraph.types import interrupt

from app.repository import get_task, list_tasks
from app.search_service import search_tasks, synchronize_index


# Starting values; further retrieval evaluation is still needed.
AUTO_MATCH_MAX_DISTANCE = 0.35
AUTO_MATCH_MIN_GAP = 0.15
CANDIDATE_MAX_DISTANCE = 0.65

GENERIC_WORDS = {
    "task", "tasks", "the", "a", "an", "my", "please",
}

YES_REPLIES = {
    "yes", "yes please", "yes that one", "yes this one",
    "correct", "that's correct", "that is correct",
    "confirm", "confirmed", "haan", "ha", "hya",
    "হ্যাঁ", "হ্যা", "জি",
}

CANCEL_REPLIES = {
    "no", "nope", "cancel", "stop", "none",
    "none of these", "na", "না",
}

POSITIONS = {
    "first": 0,
    "first one": 0,
    "the first one": 0,
    "second": 1,
    "second one": 1,
    "the second one": 1,
    "third": 2,
    "third one": 2,
    "the third one": 2,
    "fourth": 3,
    "fourth one": 3,
    "the fourth one": 3,
    "fifth": 4,
    "fifth one": 4,
    "the fifth one": 4,
}


class RequestCancelled(Exception):
    pass


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def normalize_reply(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'")
    return " ".join(text.strip().rstrip(".!?।").split())


def title_tokens(text: str) -> set[str]:
    return {
        word
        for word in normalize_title(text).split()
        if word not in GENERIC_WORDS
    }


def refresh_selected_task(task: dict) -> dict:
    current = get_task(task["id"])

    if current is None:
        raise ValueError(
            "That task was deleted. Please specify another task."
        )

    if current["version"] != task["version"]:
        raise ValueError(
            f"Task #{task['id']} changed during selection. "
            "Please restate the request."
        )

    return current


def choose_candidate(candidates: list[dict]) -> dict:
    by_id = {task["id"]: task for task in candidates}

    options = "\n".join(
        f"{position}. #{task['id']}: "
        f"{task['title']} ({task['status']})"
        for position, task in enumerate(candidates, start=1)
    )

    if len(candidates) == 1:
        base_question = (
            f"Is this the task you mean?\n{options}\n"
            "Reply yes to select it, or no to cancel."
        )
    else:
        base_question = (
            f"Which task do you mean?\n{options}\n"
            "Reply with its title, task ID, or a position such as "
            "'the second one'. Type cancel to stop."
        )

    question = base_question

    while True:
        answer = interrupt({
            "kind": "task_selection",
            "question": question,
        })

        if not isinstance(answer, str):
            question = "Please reply with text.\n" + base_question
            continue

        answer = answer.strip()

        if not 1 <= len(answer) <= 2000:
            question = "Enter 1–2000 characters.\n" + base_question
            continue

        reply = normalize_reply(answer)

        if reply in CANCEL_REPLIES:
            raise RequestCancelled()

        if reply in YES_REPLIES:
            if len(candidates) == 1:
                return refresh_selected_task(candidates[0])

            question = (
                "'Yes' does not identify one of these tasks. "
                "Please choose a title, ID, or position.\n"
                + base_question
            )
            continue

        if reply in POSITIONS:
            position = POSITIONS[reply]

            if position < len(candidates):
                return refresh_selected_task(candidates[position])

            question = (
                "That position is not in the list.\n"
                + base_question
            )
            continue

        # Bare numbers always mean task IDs, not list positions.
        id_match = re.fullmatch(
            r"(?:task\s*)?#?([0-9]+)",
            reply,
        )

        if id_match:
            task_id = int(id_match.group(1))

            if task_id in by_id:
                return refresh_selected_task(by_id[task_id])

            question = (
                "That task ID is not in the list. To select by position, "
                "say 'the first one', 'the second one', etc.\n"
                + base_question
            )
            continue

        exact_matches = [
            task
            for task in candidates
            if normalize_title(task["title"]) == normalize_title(answer)
        ]

        if len(exact_matches) == 1:
            return refresh_selected_task(exact_matches[0])

        if len(exact_matches) > 1:
            question = (
                "Several tasks have that title. Use an ID or position.\n"
                + base_question
            )
            continue

        tokens = title_tokens(answer)

        partial_matches = [
            task
            for task in candidates
            if tokens and tokens.issubset(title_tokens(task["title"]))
        ]

        if len(partial_matches) == 1:
            return refresh_selected_task(partial_matches[0])

        question = (
            "I couldn't identify one task from that reply.\n"
            + base_question
        )


def resolve_target(action: dict):
    if action["task_id"] is not None:
        task = get_task(action["task_id"])

        if task is None:
            raise ValueError(
                f"Task #{action['task_id']} does not exist."
            )

        return task

    query = (action.get("target_query") or "").strip()

    if not query:
        raise ValueError("Specify a task title or ID.")

    # Prevent unimplemented bulk requests from becoming a single-task match.
    if normalize_title(query) in {
        "all", "all tasks", "all my tasks", "every task",
        "everything", "my tasks", "these tasks", "those tasks",
    }:
        raise ValueError(
            "Bulk changes are not connected yet. "
            "Please identify one task for now."
        )

    tasks = list_tasks()

    if not tasks:
        raise ValueError("There are no tasks to update or delete.")

    normalized_query = normalize_title(query)

    exact_matches = [
        task
        for task in tasks
        if normalize_title(task["title"]) == normalized_query
    ]

    if len(exact_matches) == 1:
        return refresh_selected_task(exact_matches[0])

    if len(exact_matches) > 1:
        return choose_candidate(exact_matches)

    query_tokens = title_tokens(query)

    if not query_tokens:
        raise ValueError("Provide a meaningful task title or ID.")

    title_matches = [
        task
        for task in tasks
        if query_tokens.issubset(title_tokens(task["title"]))
    ]

    if len(title_matches) == 1:
        return refresh_selected_task(title_matches[0])

    if len(title_matches) > 1:
        return choose_candidate(title_matches)

    synchronize_index()
    matches = search_tasks(query, limit=20)

    if not matches:
        raise ValueError(
            "No current task matched. Provide a clearer title or ID."
        )

    best = matches[0]
    second_distance = (
        matches[1]["distance"] if len(matches) > 1 else None
    )

    strong_match = best["distance"] <= AUTO_MATCH_MAX_DISTANCE
    clearly_separated = (
        second_distance is None
        or second_distance - best["distance"] >= AUTO_MATCH_MIN_GAP
    )

    if strong_match and clearly_separated:
        return refresh_selected_task(best["task"])

    candidates = [
        match["task"]
        for match in matches
        if match["distance"] <= CANDIDATE_MAX_DISTANCE
    ]

    if not candidates:
        raise ValueError(
            "I couldn't confidently match that description. "
            "Provide a clearer title or ID."
        )

    return choose_candidate(candidates)