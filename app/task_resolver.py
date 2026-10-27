import re
import unicodedata

from langgraph.types import interrupt

from app.repository import get_task, list_tasks
from app.search_service import search_tasks, synchronize_index


# Conservative starting values, not calibrated guarantees.
# Lower cosine distance means greater similarity.
AUTO_MATCH_MAX_DISTANCE = 0.35
AUTO_MATCH_MIN_GAP = 0.15
CANDIDATE_MAX_DISTANCE = 0.65

GENERIC_WORDS = {
    "task",
    "tasks",
    "the",
    "a",
    "an",
    "my",
    "please",
}


class RequestCancelled(Exception):
    pass


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


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
            "Please restate the request so its current details can be checked."
        )

    return current


def choose_candidate(candidates: list[dict]) -> dict:
    by_id = {
        task["id"]: task
        for task in candidates
    }

    options = "\n".join(
        f"#{task['id']}: {task['title']} ({task['status']})"
        for task in candidates
    )

    introduction = (
        "Which task do you mean?"
        if len(candidates) > 1
        else "Is this the task you mean?"
    )

    base_question = (
        f"{introduction}\n"
        f"{options}\n"
        "Reply with its title or ID, or type cancel."
    )

    question = base_question

    while True:
        answer = interrupt({
            "kind": "task_selection",
            "question": question,
        })

        if not isinstance(answer, str):
            question = "Please enter a title or ID.\n" + base_question
            continue

        answer = answer.strip()

        if not answer or len(answer) > 2000:
            question = "Enter 1–2000 characters.\n" + base_question
            continue

        if answer.casefold() in {"cancel", "stop", "none"}:
            raise RequestCancelled()

        # Accept "7", "#7", or "task 7".
        id_match = re.fullmatch(
            r"(?:task\s*)?#?([0-9]+)",
            answer,
            flags=re.IGNORECASE,
        )

        if id_match:
            task_id = int(id_match.group(1))

            if task_id in by_id:
                return refresh_selected_task(by_id[task_id])

            question = (
                "That ID is not one of the offered tasks.\n"
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
                "Several tasks have that title. Please use an ID.\n"
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
            "That reply does not identify one offered task clearly.\n"
            + base_question
        )


def resolve_target(action: dict):
    # 1. Explicit ID.
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

    tasks = list_tasks()

    if not tasks:
        raise ValueError("There are no tasks to update or delete.")

    # 2. Unique exact title, ignoring case and punctuation.
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

    # 3. All meaningful query words appear in the title.
    # "football task" matches "Play football".
    query_tokens = title_tokens(query)

    if not query_tokens:
        raise ValueError(
            "Please provide a meaningful task title or an ID."
        )

    title_matches = [
        task
        for task in tasks
        if query_tokens.issubset(title_tokens(task["title"]))
    ]

    if len(title_matches) == 1:
        return refresh_selected_task(title_matches[0])

    if len(title_matches) > 1:
        # Do not use semantic ranking to silently choose between
        # several plausible title matches.
        return choose_candidate(title_matches)

    # 4. Semantic fallback, using a synchronized index.
    synchronize_index()
    matches = search_tasks(query, limit=20)

    if not matches:
        raise ValueError(
            "No current task matched. Please provide a clearer title or ID."
        )

    best = matches[0]

    second_distance = (
        matches[1]["distance"]
        if len(matches) > 1
        else None
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
            "I couldn't confidently match that description to a task. "
            "Please provide a clearer title or ID."
        )

    return choose_candidate(candidates)