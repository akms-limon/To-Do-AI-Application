"""Resolve targets without interrupting; graph checkpoints every offered set."""

import re
import unicodedata
from difflib import SequenceMatcher

from app.read_service import matches_filters, semantic_search
from app.repository import get_task, list_tasks

YES_REPLIES = {
    "yes",
    "yes please",
    "yes that one",
    "correct",
    "confirm",
    "confirmed",
    "haan",
    "ha",
    "hya",
    "হ্যাঁ",
    "হ্যা",
    "জি",
}
CANCEL_REPLIES = {"no", "nope", "cancel", "stop", "none", "none of these", "na", "না"}


def normalize_title(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def normalize_reply(text):
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().strip().rstrip(".!?।").split()
    )


def refresh_selected_task(task):
    current = get_task(task["id"])
    if current is None or current["version"] != task["version"]:
        raise ValueError(
            f"Task #{task['id']} changed or was deleted. Please restate the request."
        )
    return current


def select_from_reply(answer, candidates):
    reply = normalize_reply(answer)
    if reply in YES_REPLIES:
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("Please identify one task; yes alone is ambiguous.")
    match = re.fullmatch(r"(?:task\s*)?#?(\d+)", reply)
    if match:
        matches = [t for t in candidates if t["id"] == int(match[1])]
        if not matches:
            raise ValueError("That ID is not among the offered candidates.")
        return matches[0]
    positions = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
    position = reply.removeprefix("the ").removesuffix(" one")
    if position in positions:
        index = positions[position]
        if index >= len(candidates):
            raise ValueError("That position is not in the list.")
        return candidates[index]
    matches = [
        t for t in candidates if normalize_title(t["title"]) == normalize_title(answer)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Those tasks share a title; choose an ID.")
    return None  # Full natural-language replies return to the interpreter.


def resolve_targets(action, memory):
    """Return (resolved tasks, candidates). Uncertain single targets need selection."""
    scope = action.get("scope", "single")
    filters = action.get("filters", {})
    query = (action.get("target_query") or "").strip()
    tasks = [t for t in list_tasks() if matches_filters(t, filters)]
    if scope == "references":
        ids = memory.get(action["reference"], [])
        if not ids:
            raise ValueError(
                "There is no previous task set to refer to. Name the tasks or search first."
            )
        current = {t["id"]: t for t in list_tasks()}
        if any(i not in current for i in ids):
            raise ValueError("A referenced task was deleted. Please search again.")
        tasks = [current[i] for i in ids if matches_filters(current[i], filters)]
    if scope == "single" and action.get("task_id") is not None:
        task = get_task(action["task_id"])
        if task is None or not matches_filters(task, filters):
            raise ValueError("That task does not exist or does not match the filters.")
        return [task], []
    if scope == "all":
        return tasks, []
    if query:
        normalized = normalize_title(query)
        if normalized in {
            "all",
            "all tasks",
            "all my tasks",
            "everything",
            "those",
            "those tasks",
            "all of them",
        }:
            raise ValueError(
                "Specify group scope explicitly: all database tasks or the previous results."
            )
        exact = [t for t in tasks if normalize_title(t["title"]) == normalized]
        if scope == "single" and exact:
            return (exact, []) if len(exact) == 1 else ([], exact)
        tokens = set(normalized.split()) - {"task", "tasks", "the", "my"}
        title_matches = [
            t
            for t in tasks
            if tokens and tokens.issubset(set(normalize_title(t["title"]).split()))
        ]
        if scope == "single" and title_matches:
            return (
                (title_matches, []) if len(title_matches) == 1 else ([], title_matches)
            )
        eligible = {t["id"] for t in tasks}
        fuzzy = [
            t
            for t in tasks
            if SequenceMatcher(None, normalized, normalize_title(t["title"])).ratio()
            >= 0.70
        ]
        if scope == "single" and fuzzy:
            return [], fuzzy
        matches = semantic_search(query, filters)
        tasks = [m["task"] for m in matches if m["task"]["id"] in eligible]
        if scope == "single":
            if not tasks:
                raise ValueError(
                    "No sufficiently clear match. Provide a clearer title or task ID."
                )
            return [], tasks  # Semantic similarity alone never authorizes a mutation.
    if scope in {"matching", "references"}:
        return tasks, []
    if len(tasks) == 1:
        return tasks, []
    if not tasks:
        raise ValueError("No tasks match the requested criteria.")
    return [], tasks
