from app.embeddings import embed_texts
from app.repository import get_task
from app.search_service import get_collection, synchronize_index


def matches_filters(task: dict, filters: dict) -> bool:
    for field, expected in filters.items():
        if field == "status" and expected == "unfinished":
            if task["status"] not in {"to_do", "in_progress"}:
                return False
        elif task[field] != expected:
            return False

    return True


def list_filtered_tasks(conn, filters: dict) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM tasks ORDER BY id DESC"
    ).fetchall()

    return [
        dict(row)
        for row in rows
        if matches_filters(dict(row), filters)
    ]


def semantic_search(query: str, filters: dict, limit: int = 5):
    # Repair stale or missing entries before searching.
    synchronize_index()

    collection = get_collection()
    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=count,
        include=["distances", "metadatas"],
    )

    candidates = []

    for task_id, distance, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        task = get_task(int(task_id))

        if task is None:
            continue

        if metadata["version"] != task["version"]:
            continue

        if not matches_filters(task, filters):
            continue

        candidates.append({
            "task": task,
            "distance": float(distance),
        })

        if len(candidates) == limit:
            break

    return candidates


def format_task_list(tasks: list[dict]) -> str:
    if not tasks:
        return "No tasks match these filters."

    labels = {
        "to_do": "To do",
        "in_progress": "In progress",
        "done": "Done",
    }

    lines = [f"Found {len(tasks)} task(s):"]

    for task in tasks:
        due_date = task["due_date"] or "No due date"

        lines.append(
            f"#{task['id']}: {task['title']} | "
            f"{labels[task['status']]} | "
            f"{task['priority']} priority | {due_date}"
        )

    return "\n".join(lines)


def format_search_results(matches: list[dict]) -> str:
    if not matches:
        return "No current indexed tasks match these filters."

    lines = [
        "Nearest semantic candidates "
        "(ranked by meaning; relevance is not guaranteed):"
    ]

    for match in matches:
        task = match["task"]

        lines.append(
            f"#{task['id']}: {task['title']} | "
            f"{task['status']} | "
            f"distance={match['distance']:.3f}"
        )

    return "\n".join(lines)