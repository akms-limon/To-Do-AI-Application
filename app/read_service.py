from app.repository import get_task


def matches_filters(task: dict, filters: dict) -> bool:
    for field, expected in filters.items():
        if field == "status" and expected == "unfinished":
            if task["status"] not in {"to_do", "in_progress"}:
                return False
        elif task[field] != expected:
            return False

    return True


def list_filtered_tasks(conn, filters: dict) -> list[dict]:
    rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()

    return [dict(row) for row in rows if matches_filters(dict(row), filters)]


def semantic_search(query: str, filters: dict, limit: int | None = None):
    from app.embeddings import embed_texts
    from app.search_service import get_collection, synchronize_index
    from app.repository import list_tasks
    import os
    import re

    words = lambda text: set(re.findall(r"\w+", text.casefold()))
    tokens = words(query) - {"task", "tasks", "about", "related", "to", "the", "my"}
    eligible = [t for t in list_tasks() if matches_filters(t, filters)]
    if not eligible:
        return []
    synchronize_index()
    collection = get_collection()
    count = collection.count()
    if not count:
        return []
    results = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=count,
        include=["distances", "metadatas"],
    )
    by_id = {str(t["id"]): t for t in eligible}
    cutoff = float(os.getenv("SEARCH_MAX_DISTANCE", "0.60"))
    candidates = []
    for task_id, distance, metadata in zip(
        results["ids"][0], results["distances"][0], results["metadatas"][0]
    ):
        task = by_id.get(task_id)
        if task is None or not metadata or metadata.get("version") != task["version"]:
            continue
        lexical = bool(tokens) and tokens.issubset(
            words(task["title"] + " " + (task["description"] or ""))
        )
        if lexical or float(distance) <= cutoff:
            candidates.append(
                {"task": task, "distance": float(distance), "lexical": lexical}
            )
    candidates.sort(key=lambda item: (not item["lexical"], item["distance"]))
    return candidates if limit is None else candidates[:limit]


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
        return "No relevant tasks match your search and filters."
    return format_task_list([match["task"] for match in matches])
