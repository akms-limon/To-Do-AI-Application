from functools import lru_cache
from pathlib import Path

import chromadb

from app.embeddings import embed_texts
from app.repository import list_tasks

CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma"


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    return client.get_or_create_collection(
        name="tasks_minilm_v1",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def task_document(task):
    description = task["description"] or ""
    return f"{task['title']}\n{description}".strip()


def task_metadata(task):
    return {
        "status": task["status"],
        "priority": task["priority"],
        "due_date": task["due_date"] or "",
        "version": task["version"],
    }


def synchronize_index():
    """Reconcile the search index with the current SQLite tasks."""
    collection = get_collection()
    tasks = list_tasks()

    indexed = collection.get(include=["documents", "metadatas"])

    existing = {
        task_id: {
            "document": document,
            "metadata": metadata,
        }
        for task_id, document, metadata in zip(
            indexed["ids"],
            indexed["documents"],
            indexed["metadatas"],
        )
    }

    current_ids = {str(task["id"]) for task in tasks}

    counts = {
        "embedded": 0,
        "metadata_updated": 0,
        "deleted": 0,
    }

    for task in tasks:
        task_id = str(task["id"])
        document = task_document(task)
        metadata = task_metadata(task)
        previous = existing.get(task_id)

        if previous is None or previous["document"] != document:
            collection.upsert(
                ids=[task_id],
                documents=[document],
                embeddings=embed_texts([document]),
                metadatas=[metadata],
            )
            counts["embedded"] += 1

        elif previous["metadata"] != metadata:
            collection.update(
                ids=[task_id],
                metadatas=[metadata],
            )
            counts["metadata_updated"] += 1

    stale_ids = list(set(existing) - current_ids)

    if stale_ids:
        collection.delete(ids=stale_ids)
        counts["deleted"] = len(stale_ids)

    return counts


def search_tasks(query: str, limit: int = 5):
    query = query.strip()

    if not query or len(query) > 2000:
        raise ValueError("Search query must contain 1–2000 characters.")

    if not 1 <= limit <= 20:
        raise ValueError("Search limit must be between 1 and 20.")

    from app.read_service import semantic_search

    return semantic_search(query, {}, limit=limit)
