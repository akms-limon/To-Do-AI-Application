# TaskFlow

A local single-user task manager with a FastAPI task board and conversational task management using Groq, LangChain, LangGraph, SQLite and ChromaDB.

## Run

Use the existing Python 3.11 uv environment and configure `.env`:

```dotenv
GROQ_API_KEY=your-key-here
GROQ_MODEL=openai/gpt-oss-20b
SEARCH_MAX_DISTANCE=0.60
```

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8003
```

Open http://127.0.0.1:8003/ (API documentation: /docs). Keep the service local; this demo has no authentication. MiniLM's ONNX model downloads on first embedding use and then runs on CPU. No local generative LLM or external embedding API is used.

If upgrading an existing installation, follow [UPGRADE.md](UPGRADE.md) first. It explains data preservation and the new conversation store.

## Request architecture

1. Chat receipts identify each submitted message and support same-ID retries.
2. LangGraph provides the interpreter with current reference context and any pending proposal.
3. Groq produces a Pydantic-validated structured proposal, not executable code.
4. Python validates fields/dates and resolves IDs, title matches, filters and semantic candidates.
5. Group operations or deletions display a frozen preview. Ambiguous single targets display the exact proposed change alongside the candidate.
6. After approval, SQLite rechecks target versions and commits the complete batch atomically with an execution receipt.
7. Chroma synchronizes after commit. An index failure does not misreport a successful task operation as failed.
8. Completed plans clear while task/result references remain available within the chat.

SQLite is authoritative; Chroma stores 384-dimensional MiniLM vectors for titles/descriptions and metadata for retrieval.

## Examples

- Create a high-priority task to finish the demo tomorrow.
- Show unfinished tasks related to football.
- Make all of them medium priority.
- Change the football task to In progress.
- Delete all completed tasks.
- Actually, only update task #7.
- Yes, but make it low priority.
- Cancel.

The last two examples are corrections/cancellation, not unconditional approvals. Only an unambiguous approval of the displayed plan executes it.

## Checks

```bash
uv run python -m scripts.check_workflows
# Optional: calls Groq, does not write tasks
uv run python -m scripts.check_language
# Interactive CLI: DOES execute task changes
uv run python -m scripts.check_graph
# Repair search index
uv run python -m scripts.sync_search_index
```

The workflow checks use Python's standard unittest library, temporary databases and mocked model/vector responses. No pytest dependency is required. See UPGRADE.md for verified coverage and remaining limitations.
