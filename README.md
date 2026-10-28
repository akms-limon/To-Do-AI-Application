# TaskFlow — AI Task Manager

TaskFlow is a single-page task manager controlled primarily through an AI chat interface. Users can create, search, update and delete tasks using natural-language instructions.

## Features

- Create tasks with a title, optional description, status, priority and optional due date.
- List and search tasks using natural language.
- Update task details, change status, complete tasks and reopen completed tasks.
- Identify tasks by ID, title or description.
- Ask for clarification when multiple tasks match.
- Preview and confirm bulk updates and deletions.
- Remember recent task references within the current conversation.
- Preserve tasks after browser refresh or server restart.
- Send messages with Enter; use Shift+Enter for a new line.

### Task fields

| Field | Supported values |
|---|---|
| Title | Required, up to 200 characters |
| Description | Optional, up to 2,000 characters |
| Status | `to_do`, `in_progress`, `done` |
| Priority | `low`, `medium`, `high` |
| Due date | Optional, `YYYY-MM-DD` |

New tasks default to `to_do` status and `medium` priority.

## Technology choices

| Technology | Purpose and reason |
|---|---|
| Python and FastAPI | Backend API and request validation with a small setup |
| HTML, CSS and JavaScript | Responsive single-page interface without a frontend framework |
| Groq | Hosted language-model inference without running a generative model locally |
| LangChain Groq integration | Connects the application to Groq and parses structured responses |
| LangGraph | Manages interpretation, clarification, confirmation and execution |
| Pydantic | Validates structured actions and task fields |
| SQLite | Persistent task storage without a separate database server |
| ChromaDB | Stores task embeddings for semantic retrieval |
| MiniLM | Generates local embeddings without an external embedding API |

The embedding model is `all-MiniLM-L6-v2`, using Chroma's ONNX implementation. It generates 384-dimensional vectors on CPU.

MiniLM is used for embeddings, not conversational text generation. Groq handles language interpretation.

## Requirements

- Python 3.11
- uv
- Git
- Internet access for Groq requests and the initial embedding-model download
- A Groq API key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/akms-limon/To-Do-AI-Application.git
cd To-Do-AI-Application
```

### 2. Set up Python and dependencies

```bash
uv python install 3.11
uv sync --locked
```

### 3. Configure environment variables

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GROQ_API_KEY=your_actual_key_here
GROQ_MODEL=openai/gpt-oss-20b
SEARCH_MAX_DISTANCE=0.60
```

Keep the real API key only in `.env`. Never commit it.

The model name is configurable. Your Groq account must have access to the configured model.

### 4. Start the application

```bash
uv run uvicorn app.main:app --reload
```

Open:

- Application: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs

Keep the terminal running while using the application.

SQLite tables are initialized during application startup. The embedding model downloads when embeddings are first needed, so the first operation requiring embeddings can take longer.

## Example instructions

```text
Create a high-priority task to finish the demo tomorrow.
```

```text
Show unfinished tasks related to documentation.
```

```text
Change the priority of the football task to low.
```

```text
Move the demo task to In progress and mark the weekly report Done.
```

```text
Make all unfinished tasks high priority.
```

```text
Delete all completed tasks.
```

After a search, users can refer to the displayed results:

```text
Make all of them medium priority.
```

Bulk operations and deletions require confirmation. Review the affected task IDs before replying `yes`. Reply `cancel` to stop a pending operation.

## How a request works

1. The browser sends the message with a conversation ID and a unique message ID.
2. Recognized simple priority/status instructions can be interpreted locally. Other instructions are sent to Groq with relevant conversation context.
3. The interpreter produces a structured proposal containing operations, targets, filters and requested changes.
4. Pydantic and Python validation check supported fields, values, dates and operation scope.
5. The resolver identifies tasks using IDs, title matching, filters or semantic candidates.
6. The graph asks for missing information or confirmation when required.
7. The executor checks task versions and saves the complete mutation batch in a SQLite transaction.
8. The search index synchronizes with the saved task state.
9. The application returns the result and refreshes the task board.

The language model proposes actions. It does not directly execute SQL or modify the database.

## Task storage and semantic search

SQLite is the source of truth for task records. Each task has a stable integer ID and a version number.

ChromaDB stores embeddings generated from task titles and descriptions, along with metadata used to detect outdated entries.

Search combines:

- Word matching in titles and descriptions.
- Semantic similarity from MiniLM embeddings.
- Structured filters such as status, priority and due date.

For example, a query about buying ingredients can retrieve a grocery-shopping task even when the wording differs.

Semantic distance is not a confidence percentage. The distance cutoff is configurable and may require tuning for different task collections.

Task updates and deletions are synchronized with the search index. The index can also be rebuilt manually.

## Clarification and confirmation

- Clear single-task creation and updates execute after validation.
- Ambiguous targets produce a choice of actual matching tasks.
- Uncertain single-task matches show the task and proposed change together.
- Bulk changes and deletions require confirmation.
- Replies such as `both` and `the first one` refer to the currently displayed candidates or plan.
- Corrections replace the relevant pending details.
- A replacement request discards the unexecuted previous plan.

Completed operations clear their pending plan. Lightweight references remain available for follow-up instructions such as “Make it high priority.”

## Validation and reliability

The backend checks:

- Required titles and field-length limits.
- Allowed statuses and priorities.
- Valid calendar dates.
- Newly assigned deadlines that are in the past.
- Missing or ambiguous targets.
- Unsupported fields and invalid operation combinations.
- Task versions before applying confirmed changes.
- Duplicate targets within a mutation batch.

Completed and overdue tasks remain editable. Changing another field does not require replacing an existing overdue date.

Mutation batches are atomic: either all their changes commit or the transaction rolls back.

Saved execution receipts prevent a retried request from duplicating an already committed mutation. If a response fails, use the UI's Retry button to resend the same message ID.

If task changes commit but search synchronization fails, the application reports the saved changes separately from the search-index warning.

## Testing

### Offline workflow checks

```bash
uv run python -m scripts.check_workflows
```

These checks use temporary databases and mocked model/vector responses. They do not modify normal application data or consume Groq API quota.

Coverage includes:

- Ambiguous task selection and confirmation.
- Contextual references and replacement requests.
- Bulk updates and deletions.
- Cancellation and stale task versions.
- Atomic execution and duplicate-request protection.
- Search-operation normalization.
- Empty-container normalization in AI responses.

Offline tests verify application logic. They do not prove that the live model understands every possible instruction.

## Project structure

```text
app/
├── main.py                 # FastAPI routes and application startup
├── agent_schemas.py        # Structured AI proposals
├── schemas.py              # Task input validation
├── action_validation.py    # Operation validation and normalization
├── llm.py                  # Groq integration and interpretation prompt
├── conversation.py         # Local shortcuts and reference checks
├── graph.py                # Conversation workflow
├── task_resolver.py        # Task matching and candidate selection
├── executor.py             # Transactional action execution
├── chat_service.py         # Conversation persistence and message receipts
├── repository.py           # Task database operations
├── database.py             # SQLite connection and initialization
├── embeddings.py           # MiniLM embeddings
├── search_service.py       # Chroma index management
├── read_service.py         # Filtered listing and retrieval
└── static/                 # Frontend files

scripts/                    # Checks and maintenance utilities
prompts.md                  # Development prompts, failures and corrections
.env.example                # Configuration placeholders
pyproject.toml              # Project dependencies
uv.lock                     # Dependency lockfile
```

## Development prompts and debugging

See [prompts.md](prompts.md) for meaningful development prompts, failures encountered and the corrections made.