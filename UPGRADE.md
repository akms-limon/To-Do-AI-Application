# TaskFlow contextual chat upgrade

## Install into your existing clone

1. Finish or cancel any old pending request. If an old request failed after a possible write, inspect the board before repeating it.
2. Stop Uvicorn with Ctrl+C. Back up `data/` while the server is stopped and commit your current source changes.
3. Extract this ZIP to a separate folder. Copy its `app/` and `scripts/` folders and this README/UPGRADE guide into your existing repository, replacing matching source files. Do not replace/delete your `.env`, `.git`, `.venv`, or `data/`.
4. No new runtime dependencies were added; `pyproject.toml` and `uv.lock` are unchanged.
5. Run `uv run python -m scripts.check_workflows`. These checks use temporary databases and mocked model/search calls. They do not change your tasks or call Groq.
6. Run `uv run uvicorn app.main:app --reload --port 8003` and hard-refresh http://127.0.0.1:8003/ with Ctrl+Shift+R.

Your SQLite tasks and Chroma collection are retained. New chat checkpoints/receipts use `checkpoints_v2.db` and `chat_receipts_v2.db`. Browser session keys also use v2. Old files are left intact; old pending operations are NOT migrated or resumed. The new conversation starts without old chat memory.

## Optional real-language check

With your existing GROQ_API_KEY configured, run:

```bash
uv run python -m scripts.check_language
```

This calls Groq and consumes API quota, but does not execute task operations. It prints PASS/REVIEW for six English/Banglish interpretation scenarios. Report REVIEW results before assuming model comprehension is reliable.

Then exercise the real UI on disposable demo tasks:

1. Create a high-priority task to finish the demo tomorrow. Verify title, priority and due date.
2. Show unfinished tasks related to play. Verify relevance and that completed tasks are excluded.
3. Make all of them high priority. Verify the preview contains exactly the displayed result IDs; reply no first and verify nothing changed.
4. Repeat and reply yes. Verify before/after fields.
5. Delete all completed tasks. Inspect the preview; cancel unless those are disposable tasks.
6. Start a bulk change, then say "Actually, only change task #X to low". Verify a new plan replaces the old one.
7. Create a task, then "Make it high priority". Verify the latest created task is referenced.
8. Misspell a unique title, inspect the offered change, and confirm once.

## What changed

- Explicit single/all/matching/references scopes; bulk update and deletion.
- Group mutations and all deletions require previews; exact single creates/updates execute directly.
- Uncertain single match: one combined target/action confirmation.
- Stable IDs and expected versions are checkpointed before confirmation; newly created tasks never join an already displayed plan.
- Duplicate target changes within a batch are rejected; maximum 100 prepared actions, never silently truncated.
- Atomic writes and saved execution receipts prevent partial batches and duplicate mutations.
- Corrections/replacements return to interpretation without executing the old plan.
- Completed plans clear; small task/result references remain for the next message.
- Search uses lexical coverage plus cosine distance filtering; semantic matches alone do not auto-authorize a single-task mutation.
- Same-message retries recover failed graph nodes or a response lost after commit.
- Browser retains up to 60 chat bubbles in the current tab so a pending question remains visible after refresh.
- Existing Enter-to-send and Shift+Enter-to-newline are retained.

## Validation and limits

29 offline regression checks passed during development on Python 3.12.14, LangGraph 1.2.18, SQLite checkpointer 3.1.2, LangChain Groq 1.1.3 and Pydantic 2.13.4. Your project targets Python 3.11 and its existing dependency lock; run the included checks in your environment too.

No live Groq response or actual MiniLM/Chroma embedding generation was tested in the development environment. Offline tests mock interpretation and vector distances; they verify orchestration, data integrity and filtering logic, not universal natural-language understanding or embedding quality.

The semantic cutoff defaults to SEARCH_MAX_DISTANCE=0.60 (cosine distance); it is a starting point, not a confidence percentage or a calibrated guarantee. Tune against representative task/query examples. Exact word coverage can retain a relevant task above that cutoff.

Mixed read-and-write batches require separate messages. Newest/oldest database selection, recurrence, reminders, authentication and multi-user hosting are not implemented. Chat runs with one server process; the lock is process-local. Search currently scans all indexed candidates and is intended for a small internship dataset. Structured date filters support one exact date, not ranges. Descriptions are limited to 2,000 characters.

Confirmations apply to chat operations. Existing direct CRUD endpoints in /docs remain direct developer operations and do not use chat confirmation.
