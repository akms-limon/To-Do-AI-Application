"""Single-process local chat transport with persisted receipts and safe retries."""

import json
import logging
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import UUID

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.graph import build_graph, initial_state

logger = logging.getLogger(__name__)
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
# Versioned graph store: never resume old graph topology with new nodes.
CHECKPOINT_PATH = DATA_DIRECTORY / "checkpoints_v3.db"
RECEIPT_PATH = DATA_DIRECTORY / "chat_receipts_v3.db"
CHAT_LOCK = Lock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thread_id: UUID
    message_id: UUID
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value):
        return value.strip() if isinstance(value, str) else value


class ChatServiceError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail
        super().__init__(detail)


def save_receipt(conn, message_id, status_code, body):
    conn.execute(
        "UPDATE receipts SET status_code=?, response_json=? WHERE message_id=?",
        (status_code, json.dumps(body), message_id),
    )
    conn.commit()


def handle_chat(payload):
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    thread_id, message_id = str(payload.thread_id), str(payload.message_id)
    with CHAT_LOCK:
        conn = sqlite3.connect(RECEIPT_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS receipts (
                message_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                message TEXT NOT NULL, status_code INTEGER, response_json TEXT)""")
            conn.commit()
            previous = conn.execute(
                "SELECT * FROM receipts WHERE message_id=?", (message_id,)
            ).fetchone()
            if previous:
                if (
                    previous["thread_id"] != thread_id
                    or previous["message"] != payload.message
                ):
                    raise ChatServiceError(
                        409, "This message ID belongs to different input."
                    )
                if previous["status_code"] == 200:
                    return json.loads(previous["response_json"])
            unfinished = conn.execute(
                "SELECT message_id FROM receipts WHERE thread_id=? AND (status_code IS NULL OR status_code=502) AND message_id != ? LIMIT 1",
                (thread_id, message_id),
            ).fetchone()
            if unfinished:
                raise ChatServiceError(
                    409,
                    "Retry the previous failed message first. It may have saved changes.",
                )
            if not previous:
                conn.execute(
                    "INSERT INTO receipts(message_id,thread_id,message) VALUES(?,?,?)",
                    (message_id, thread_id, payload.message),
                )
                conn.commit()
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
            try:
                with SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
                    graph = build_graph(saver)
                    snapshot = graph.get_state(config)
                    values = snapshot.values or {}
                    pending = [i for t in snapshot.tasks for i in t.interrupts]
                    delivered = values.get("delivery_id") == message_id
                    if previous and delivered:
                        # Resume failed nodes, or reconstruct a response after receipt-write failure.
                        result = (
                            graph.invoke(None, config)
                            if snapshot.next and not pending
                            else values
                        )
                        if pending:
                            result = {**values, "__interrupt__": pending}
                    elif pending:
                        result = graph.invoke(
                            Command(
                                resume={
                                    "message": payload.message,
                                    "message_id": message_id,
                                }
                            ),
                            config,
                        )
                    elif snapshot.next:
                        # Resume the same checkpoint after a transient workflow failure.
                        result = graph.invoke(None, config)
                    else:
                        result = graph.invoke(
                            initial_state(
                                payload.message, message_id, values.get("memory")
                            ),
                            config,
                        )
                    interruptions = result.get("__interrupt__")
                    body = (
                        {
                            "status": "clarification",
                            "reply": interruptions[0].value["question"],
                            "results": [],
                        }
                        if interruptions
                        else {
                            "status": (
                                "error"
                                if result.get("execution_error")
                                else "completed"
                            ),
                            "reply": result["reply"],
                            "results": (result.get("execution_result") or {}).get(
                                "results", []
                            ),
                        }
                    )
            except ChatServiceError:
                raise
            except Exception as error:
                logger.exception("Chat workflow failed")
                detail = "The workflow could not finish. Retry the SAME message using Retry; saved changes will not be duplicated."
                save_receipt(conn, message_id, 502, {"detail": detail})
                raise ChatServiceError(502, detail) from error
            save_receipt(conn, message_id, 200, body)
            return body
        finally:
            conn.close()
