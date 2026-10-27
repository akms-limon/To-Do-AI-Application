import json
import logging
import sqlite3
from pathlib import Path
from threading import Lock
from uuid import UUID

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.graph import build_graph


logger = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
CHECKPOINT_PATH = DATA_DIRECTORY / "checkpoints.db"
RECEIPT_PATH = DATA_DIRECTORY / "chat_receipts.db"

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
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def save_receipt(conn, message_id, status_code, body):
    conn.execute(
        """
        UPDATE receipts
        SET status_code = ?, response_json = ?
        WHERE message_id = ?
        """,
        (status_code, json.dumps(body), message_id),
    )
    conn.commit()


def handle_chat(payload: ChatRequest):
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    thread_id = str(payload.thread_id)
    message_id = str(payload.message_id)

    # Serialize local graph requests to avoid concurrent resumes.
    with CHAT_LOCK:
        conn = sqlite3.connect(RECEIPT_PATH, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS receipts (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status_code INTEGER,
                    response_json TEXT
                )
            """)
            conn.commit()

            receipt = conn.execute(
                "SELECT * FROM receipts WHERE message_id = ?",
                (message_id,),
            ).fetchone()

            if receipt:
                if (
                    receipt["thread_id"] != thread_id
                    or receipt["message"] != payload.message
                ):
                    raise ChatServiceError(
                        409, "This message ID was used for different input."
                    )

                if receipt["status_code"] is None:
                    raise ChatServiceError(
                        409,
                        "This message has an unfinished receipt. "
                        "Check saved tasks before recovery; do not resubmit "
                        "the operation with a new message ID.",
                    )

                body = json.loads(receipt["response_json"])

                if receipt["status_code"] != 200:
                    raise ChatServiceError(
                        receipt["status_code"], body["detail"]
                    )

                return body

            # Commit before execution. An interrupted request will not be
            # silently executed a second time.
            conn.execute(
                """
                INSERT INTO receipts (message_id, thread_id, message)
                VALUES (?, ?, ?)
                """,
                (message_id, thread_id, payload.message),
            )
            conn.commit()

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 30,
            }

            try:
                with SqliteSaver.from_conn_string(
                    str(CHECKPOINT_PATH)
                ) as checkpointer:
                    graph = build_graph(checkpointer)
                    snapshot = graph.get_state(config)

                    pending = [
                        item
                        for task in snapshot.tasks
                        for item in task.interrupts
                    ]

                    if pending:
                        result = graph.invoke(
                            Command(resume=payload.message),
                            config,
                        )

                    elif snapshot.next:
                        raise ChatServiceError(
                            409,
                            "This conversation has an unfinished workflow. "
                            "Inspect it before continuing.",
                        )

                    else:
                        result = graph.invoke(
                            {
                                "message": payload.message,
                                "history": [],
                                "proposal": None,
                                "reply": "",
                                "cancelled": False,
                                "prepared_actions": [],
                                "request_id": message_id,
                                "execution_result": None,
                                "execution_error": None,
                                "index_warning": None,
                            },
                            config,
                        )

                    interruptions = result.get("__interrupt__")

                    if interruptions:
                        body = {
                            "status": "clarification",
                            "reply": interruptions[0].value["question"],
                            "results": [],
                        }
                    else:
                        execution = result.get("execution_result") or {}

                        body = {
                            "status": (
                                "error"
                                if result.get("execution_error")
                                else "completed"
                            ),
                            "reply": result["reply"],
                            "results": execution.get("results", []),
                        }

            except ChatServiceError as error:
                save_receipt(
                    conn,
                    message_id,
                    error.status_code,
                    {"detail": error.detail},
                )
                raise

            except Exception:
                logger.exception("Chat workflow failed")

                detail = (
                    "The workflow could not finish. Check the server terminal "
                    "and refresh the task board before retrying an operation."
                )

                save_receipt(
                    conn, message_id, 502, {"detail": detail}
                )
                raise ChatServiceError(502, detail)

            save_receipt(conn, message_id, 200, body)
            return body

        finally:
            conn.close()