import json
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.database import initialize_database
from app.graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints.db"


def read_message(prompt):
    while True:
        value = input(prompt).strip()

        if 1 <= len(value) <= 2000:
            return value

        print("Enter 1–2000 characters.")


def read_resume_answer(payload):
    print("\nAssistant:", payload["question"])

    return read_message(
        "Your reply (or cancel): "
    )


def main():
    initialize_database()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("TaskFlow workflow check")
    print("This version executes task changes in your database.")
    print("Use a new conversation ID after changing graph structure.\n")

    default_thread = f"demo-{uuid4().hex[:8]}"

    thread_id = input(
        f"Conversation ID [{default_thread}]: "
    ).strip() or default_thread

    print(f"Conversation ID: {thread_id}")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_PATH)
    ) as checkpointer:
        graph = build_graph(checkpointer)
        snapshot = graph.get_state(config)

        pending_interrupts = [
            item
            for task in snapshot.tasks
            for item in task.interrupts
        ]

        # Old preview checkpoints lack the new execution state.
        if snapshot.next and not snapshot.values.get("request_id"):
            print(
                "This unfinished conversation belongs to the older preview "
                "workflow. Restart this script and use a new conversation ID."
            )
            return

        if pending_interrupts:
            answer = read_resume_answer(
                pending_interrupts[0].value
            )

            result = graph.invoke(
                Command(resume=answer),
                config,
            )

        elif snapshot.next:
            print("Retrying unfinished workflow...")

            result = graph.invoke(
                None,
                config,
            )

        else:
            message = read_message("Your instruction: ")

            result = graph.invoke(
                {
                    "message": message,
                    "history": [],
                    "proposal": None,
                    "reply": "",
                    "cancelled": False,
                    "prepared_actions": [],
                    "request_id": str(uuid4()),
                    "execution_result": None,
                    "execution_error": None,
                    "index_warning": None,
                },
                config,
            )

        while result.get("__interrupt__"):
            answer = read_resume_answer(
                result["__interrupt__"][0].value
            )

            result = graph.invoke(
                Command(resume=answer),
                config,
            )

        print("\nAssistant:", result["reply"])

        if result.get("prepared_actions"):
            print("\nPrepared actions:")
            print(
                json.dumps(
                    result["prepared_actions"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if result.get("execution_result"):
            print("\nSaved execution result:")
            print(
                json.dumps(
                    result["execution_result"],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if result.get("execution_error"):
            print("\nReview the error before submitting another request.")


if __name__ == "__main__":
    main()