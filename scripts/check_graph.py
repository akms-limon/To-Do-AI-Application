import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints.db"


def read_message(prompt):
    while True:
        value = input(prompt).strip()

        if 1 <= len(value) <= 2000:
            return value

        print("Enter 1–2000 characters.")


def main():
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    thread_id = input(
        "Conversation ID [demo-1]: "
    ).strip() or "demo-1"

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

        if pending_interrupts:
            question = pending_interrupts[0].value["question"]
            print("Assistant:", question)

            answer = read_message("Your clarification: ")
            result = graph.invoke(Command(resume=answer), config)

        elif snapshot.next:
            print("Retrying unfinished workflow...")
            result = graph.invoke(None, config)

        else:
            message = read_message("Your instruction: ")

            result = graph.invoke(
                {
                    "message": message,
                    "history": [],
                    "proposal": None,
                    "reply": "",
                    "cancelled": False,
                },
                config,
            )

        while result.get("__interrupt__"):
            question = result["__interrupt__"][0].value["question"]
            print("Assistant:", question)

            answer = read_message(
                "Your clarification (or cancel): "
            )

            result = graph.invoke(Command(resume=answer), config)

        print("\nAssistant:", result["reply"])

        if result.get("proposal"):
            print(json.dumps(result["proposal"], indent=2))

        print("\nPreview only: no task data was changed.")


if __name__ == "__main__":
    main()