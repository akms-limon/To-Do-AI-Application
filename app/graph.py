import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from app.action_validation import validate_action
from app.executor import execute_changes
from app.llm import interpret_message
from app.search_service import synchronize_index
from app.task_resolver import RequestCancelled, resolve_target


logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    message: str
    history: list[dict[str, str]]
    proposal: dict | None
    reply: str
    cancelled: bool
    prepared_actions: list[dict]
    request_id: str
    execution_result: dict | None
    execution_error: str | None
    index_warning: str | None


def interpret_node(state: AgentState):
    proposal = interpret_message(
        state["message"],
        history=state["history"],
    )

    return {
        "proposal": proposal.model_dump(mode="json"),
        "prepared_actions": [],
        "execution_result": None,
        "execution_error": None,
        "index_warning": None,
        "history": [
            *state["history"],
            {
                "role": "human",
                "content": state["message"],
            },
        ],
    }


def route_interpretation(state: AgentState):
    outcome = state["proposal"]["outcome"]

    if outcome == "clarify":
        return "clarify"

    if outcome == "unsupported":
        return "finish"

    return "prepare"


def clarify_node(state: AgentState):
    question = state["proposal"]["message"]

    while True:
        answer = interrupt({
            "kind": "clarification",
            "question": question,
        })

        if not isinstance(answer, str):
            question = "Please reply with text, or type cancel."
            continue

        answer = answer.strip()

        if not 1 <= len(answer) <= 2000:
            question = "Enter 1–2000 characters, or type cancel."
            continue

        break

    if answer.lower() in {"cancel", "stop"}:
        return {
            "cancelled": True,
            "proposal": None,
            "prepared_actions": [],
            "history": [],
            "reply": "Cancelled. No tasks were changed.",
        }

    return {
        "message": answer,
        "history": [
            *state["history"],
            {
                "role": "assistant",
                "content": question,
            },
        ],
    }


def route_clarification(state: AgentState):
    return "finish" if state["cancelled"] else "interpret"


def prepare_node(state: AgentState):
    try:
        actions = [
            validate_action(raw)
            for raw in state["proposal"]["actions"]
        ]

        prepared = []

        for action in actions:
            if action["operation"] in {"update_task", "delete_task"}:
                task = resolve_target(action)

                action = {
                    **action,
                    "task_id": task["id"],
                    "target_query": None,
                    "expected_version": task["version"],
                    "resolved_title": task["title"],
                }

            prepared.append(action)

        return {
            "prepared_actions": prepared,
            "reply": "Actions validated and targets resolved.",
        }

    except RequestCancelled:
        return {
            "cancelled": True,
            "prepared_actions": [],
            "proposal": None,
            "history": [],
            "reply": "Cancelled. No tasks were changed.",
        }

    except ValueError as error:
        if isinstance(error, ValidationError):
            details = "; ".join(
                item["msg"]
                for item in error.errors(include_input=False)
            )
        else:
            details = str(error)

        return {
            "prepared_actions": [],
            "proposal": {
                **state["proposal"],
                "outcome": "clarify",
                "message": (
                    f"I couldn't prepare this request: {details} "
                    "Please correct it, or type cancel."
                ),
            },
        }


def route_preparation(state: AgentState):
    if state["cancelled"]:
        return "finish"

    if state["proposal"]["outcome"] == "clarify":
        return "clarify"

    return "execute"


def execute_node(state: AgentState):
    try:
        result = execute_changes(
            state["request_id"],
            state["prepared_actions"],
        )
    except ValueError as error:
        return {
            "execution_result": None,
            "execution_error": str(error),
        }

    return {
        "execution_result": result,
        "execution_error": None,
    }


def synchronize_node(state: AgentState):
    if state.get("execution_error") or not state.get("execution_result"):
        return {"index_warning": None}

    try:
        synchronize_index()
        return {"index_warning": None}
    except Exception:
        logger.exception("Search index synchronization failed")

        return {
            "index_warning": (
                "Task changes were saved, but search synchronization failed. "
                "Run the index synchronization script before semantic search."
            )
        }


def finish_node(state: AgentState):
    if state["cancelled"]:
        return {"reply": "Cancelled. No tasks were changed."}

    if state.get("execution_error"):
        return {
            "reply": (
                "The execution was rejected: "
                + state["execution_error"]
            )
        }

    if state.get("execution_result"):
        messages = [
            item["message"]
            for item in state["execution_result"]["results"]
        ]

        if state.get("index_warning"):
            messages.append(state["index_warning"])

        return {"reply": "\n".join(messages)}

    proposal = state.get("proposal")

    return {
        "reply": (
            proposal["message"]
            if proposal
            else "No action was executed."
        )
    }


def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("interpret", interpret_node)
    builder.add_node("prepare", prepare_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("execute", execute_node)
    builder.add_node("synchronize", synchronize_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "interpret")

    builder.add_conditional_edges(
        "interpret",
        route_interpretation,
        {
            "clarify": "clarify",
            "prepare": "prepare",
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "prepare",
        route_preparation,
        {
            "clarify": "clarify",
            "execute": "execute",
            "finish": "finish",
        },
    )

    builder.add_conditional_edges(
        "clarify",
        route_clarification,
        {
            "interpret": "interpret",
            "finish": "finish",
        },
    )

    builder.add_edge("execute", "synchronize")
    builder.add_edge("synchronize", "finish")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)