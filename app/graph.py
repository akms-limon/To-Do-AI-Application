from typing import TypedDict
from pydantic import ValidationError

from app.action_validation import validate_action
from app.task_resolver import RequestCancelled, resolve_target
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.llm import interpret_message


class AgentState(TypedDict):
    message: str
    history: list[dict[str, str]]
    proposal: dict | None
    reply: str
    cancelled: bool
    prepared_actions: list[dict]


def interpret_node(state: AgentState):
    proposal = interpret_message(
        state["message"],
        history=state["history"],
    )

    return {
        "proposal": proposal.model_dump(mode="json"),
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

    answer = interrupt({
        "kind": "clarification",
        "question": question,
    })

    if not isinstance(answer, str):
        raise ValueError("Clarification reply must be text.")

    answer = answer.strip()

    if not answer or len(answer) > 2000:
        raise ValueError("Reply must contain 1–2000 characters.")

    if answer.lower() in {"cancel", "stop"}:
        return {
            "cancelled": True,
            "proposal": None,
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


def finish_node(state: AgentState):
    if state["cancelled"]:
        return {"reply": "Cancelled. No tasks were changed."}

    if state["proposal"]["outcome"] == "unsupported":
        return {"reply": state["proposal"]["message"]}

    return {
        "reply": (
            "Actions validated and targets resolved. "
            "This is still a preview; no tasks were changed."
        )
    }

def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("interpret", interpret_node)
    builder.add_node("prepare", prepare_node)
    builder.add_node("clarify", clarify_node)
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

    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)

def prepare_node(state: AgentState):
    try:
        # Validate the entire batch before asking target-selection questions.
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
            "reply": "Actions validated and targets resolved. Nothing executed.",
        }

    except RequestCancelled:
        return {
            "cancelled": True,
            "prepared_actions": [],
            "proposal": None,
            "history": [],
            "reply": "Cancelled. No tasks were changed.",
        }

    except (ValidationError, ValueError) as error:
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

    return "finish"