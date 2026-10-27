from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.llm import interpret_message


class AgentState(TypedDict):
    message: str
    history: list[dict[str, str]]
    proposal: dict | None
    reply: str
    cancelled: bool


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
    if state["proposal"]["outcome"] == "clarify":
        return "clarify"

    return "finish"


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

    proposal = state["proposal"]

    if proposal["outcome"] == "unsupported":
        return {"reply": proposal["message"]}

    return {
        "reply": (
            "Proposal prepared. Task lookup and execution "
            "are not connected yet."
        )
    }


def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("interpret", interpret_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "interpret")

    builder.add_conditional_edges(
        "interpret",
        route_interpretation,
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