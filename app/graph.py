"""Interpret -> prepare frozen targets -> select/confirm -> atomic execution."""

import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.action_validation import validate_action
from app.executor import execute_changes
from app.llm import interpret_message
from app.repository import get_task
from app.task_resolver import (
    CANCEL_REPLIES,
    YES_REPLIES,
    normalize_reply,
    refresh_selected_task,
    resolve_targets,
    select_from_reply,
)

logger = logging.getLogger(__name__)
MUTATIONS = {"create_task", "update_task", "delete_task"}
MAX_PREPARED_ACTIONS = 100


class AgentState(TypedDict, total=False):
    message: str
    delivery_id: str
    history: list[dict]
    proposal: dict | None
    reply: str
    cancelled: bool
    prepared_actions: list[dict]
    request_id: str
    execution_result: dict | None
    execution_error: str | None
    index_warning: str | None
    memory: dict
    cursor: int
    candidates: list[dict]
    question: str
    mode: str
    next_node: str
    approved: bool
    force_confirmation: bool


def initial_state(message, request_id, memory=None):
    return dict(
        message=message,
        request_id=request_id,
        delivery_id=request_id,
        memory=memory or {},
        history=[],
        proposal=None,
        reply="",
        cancelled=False,
        prepared_actions=[],
        execution_result=None,
        execution_error=None,
        index_warning=None,
        cursor=0,
        candidates=[],
        question="",
        mode="",
        approved=False,
        force_confirmation=False,
    )


def interpret_node(state):
    memory = state.get("memory", {})
    ids = list(
        dict.fromkeys(memory.get("last_tasks", []) + memory.get("last_results", []))
    )
    references = [get_task(i) for i in ids]
    context = {
        "references": [
            {k: t[k] for k in ("id", "title", "status", "priority", "due_date")}
            for t in references[:100]
            if t
        ],
        "memory": memory,
        "pending_proposal": state.get("proposal"),
        "pending_question": state.get("question"),
        "instruction": "Use pending details for answers/corrections; discard them for replacements.",
    }
    proposal = interpret_message(
        state["message"], history=state.get("history", []), context=context
    )
    data = proposal.model_dump(mode="json")
    outcome = data["outcome"]
    return {
        "proposal": data,
        "prepared_actions": [],
        "cursor": 0,
        "candidates": [],
        "approved": False,
        "cancelled": False,
        "execution_result": None,
        "execution_error": None,
        "index_warning": None,
        "history": [
            *state.get("history", []),
            {"role": "human", "content": state["message"]},
        ],
        "question": data["message"],
        "mode": "clarify",
        "reply": data["message"],
        "next_node": (
            "prepare"
            if outcome == "propose"
            else "await_reply" if outcome == "clarify" else "finish"
        ),
    }


def bind(action, task):
    return {
        **action,
        "scope": "single",
        "reference": None,
        "task_id": task["id"],
        "target_query": None,
        "expected_version": task["version"],
        "resolved_title": task["title"],
        "before": {k: task[k] for k in action.get("values", {})},
    }


def describe(action):
    label = f"#{action.get('task_id')} — {action.get('resolved_title', action.get('target_query') or '')}"
    if action["operation"] == "update_task":
        changes = "; ".join(
            f"{k}: {json.dumps(action.get('before', {}).get(k), ensure_ascii=False)} → {json.dumps(v, ensure_ascii=False)}"
            for k, v in action["values"].items()
        )
        return f"Update {label}: {changes}"
    if action["operation"] == "delete_task":
        return f"DELETE {label}"
    return (
        action["operation"]
        + ": "
        + json.dumps(
            action.get("values") or action.get("filters") or action.get("target_query"),
            ensure_ascii=False,
        )
    )


def plan_text(actions):
    return "\n".join(f"{i}. {describe(a)}" for i, a in enumerate(actions, 1))


def clarification(error):
    return {
        "prepared_actions": [],
        "cursor": 0,
        "candidates": [],
        "approved": False,
        "force_confirmation": True,
        "mode": "clarify",
        "next_node": "await_reply",
        "question": f"I couldn't prepare this request: {error} Please correct it, or cancel.",
    }


def prepare_node(state):
    try:
        actions = [validate_action(a) for a in state["proposal"]["actions"]]
        if any(a["operation"] in MUTATIONS for a in actions) and any(
            a["operation"] in {"list_tasks", "search_tasks"} for a in actions
        ):
            raise ValueError(
                "Please search/list first, then request changes to those results."
            )
        prepared = list(state.get("prepared_actions", []))
        for index in range(state.get("cursor", 0), len(actions)):
            action = actions[index]
            if action["operation"] in {"update_task", "delete_task"}:
                targets, candidates = resolve_targets(action, state.get("memory", {}))
                if candidates:
                    options = "\n".join(
                        f"{i}. #{t['id']}: {t['title']} ({t['status']}, {t['priority']})"
                        for i, t in enumerate(candidates, 1)
                    )
                    one = len(candidates) == 1 and len(actions) == 1
                    question = (
                        (
                            "Confirm this change:\n"
                            + describe(bind(action, candidates[0]))
                            + "\nReply yes to execute, or cancel/correct it."
                        )
                        if one
                        else (
                            f"Select the target for action {index + 1}: {action['operation']} {json.dumps(action['values'])}\n{options}\nReply with a task ID/title or 'the second one'; cancel or describe a replacement request."
                        )
                    )
                    return {
                        "prepared_actions": prepared,
                        "cursor": index,
                        "candidates": candidates,
                        "mode": "select_execute" if one else "select",
                        "question": question,
                        "force_confirmation": True,
                        "next_node": "await_reply",
                    }
                if not targets:
                    raise ValueError(
                        "No tasks match this scope. Nothing will be changed."
                    )
                for task in targets:
                    prepared.append(bind(action, task))
            else:
                prepared.append(action)
            if len(prepared) > MAX_PREPARED_ACTIONS:
                raise ValueError(
                    "This request affects more than 100 actions. Narrow the filters; no subset was executed."
                )
        if len(prepared) > MAX_PREPARED_ACTIONS:
            raise ValueError(
                "More than 100 actions. Narrow the scope; nothing was executed."
            )
        ids = [
            a["task_id"]
            for a in prepared
            if a["operation"] in {"update_task", "delete_task"}
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "The same task appears in multiple changes. Combine its changes into one action."
            )
        unchanged = [
            a
            for a in prepared
            if a["operation"] == "update_task" and a.get("before") == a["values"]
        ]
        prepared = [a for a in prepared if a not in unchanged]
        if not prepared:
            memory = {
                **state.get("memory", {}),
                "last_tasks": ids,
                "focus": "last_tasks",
            }
            return {
                "prepared_actions": [],
                "memory": memory,
                "reply": "Matching tasks already have the requested values. No changes needed.",
                "next_node": "finish",
            }
        mutations = [a for a in prepared if a["operation"] in MUTATIONS]
        needs = bool(mutations) and (
            state.get("force_confirmation")
            or len(mutations) > 1
            or any(
                a["operation"] == "delete_task" or a.get("scope", "single") != "single"
                for a in actions
            )
        )
        if needs:
            return {
                "prepared_actions": prepared,
                "cursor": len(actions),
                "mode": "confirm",
                "candidates": [],
                "question": f"Confirm this plan ({len(mutations)} change(s)); nothing has executed:\n{plan_text(prepared)}\nReply yes to apply exactly this plan, no to cancel, or describe a correction.",
                "next_node": "await_reply",
            }
        return {"prepared_actions": prepared, "next_node": "execute"}
    except ValueError as error:
        return clarification(str(error))


def await_reply_node(state):
    delivery = interrupt({"kind": state["mode"], "question": state["question"]})
    answer = delivery.get("message") if isinstance(delivery, dict) else delivery
    update = handle_reply(state, answer)
    if isinstance(delivery, dict):
        update["delivery_id"] = delivery["message_id"]
    return update


def handle_reply(state, answer):
    if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 2000:
        return {
            "question": "Enter 1–2000 characters.\n" + state["question"],
            "next_node": "await_reply",
        }
    answer = answer.strip()
    reply = normalize_reply(answer)
    if reply in CANCEL_REPLIES:
        return {
            "cancelled": True,
            "reply": "Cancelled. No tasks were changed.",
            "next_node": "finish",
        }
    if state["mode"] == "confirm" and reply in YES_REPLIES:
        return {"approved": True, "next_node": "execute"}
    if state["mode"] in {"select", "select_execute"}:
        try:
            selected = select_from_reply(answer, state["candidates"])
        except ValueError as error:
            return {
                "question": str(error) + "\n" + state["question"],
                "next_node": "await_reply",
            }
        try:
            if selected:
                task = refresh_selected_task(selected)
                action = validate_action(state["proposal"]["actions"][state["cursor"]])
                prepared = [*state["prepared_actions"], bind(action, task)]
                # The one-candidate question already showed the exact operation.
                if state["mode"] == "select_execute":
                    return {
                        "prepared_actions": prepared,
                        "approved": True,
                        "next_node": "execute",
                    }
                return {
                    "prepared_actions": prepared,
                    "cursor": state["cursor"] + 1,
                    "candidates": [],
                    "next_node": "prepare",
                }
        except ValueError as error:
            return clarification(str(error))
    # Anything other than exact approval is interpreted as an answer/correction/replacement.
    return {
        "message": answer,
        "approved": False,
        "prepared_actions": [],
        "candidates": [],
        "cursor": 0,
        "force_confirmation": True,
        "history": [
            *state.get("history", []),
            {"role": "assistant", "content": state["question"]},
        ],
        "next_node": "interpret",
    }


def execute_node(state):
    try:
        # Defense in depth: all deletes and multi-action mutations require approval.
        mutations = [
            a for a in state["prepared_actions"] if a["operation"] in MUTATIONS
        ]
        if not state.get("approved") and (
            len(mutations) > 1
            or any(a["operation"] == "delete_task" for a in mutations)
        ):
            raise ValueError("This plan requires confirmation.")
        result = execute_changes(state["request_id"], state["prepared_actions"])
        return {"execution_result": result, "execution_error": None}
    except ValueError as error:
        return {"execution_result": None, "execution_error": str(error)}


def synchronize_node(state):
    if not state.get("execution_result") or not any(
        a["operation"] in MUTATIONS for a in state["prepared_actions"]
    ):
        return {}
    try:
        from app.search_service import synchronize_index

        synchronize_index()
        return {"index_warning": None}
    except Exception:
        logger.exception("Index synchronization failed after committed changes")
        return {
            "index_warning": "Changes were saved, but search synchronization failed. Run scripts.sync_search_index before searching again."
        }


def finish_node(state):
    memory = dict(state.get("memory", {}))
    result = state.get("execution_result")
    reply = state.get("reply", "No action was executed.")
    if state.get("cancelled"):
        reply = "Cancelled. No tasks were changed."
    elif state.get("execution_error"):
        reply = "No changes committed for this request: " + state["execution_error"]
    elif result:
        reply = "\n".join(r["message"] for r in result["results"])
        last_tasks = []
        for item in result["results"]:
            if "task" in item:
                last_tasks.append(item["task"]["id"])
            if item["operation"] == "delete_task":
                deleted = item["task_id"]
                for key in ("last_tasks", "last_results"):
                    memory[key] = [i for i in memory.get(key, []) if i != deleted]
            if "tasks" in item:
                memory["last_results"] = [t["id"] for t in item["tasks"]]
                memory["focus"] = "last_results"
            if "matches" in item:
                memory["last_results"] = [m["task"]["id"] for m in item["matches"]]
                memory["focus"] = "last_results"
        if last_tasks:
            memory["last_tasks"] = list(dict.fromkeys(last_tasks))
            memory["focus"] = "last_tasks"
        memory["last_summary"] = reply[:1500]
        if state.get("index_warning"):
            reply += "\n" + state["index_warning"]
    return {
        "reply": reply,
        "memory": memory,
        "history": [],
        "proposal": None,
        "prepared_actions": [],
        "candidates": [],
        "question": "",
        "cursor": 0,
        "approved": False,
    }


def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    nodes = {
        "interpret": interpret_node,
        "prepare": prepare_node,
        "await_reply": await_reply_node,
        "execute": execute_node,
        "synchronize": synchronize_node,
        "finish": finish_node,
    }
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "interpret")
    for name in ("interpret", "prepare", "await_reply"):
        builder.add_conditional_edges(
            name, lambda state: state["next_node"], {n: n for n in nodes}
        )
    builder.add_edge("execute", "synchronize")
    builder.add_edge("synchronize", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
