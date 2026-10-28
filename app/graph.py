"""Interpret -> prepare frozen targets -> select/confirm -> atomic execution."""

import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.action_validation import validate_action
from app.executor import execute_changes
from app.llm import interpret_message
from app.repository import get_task, list_tasks
from app.conversation import (
    GROUP_REPLIES,
    REFERENCE_WORDS,
    calendar_context,
    clarified_values,
    proposal_for,
    simple_change,
    simple_request,
    starts_new_request,
    exceeds_action_limit,
    validate_id_sources,
    refers_to_pending_group,
)
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
    pending_targets: list[dict]
    group_confirmation: bool
    target_authority: str
    preparation_failures: int


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
        pending_targets=[],
        group_confirmation=False,
        target_authority="",
        preparation_failures=0,
    )


def interpret_node(state):
    message = state["message"]
    if exceeds_action_limit(message):
        return {
            "proposal": None,
            "prepared_actions": [],
            "execution_result": None,
            "execution_error": None,
            "reply": "This request contains more than five actions. Nothing was changed. Please complete the existing task in one request, then create up to five tasks in another request.",
            "next_node": "finish",
        }
    memory = state.get("memory", {})
    pending_targets = state.get("pending_targets", [])
    direct = simple_request(message)
    data = proposal_for(direct) if direct else None
    # Completed-task IDs are useful only for references. Never present them as
    # possible replacements for a newly named target.
    use_memory = bool(REFERENCE_WORDS.search(message)) and not pending_targets
    visible_memory = memory if use_memory else {}
    ids = visible_memory.get(visible_memory.get("focus", ""), [])
    references = [get_task(i) for i in ids[:100]]
    context = {
        "references": [t for t in references if t],
        "memory": visible_memory,
        "pending_targets": pending_targets,
        "pending_proposal": state.get("proposal"),
        "pending_question": state.get("question"),
        "calendar": calendar_context(),
        "instruction": "Resolve names through Python; pending candidates override old memory. Never claim a task does not exist.",
    }
    if data is None:
        proposal = interpret_message(
            message, history=state.get("history", []), context=context
        )
        data = proposal.model_dump(mode="json")
    if len(data.get("actions", [])) > 5:
        return {
            "proposal": None,
            "prepared_actions": [],
            "execution_result": None,
            "execution_error": None,
            "reply": "This request contains more than five actions. Nothing was changed. Please split it into requests of at most five actions (for example, complete the task first, then create the five meeting tasks).",
            "next_node": "finish",
        }
    outcome = data["outcome"]
    # If the only uncertainty is target identity, let the real resolver check it.
    if (
        outcome == "clarify"
        and data.get("clarification_field") == "target"
        and data.get("actions")
    ):
        try:
            for action in data["actions"]:
                validate_action(action)
        except ValueError:
            pass
        else:
            data["outcome"] = outcome = "propose"
    if outcome == "propose" and not direct:
        try:
            if pending_targets:
                for action in data["actions"]:
                    if action["operation"] in {
                        "update_task",
                        "delete_task",
                    } and refers_to_pending_group(message):
                        action.update(
                            scope="references",
                            reference="pending_targets",
                            task_id=None,
                        )
                    elif action.get("scope") == "references":
                        action["reference"] = "pending_targets"
            # A confirmed explicit title correction must not be replaced by an old ID.
            authority = state.get("target_authority", "")
            if authority and len(data["actions"]) == 1:
                action = data["actions"][0]
                if (
                    action["operation"] in {"update_task", "delete_task"}
                    and action.get("scope", "single") == "single"
                ):
                    action.update(task_id=None, target_query=authority, reference=None)
            validate_id_sources(
                data["actions"],
                message,
                state.get("history", []),
                visible_memory,
                pending_targets,
            )
        except ValueError as error:
            data = {
                "outcome": "clarify",
                "actions": [],
                "message": str(error),
                "clarification_field": "target",
            }
            outcome = "clarify"
    field = data.get("clarification_field")
    question = data["message"]
    if outcome == "clarify":
        # These questions come from the backend, not invented database/calendar facts.
        questions = {
            "target": "Which task do you mean? Give its title, description or ID.",
            "title": "What should the task be called?",
            "changes": "What would you like to change?",
            "due_date": "What due date do you mean? Use YYYY-MM-DD, or say 'no due date'.",
            "priority": "Which priority: low, medium or high?",
            "status": "Which status: To do, In progress or Done?",
        }
        question = questions.get(field, question)
        if any(
            term in question.casefold()
            for term in (
                "couldn't find",
                "could not find",
                "does not exist",
                "no task",
                "did you mean task",
            )
        ):
            question = questions["target"]
        if "weekend" in message.casefold():
            question = questions["due_date"]
    return {
        "proposal": data,
        "prepared_actions": [],
        "cursor": 0,
        "candidates": [],
        "approved": False,
        "cancelled": False,
        "force_confirmation": False,
        "execution_result": None,
        "execution_error": None,
        "index_warning": None,
        "history": [*state.get("history", []), {"role": "human", "content": message}],
        "question": question,
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
        "target_snapshot": dict(task),
        "before": {k: task[k] for k in action.get("values", {})},
    }


def display_value(value):
    if value is None:
        return "none"
    return {"to_do": "To do", "in_progress": "In progress", "done": "Done"}.get(
        str(value), str(value)
    )


def describe(action):
    label = f"#{action.get('task_id')} — {action.get('resolved_title', action.get('target_query') or '')}"
    if action["operation"] == "update_task":
        changes = "; ".join(
            f"{k.replace('_', ' ')}: {display_value(action.get('before', {}).get(k))} → {display_value(v)}"
            for k, v in action["values"].items()
        )
        return f"Update {label}: {changes}"
    if action["operation"] == "delete_task":
        return f"Delete {label}"
    if action["operation"] == "create_task":
        values = action["values"]
        details = f"Create {values['title']} | {display_value(values.get('status', 'to_do'))} | {values.get('priority', 'medium')} priority | Due: {values.get('due_date') or 'none'}"
        if values.get("description"):
            details += f" | Description: {values['description']}"
        return details
    return "Search tasks" if action["operation"] == "search_tasks" else "List tasks"


def plan_text(actions):
    return "\n".join(f"{i}. {describe(a)}" for i, a in enumerate(actions, 1))


def clarification(error):
    return {
        "prepared_actions": [],
        "cursor": 0,
        "candidates": [],
        "approved": False,
        "force_confirmation": False,
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
                references = {
                    **state.get("memory", {}),
                    "pending_targets": [
                        t["id"] for t in state.get("pending_targets", [])
                    ],
                }
                if action.get("reference") == "pending_targets":
                    for task in state.get("pending_targets", []):
                        refresh_selected_task(task)
                targets, candidates = resolve_targets(action, references)
                if candidates:
                    options = "\n".join(
                        f"{i}. #{t['id']}: {t['title']} ({display_value(t['status'])}, {t['priority']}) — {t.get('description') or 'No description'}"
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
                            f"Which task should I {'delete' if action['operation'] == 'delete_task' else 'update: ' + '; '.join(k.replace('_', ' ') + ' → ' + display_value(v) for k, v in action['values'].items())}?\n{options}\nChoose a task ID, description or position. You can also say 'both' or 'all of these', cancel, or give a replacement request."
                        )
                    )
                    return {
                        "prepared_actions": prepared,
                        "cursor": index,
                        "candidates": candidates,
                        "pending_targets": candidates,
                        "mode": "select_execute" if one else "select",
                        "question": question,
                        "force_confirmation": False,
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
            state.get("group_confirmation")
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
                "pending_targets": [
                    a["target_snapshot"] for a in prepared if a.get("task_id")
                ],
                "question": f"Confirm this plan ({len(mutations)} change(s)); nothing has executed:\n{plan_text(prepared)}\nReply yes to apply exactly this plan, no to cancel, or describe a correction.",
                "next_node": "await_reply",
            }
        return {"prepared_actions": prepared, "next_node": "execute"}
    except ValueError as error:
        failures = state.get("preparation_failures", 0) + 1
        if failures >= 2:
            return {
                "proposal": None,
                "prepared_actions": [],
                "execution_result": None,
                "execution_error": None,
                "preparation_failures": 0,
                "reply": "I couldn't prepare a valid action after your correction. No changes were made. I've cleared the pending request; please send the complete instruction again. Details: "
                + str(error),
                "next_node": "finish",
            }
        return {**clarification(str(error)), "preparation_failures": failures}


def await_reply_node(state):
    delivery = interrupt({"kind": state["mode"], "question": state["question"]})
    answer = delivery.get("message") if isinstance(delivery, dict) else delivery
    update = handle_reply(state, answer)
    if isinstance(delivery, dict):
        update["delivery_id"] = delivery["message_id"]
    return update


def replan(state, actions, **extra):
    return {
        "proposal": {
            "outcome": "propose",
            "actions": actions,
            "message": "Preparing the requested change.",
        },
        "prepared_actions": [],
        "cursor": 0,
        "candidates": [],
        "approved": False,
        "force_confirmation": False,
        "group_confirmation": False,
        "next_node": "prepare",
        **extra,
    }


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
    # Clear named instructions are replacements, not candidate answers.
    direct = simple_request(answer)
    if direct:
        return replan(
            state,
            [direct],
            history=[],
            pending_targets=[],
            target_authority="",
            preparation_failures=0,
        )
    if starts_new_request(answer):
        return {
            "message": answer,
            "history": [],
            "proposal": None,
            "prepared_actions": [],
            "candidates": [],
            "pending_targets": [],
            "question": "",
            "cursor": 0,
            "approved": False,
            "group_confirmation": False,
            "target_authority": "",
            "preparation_failures": 0,
            "next_node": "interpret",
        }
    if state["mode"] in {"select", "select_execute"}:
        candidates = state["candidates"]
        try:
            if reply in GROUP_REPLIES:
                if reply.startswith("both") and len(candidates) != 2:
                    raise ValueError(
                        "Which two tasks? There are not exactly two offered candidates."
                    )
                selected = candidates
            else:
                item = select_from_reply(answer, candidates)
                selected = [item] if item else []
        except ValueError as error:
            return {
                "question": str(error) + "\n" + state["question"],
                "next_node": "await_reply",
            }
        try:
            if selected:
                action = validate_action(state["proposal"]["actions"][state["cursor"]])
                prepared = [
                    *state["prepared_actions"],
                    *[bind(action, refresh_selected_task(t)) for t in selected],
                ]
                if state["mode"] == "select_execute" and len(selected) == 1:
                    return {
                        "prepared_actions": prepared,
                        "approved": True,
                        "next_node": "execute",
                    }
                return {
                    "prepared_actions": prepared,
                    "cursor": state["cursor"] + 1,
                    "candidates": [],
                    "group_confirmation": len(selected) > 1,
                    "next_node": "prepare",
                }
        except ValueError as error:
            return clarification(str(error))
    if state["mode"] in {"select", "select_execute"}:
        actions = state["proposal"]["actions"]
        index = state["cursor"]
        corrected = clarified_values(answer, actions[index])
        if corrected:
            actions = [*actions[:index], corrected, *actions[index + 1 :]]
            return {
                "proposal": {**state["proposal"], "actions": actions},
                "approved": False,
                "candidates": [],
                "next_node": "prepare",
            }
    if state["mode"] == "confirm" and state.get("pending_targets"):
        # Ordinals refer to THIS displayed plan, never an earlier candidate list.
        try:
            selected = select_from_reply(answer, state["pending_targets"])
            if selected:
                refresh_selected_task(selected)
                prepared = [
                    a
                    for a in state["prepared_actions"]
                    if a.get("task_id") == selected["id"]
                ]
                return {
                    "prepared_actions": prepared,
                    "cursor": len(state["proposal"]["actions"]),
                    "group_confirmation": False,
                    "approved": False,
                    "proposal": {
                        "outcome": "propose",
                        "actions": [
                            {
                                k: v
                                for k, v in a.items()
                                if k
                                in {
                                    "operation",
                                    "task_id",
                                    "target_query",
                                    "scope",
                                    "reference",
                                    "values",
                                    "filters",
                                }
                            }
                            for a in prepared
                        ],
                        "message": "Selected one change.",
                    },
                    "next_node": "prepare",
                }
        except ValueError as error:
            return {
                "question": str(error) + "\n" + state["question"],
                "next_node": "await_reply",
            }
    pending = (state.get("proposal") or {}).get("actions", [])
    if state["mode"] == "clarify" and len(pending) == 1:
        corrected = clarified_values(answer, pending[0])
        if corrected:
            return replan(state, [corrected])
        # A title or ID answer completes a pending target without losing its changes.
        task = next(
            (t for t in list_tasks() if normalize_reply(t["title"]) == reply), None
        )
        if task and pending[0]["operation"] in {"update_task", "delete_task"}:
            action = {
                **pending[0],
                "task_id": None,
                "target_query": answer,
                "scope": "single",
                "reference": None,
            }
            return replan(state, [action], target_authority=answer)
    if reply in YES_REPLIES or reply in GROUP_REPLIES:
        # A model clarification is not an execution preview. A bare yes cannot
        # turn a hallucinated suggestion into permission to change an old task.
        return {
            "question": "No executable change is awaiting approval. Give the task title and requested change, or cancel.",
            "next_node": "await_reply",
        }
    # Preserve the exact displayed set for interpreted corrections. Do not mix it
    # with completed-task memory; the model can reference pending_targets only.
    title_match = next(
        (t for t in list_tasks() if normalize_reply(t["title"]) == reply), None
    )
    return {
        "message": answer,
        "approved": False,
        "prepared_actions": [],
        "candidates": [],
        "cursor": 0,
        "force_confirmation": False,
        "group_confirmation": False,
        "target_authority": answer if title_match else "",
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
        "pending_targets": [],
        "group_confirmation": False,
        "target_authority": "",
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
