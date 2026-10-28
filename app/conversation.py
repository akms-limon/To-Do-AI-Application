"""Small deterministic shortcuts. Unrecognized language still goes to Groq.

These shortcuts never guess a task ID and never approve a changed plan.
"""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.task_resolver import normalize_reply, normalize_title

GROUP_REPLIES = {
    "both",
    "both of them",
    "both tasks",
    "all of them",
    "all these",
    "all these tasks",
    "all of these",
    "all of these tasks",
    "sob gula",
    "sobgula",
    "duita",
    "দুটো",
    "দুটাই",
    "সবগুলো",
}
REFERENCE_WORDS = re.compile(
    r"\b(it|its|them|their|those|these|both|same|one|ones)\b", re.I
)
PRIORITIES = r"low|medium|high"
STATUSES = {
    "to do": "to_do",
    "to_do": "to_do",
    "in progress": "in_progress",
    "in_progress": "in_progress",
    "done": "done",
    "completed": "done",
    "complete": "done",
}


def calendar_context():
    today = datetime.now(ZoneInfo("Asia/Dhaka")).date()
    return [
        {
            "date": (today + timedelta(days=i)).isoformat(),
            "weekday": (today + timedelta(days=i)).strftime("%A"),
        }
        for i in range(15)
    ]


def simple_change(message):
    """Only full, unambiguous single-change English forms; otherwise return None."""
    text = message.strip().rstrip(".!?").strip()
    text = re.sub(r"^please\s+", "", text, flags=re.I)
    delete_match = re.fullmatch(
        r"(?:delete|remove) (?:the )?task(?: has)? (?:id\s+|#)(\d+)",
        text,
        flags=re.I,
    ) or re.fullmatch(r"(?:delete|remove) (?:the )?task\s+#?(\d+)", text, flags=re.I)
    if delete_match:
        return {
            "operation": "delete_task",
            "task_id": int(delete_match[1]),
            "target_query": None,
            "scope": "single",
            "reference": None,
            "values": {},
            "filters": {},
        }
    patterns = [
        (
            rf"(?:change|set|update) (?:the )?priority of (.+?) to ({PRIORITIES})",
            "priority",
        ),
        (rf"(?:change|set|make) (.+?) (?:to )?({PRIORITIES}) priority", "priority"),
        (rf"(?:change|set|update) (.+?)(?:'s)? priority to ({PRIORITIES})", "priority"),
        (
            r"(?:move|change|set|reopen) (.+?) (?:to|as) (to do|to_do|in progress|in_progress|done)",
            "status",
        ),
        (r"mark (.+?) (?:as )?(done|completed|complete)", "status"),
    ]
    for pattern, field in patterns:
        match = re.fullmatch(pattern, text, flags=re.I)
        if not match:
            continue
        target, value = match.groups()
        target = target.strip().strip('"')
        # Do not interpret batches, references or groups as literal task titles.
        if (
            re.search(
                r"\b(and|or|all|both|every|these|those|them|it|its|their)\b",
                target,
                re.I,
            )
            or "," in target
            or ";" in target
            or "\n" in target
        ):
            return None
        normalized = normalize_title(target)
        if normalized in {"task", "tasks", "the task", "my task"}:
            return None
        target_id = re.fullmatch(r"(?:task\s*)?#(\d+)|task\s+(\d+)", target, re.I)
        # Restrict this shortcut to concrete names/IDs; leave property descriptions to the model.
        if not target_id and re.search(
            r"\b(with|whose|priority|status|deadline|due)\b", target, re.I
        ):
            return None
        value = value.casefold()
        if field == "status":
            value = STATUSES[value]
        return {
            "operation": "update_task",
            "task_id": (
                int(next(g for g in target_id.groups() if g)) if target_id else None
            ),
            "target_query": None if target_id else target,
            "scope": "single",
            "reference": None,
            "values": {field: value},
            "filters": {},
        }
    return None


def proposal_for(action):
    return {
        "outcome": "propose",
        "actions": [action],
        "message": "Preparing the requested change.",
        "clarification_field": None,
    }


def target_evidence(message, history):
    """Only user text, never model-generated questions or previous completed replies."""
    return "\n".join(
        [item["content"] for item in history if item["role"] == "human"] + [message]
    )


def explicit_ids(text):
    return {
        int(m.group(1))
        for m in re.finditer(r"(?:#|\btask\s+|\bid\s+)(\d+)\b", text, re.I)
    }


def validate_id_sources(actions, message, history, memory, pending_targets):
    """An old context ID is not permission to target it in a new named request."""
    # A new imperative cannot borrow an explicit ID from an older instruction.
    replacement = re.search(
        r"\b(change|update|delete|remove|move|mark|set|make|create|koro|korun|dao)\b",
        message,
        re.I,
    )
    evidence = message if replacement else target_evidence(message, history)
    allowed = explicit_ids(evidence)
    if REFERENCE_WORDS.search(message):
        if pending_targets:
            allowed.update(t["id"] for t in pending_targets)
        else:
            allowed.update(memory.get(memory.get("focus", ""), []))
    # Bare numeric replies are allowed only for visible/pending targets, handled locally.
    for action in actions:
        task_id = action.get("task_id")
        if task_id is not None and task_id not in allowed:
            raise ValueError(
                "Please identify the target by title or an explicit task ID. I won't use an unrelated task from earlier context."
            )


def clarified_values(answer, action):
    """Apply short field-only answers without asking the model to repeat the target."""
    text = normalize_reply(answer)
    values = dict(action.get("values", {}))
    if text in {
        "no due date",
        "no deu date",
        "without a due date",
        "no deadline",
        "remove the deadline",
    }:
        values["due_date"] = None
    else:
        match = re.fullmatch(
            r"(?:no[, ]+|actually[, ]+)?(?:make (?:it |its priority )?|set (?:it |its priority )?(?:to )?)?(low|medium|high)(?: priority)?(?: instead)?",
            text,
        )
        status = re.fullmatch(
            r"(?:move (?:it )?(?:to )?|make (?:it )?|mark (?:it )?(?:as )?)?(to do|in progress|done|completed)",
            text,
        )
        if match and action["operation"] in {"create_task", "update_task"}:
            values["priority"] = match.group(1)
        elif status and action["operation"] in {"create_task", "update_task"}:
            values["status"] = STATUSES[status.group(1)]
        else:
            return None
    if action["operation"] not in {"create_task", "update_task"}:
        return None
    return {**action, "values": values}


def refers_to_pending_group(message):
    return bool(
        re.search(r"\b(both|them|their|those|these)\b", message, re.I)
    ) and not re.search(r"\b(?:all|every)\s+(?:my\s+)?tasks?\b", message, re.I)


def simple_request(message):
    """Exact read commands and plain named creations; never parse a partial batch."""
    text = normalize_reply(message)
    if text in {
        "show me all tasks",
        "show all tasks",
        "list all tasks",
        "list tasks",
        "show my tasks",
        "show all my tasks",
        "show me all my tasks",
        "sob task dekhao",
        "sob task dekhao amar",
        "সব টাস্ক দেখাও",
    }:
        return {"operation": "list_tasks", "filters": {}}
    if text in {
        "clear all tasks",
        "delete all tasks",
        "remove all tasks",
        "clear all my tasks",
        "delete all my tasks",
        "remove all my tasks",
    }:
        return {
            "operation": "delete_task",
            "task_id": None,
            "target_query": None,
            "scope": "all",
            "reference": None,
            "values": {},
            "filters": {},
        }
    match = re.fullmatch(
        r"(?:please\s+)?create\s+(?:a\s+)?task\s+(?:named|nammed|called)\s*:?\s*(.+)",
        message.strip(),
        re.I,
    )
    if not match:
        match = re.fullmatch(
            r"(?:please\s+)?create\s+(?:a\s+)?task\s+(.+)",
            message.strip(),
            re.I,
        )
    if match:
        title = match[1].strip().rstrip(".").strip()
        # Additional clauses/fields need full interpretation, not title-only creation.
        if title and not re.search(
            r"\b(and|then|with|priority|status|due|tomorrow|today|deadline|description|instead)\b|[;\n]",
            title,
            re.I,
        ):
            return {"operation": "create_task", "values": {"title": title.strip('"')}}
    return simple_change(message)


def starts_new_request(message):
    # A creation imperative is a new request, not an answer selecting an old task.
    return bool(re.match(r"^(?:please\s+)?create\b", message.strip(), re.I))


def exceeds_action_limit(message):
    """Conservative preflight for explicit counted creations, not arbitrary prose."""
    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\bcreate\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+[^.;\n]*?\btasks?\b",
        message,
        re.I,
    )
    if not match:
        return False
    word = match[1].casefold()
    count = int(word) if word.isdigit() else numbers[word]
    if count > 5:
        return True
    prefix = message[: match.start()]
    return (
        count == 5
        and bool(
            re.search(r"\b(complete|update|delete|change|mark|move)\b", prefix, re.I)
        )
        and bool(re.search(r"\b(and|then)\b", prefix, re.I))
    )
