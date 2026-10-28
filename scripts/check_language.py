"""Optional LIVE Groq interpretation checks; never executes task changes.
Consumes API quota. Run: uv run python -m scripts.check_language
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from app.llm import interpret_message
from app.action_validation import validate_action


def main():
    tomorrow = (
        datetime.now(ZoneInfo("Asia/Dhaka")).date() + timedelta(days=1)
    ).isoformat()
    context = {
        "memory": {"last_tasks": [7], "last_results": [7, 8], "focus": "last_results"},
        "references": [
            {"id": 7, "title": "Play football", "status": "to_do", "priority": "low"},
            {
                "id": 8,
                "title": "Play badminton in Gulshan",
                "status": "to_do",
                "priority": "medium",
            },
        ],
    }
    cases = [
        (
            "Create a high-priority task to finish the demo tomorrow",
            lambda a: a["operation"] == "create_task"
            and a["values"].get("priority") == "high"
            and a["values"].get("due_date") == tomorrow,
        ),
        (
            "Make all tasks high priority",
            lambda a: a["operation"] == "update_task"
            and a["scope"] == "all"
            and a["values"] == {"priority": "high"},
        ),
        (
            "Delete all completed tasks",
            lambda a: a["operation"] == "delete_task"
            and a["scope"] == "matching"
            and a["filters"] == {"status": "done"},
        ),
        (
            "Make all of them high priority",
            lambda a: a["scope"] == "references"
            and a["reference"] == "last_results"
            and a["values"] == {"priority": "high"},
        ),
        (
            "Show unfinished tasks related to Gulshan",
            lambda a: a["operation"] == "search_tasks"
            and a["filters"] == {"status": "unfinished"}
            and "gulshan" in a["target_query"].lower(),
        ),
        (
            "sob task er priority low koro",
            lambda a: a["operation"] == "update_task"
            and a["scope"] == "all"
            and a["values"] == {"priority": "low"},
        ),
    ]
    failures = 0
    for text, check in cases:
        try:
            result = interpret_message(text, context=context)
            actions = [
                validate_action(a.model_dump(mode="json")) for a in result.actions
            ]
            passed = (
                result.outcome == "propose" and len(actions) == 1 and check(actions[0])
            )
            print(("PASS" if passed else "REVIEW"), text)
            if not passed:
                failures += 1
                print(result.model_dump_json(indent=2))
        except Exception as error:
            failures += 1
            print("ERROR", text, type(error).__name__, str(error))
    print(
        f"{len(cases)-failures}/{len(cases)} language checks passed. No task data was changed."
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
