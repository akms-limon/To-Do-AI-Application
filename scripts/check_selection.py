from unittest.mock import patch

from app.task_resolver import RequestCancelled, choose_candidate


def run_case(label, candidates, replies, expected_id=None, cancel=False):
    with (
        patch(
            "app.task_resolver.interrupt",
            side_effect=replies,
        ),
        patch(
            "app.task_resolver.refresh_selected_task",
            side_effect=lambda task: task,
        ),
    ):
        try:
            selected = choose_candidate(candidates)
        except RequestCancelled:
            assert cancel, f"{label}: unexpected cancellation"
        else:
            assert not cancel, f"{label}: expected cancellation"
            assert selected["id"] == expected_id, label

    print(f"PASS: {label}")


def main():
    first = {"id": 7, "title": "Play football", "status": "done"}
    second = {"id": 9, "title": "Watch football", "status": "to_do"}

    run_case("Single candidate accepts yes", [first], ["yes"], 7)
    run_case("Single candidate accepts ha", [first], ["ha"], 7)
    run_case("No cancels selection", [first], ["no"], cancel=True)

    run_case(
        "Multiple candidates require more than yes",
        [first, second],
        ["yes", "the second one"],
        9,
    )

    run_case(
        "Selection by title",
        [first, second],
        ["Play football"],
        7,
    )

    run_case(
        "Selection by task ID",
        [first, second],
        ["#9"],
        9,
    )


if __name__ == "__main__":
    main()