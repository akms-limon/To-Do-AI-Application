import re
from datetime import date, datetime
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

TaskStatus = Literal["to_do", "in_progress", "done"]
TaskPriority = Literal["low", "medium", "high"]

Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]

Description = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=2000),
]


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("due_date", mode="before", check_fields=False)
    @classmethod
    def parse_due_date(cls, value):
        if value is None:
            return None

        # Internal Python calls may supply a date object.
        if type(value) is date:
            return value

        # API input must use an unambiguous calendar date.
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value
        ):
            raise ValueError("Use a date in YYYY-MM-DD format.")

        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError("The due date is not a valid calendar date.")

    @field_validator("due_date", check_fields=False)
    @classmethod
    def reject_past_due_date(cls, value):
        today = datetime.now(ZoneInfo("Asia/Dhaka")).date()

        if value is not None and value < today:
            raise ValueError("The due date cannot be before today.")

        return value


class TaskCreate(TaskInput):
    title: Title
    description: Description | None = None
    status: TaskStatus = "to_do"
    priority: TaskPriority = "medium"
    due_date: date | None = None


class TaskUpdate(TaskInput):
    title: Title | None = None
    description: Description | None = None
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    due_date: date | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")

        for name in ("title", "status", "priority"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null.")

        return self