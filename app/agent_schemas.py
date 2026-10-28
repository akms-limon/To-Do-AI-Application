from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "create_task",
        "list_tasks",
        "search_tasks",
        "update_task",
        "delete_task",
    ]

    task_id: int | None = Field(
        default=None,
        gt=0,
        description="Only an ID explicitly provided by the user or present in supplied reference context.",
    )

    target_query: str | None = Field(
        default=None,
        description="Task description to resolve, or semantic search query.",
    )

    scope: Literal["single", "all", "matching", "references"] = "single"
    reference: Literal["last_tasks", "last_results"] | None = None

    values: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "Fields for creation or update. Allowed keys: title, description, "
            "status, priority, due_date. Omit unchanged fields. "
            "Use null only to explicitly clear description or due_date."
        ),
    )

    filters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Current-property target or listing/search filters. Allowed keys: status, priority, due_date. "
            "Status can also be unfinished."
        ),
    )

    @model_validator(mode="after")
    def check_field_names(self):
        allowed_values = {"title", "description", "status", "priority", "due_date"}
        allowed_filters = {"status", "priority", "due_date"}

        if not set(self.values).issubset(allowed_values):
            raise ValueError("Unsupported task field.")

        if not set(self.filters).issubset(allowed_filters):
            raise ValueError("Unsupported filter.")

        return self


class AgentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["propose", "clarify", "unsupported"]

    actions: list[ActionProposal] = Field(
        default_factory=list,
        max_length=5,
    )

    message: str = Field(
        min_length=1,
        max_length=1000,
        description="Clarification question or brief explanation; never claim execution.",
    )

    @model_validator(mode="after")
    def check_outcome(self):
        if self.outcome == "propose" and not self.actions:
            raise ValueError("A proposal must contain an action.")

        if self.outcome == "unsupported" and self.actions:
            raise ValueError("Unsupported requests cannot contain actions.")

        return self
