from fastapi import FastAPI

from app.schemas import TaskCreate, TaskUpdate


app = FastAPI(
    title="TaskFlow API",
    description="AI-powered task manager",
    version="0.1.0",
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/validate/task-create")
def validate_task_create(payload: TaskCreate):
    return {
        "valid": True,
        "task": payload.model_dump(mode="json"),
    }


@app.post("/api/validate/task-update")
def validate_task_update(payload: TaskUpdate):
    return {
        "valid": True,
        "changes": payload.model_dump(
            mode="json",
            exclude_unset=True,
        ),
    }