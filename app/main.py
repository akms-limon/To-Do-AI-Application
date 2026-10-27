from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import repository
from app.chat_service import (
    ChatRequest,
    ChatServiceError,
    handle_chat,
)
from app.database import initialize_database
from app.schemas import TaskCreate, TaskUpdate


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="TaskFlow API",
    description="AI-powered task manager",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIRECTORY = FilePath(__file__).resolve().parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIRECTORY)),
    name="static",
)

TaskId = Annotated[int, Path(gt=0)]


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    try:
        return handle_chat(payload)
    except ChatServiceError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


@app.post("/api/tasks", status_code=201)
def create_task(payload: TaskCreate):
    return repository.create_task(payload)


@app.get("/api/tasks")
def list_tasks():
    return repository.list_tasks()


@app.get("/api/tasks/{task_id}")
def get_task(task_id: TaskId):
    task = repository.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return task


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: TaskId, payload: TaskUpdate):
    task = repository.update_task(task_id, payload)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def delete_task(task_id: TaskId):
    if not repository.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found.")

    return Response(status_code=204)