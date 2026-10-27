from contextlib import asynccontextmanager
from typing import Annotated
from pathlib import Path as FilePath

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException, Path, Response

from app import repository
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


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(STATIC_DIRECTORY / "index.html")

TaskId = Annotated[int, Path(gt=0)]


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


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
    deleted = repository.delete_task(task_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found.")

    return Response(status_code=204)