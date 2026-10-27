const taskList = document.querySelector("#task-list");
const taskCount = document.querySelector("#task-count");
const boardStatus = document.querySelector("#board-status");
const refreshButton = document.querySelector("#refresh-button");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");

const statusLabels = {
  to_do: "To do",
  in_progress: "In progress",
  done: "Done",
};

const priorityLabels = {
  low: "Low",
  medium: "Medium",
  high: "High",
};

function addMessage(text, role) {
  const message = document.createElement("div");

  message.classList.add(
    "message",
    role === "user" ? "user-message" : "assistant-message"
  );

  // Treat user and task content as text, never executable HTML.
  message.textContent = text;
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
}

function createTaskCard(task) {
  const card = document.createElement("li");
  card.classList.add("task-card");

  if (task.status === "done") {
    card.classList.add("completed");
  }

  const content = document.createElement("div");
  content.className = "task-content";

  const title = document.createElement("h4");
  title.className = "task-title";
  title.textContent = task.title;
  content.append(title);

  if (task.description) {
    const description = document.createElement("p");
    description.className = "task-description";
    description.textContent = task.description;
    content.append(description);
  }

  const metadata = document.createElement("p");
  metadata.className = "task-meta";
  metadata.textContent = [
    `#${task.id}`,
    statusLabels[task.status] ?? task.status,
    task.due_date ? `Due: ${task.due_date}` : "No due date",
  ].join(" • ");
  content.append(metadata);

  const priority = document.createElement("span");
  priority.classList.add("badge");

  if (Object.hasOwn(priorityLabels, task.priority)) {
    priority.classList.add(`priority-${task.priority}`);
  }

  priority.textContent =
    priorityLabels[task.priority] ?? task.priority;

  card.append(content, priority);
  return card;
}

async function loadTasks() {
  refreshButton.disabled = true;
  boardStatus.classList.remove("error");
  boardStatus.textContent = "Loading tasks…";

  try {
    const response = await fetch("/api/tasks");

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const tasks = await response.json();

    if (!Array.isArray(tasks)) {
      throw new Error("Unexpected task response");
    }

    const cards = tasks.map(createTaskCard);
    taskList.replaceChildren(...cards);

    taskCount.textContent =
      `${tasks.length} ${tasks.length === 1 ? "task" : "tasks"}`;

    boardStatus.textContent = tasks.length
      ? ""
      : "No tasks yet. Create one through the API docs for now.";
  } catch (error) {
    boardStatus.classList.add("error");
    boardStatus.textContent =
      "Could not refresh tasks. Any displayed tasks may be outdated. Try again.";
    console.error("Task loading failed:", error);
  } finally {
    refreshButton.disabled = false;
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const text = messageInput.value.trim();

  if (!text) {
    messageInput.setCustomValidity("Enter a task instruction.");
    messageInput.reportValidity();
    return;
  }

  addMessage(text, "user");
  messageInput.value = "";

  addMessage(
    "Chat execution is not connected yet. Your message has not changed any tasks.",
    "assistant"
  );

  messageInput.focus();
});

messageInput.addEventListener("input", () => {
  messageInput.setCustomValidity("");
});

refreshButton.addEventListener("click", loadTasks);

addMessage(
  "Welcome to TaskFlow. Your saved tasks appear on the board. AI chat will be connected in a later step.",
  "assistant"
);

loadTasks();