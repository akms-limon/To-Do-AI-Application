const taskList = document.querySelector("#task-list");
const taskCount = document.querySelector("#task-count");
const boardStatus = document.querySelector("#board-status");
const refreshButton = document.querySelector("#refresh-button");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = chatForm.querySelector('button[type="submit"]');
const cancelButton = document.querySelector("#cancel-button");

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

const THREAD_KEY = "taskflow-v3-thread";
const PENDING_KEY = "taskflow-v3-pending-message";

let threadId = sessionStorage.getItem(THREAD_KEY);

if (!threadId) {
  threadId = crypto.randomUUID();
  sessionStorage.setItem(THREAD_KEY, threadId);
}

const TRANSCRIPT_KEY = "taskflow-v3-transcript";
let restoredTranscript = [];
try {
  const stored = JSON.parse(sessionStorage.getItem(TRANSCRIPT_KEY) || "[]");
  if (Array.isArray(stored)) restoredTranscript = stored;
} catch { sessionStorage.removeItem(TRANSCRIPT_KEY); }

function saveTranscript() {
  const transcript = [...messages.querySelectorAll(".message")].slice(-60).map((element) => ({
    text: element.textContent,
    role: element.classList.contains("user-message") ? "user" : "assistant",
  }));
  try { sessionStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(transcript)); } catch { /* Storage may be full. */ }
}

let pendingRequest = null;
let sending = false;

try {
  pendingRequest = JSON.parse(
    sessionStorage.getItem(PENDING_KEY) || "null"
  );
} catch {
  sessionStorage.removeItem(PENDING_KEY);
}

function addMessage(text, role) {
  const element = document.createElement("div");

  element.classList.add(
    "message",
    role === "user" ? "user-message" : "assistant-message"
  );

  element.textContent = text;
  messages.append(element);
  messages.scrollTop = messages.scrollHeight;

  return element;
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
      throw new Error(`Task request failed: ${response.status}`);
    }

    const tasks = await response.json();

    if (!Array.isArray(tasks)) {
      throw new Error("Unexpected task response");
    }

    taskList.replaceChildren(...tasks.map(createTaskCard));

    taskCount.textContent =
      `${tasks.length} ${tasks.length === 1 ? "task" : "tasks"}`;

    boardStatus.textContent = tasks.length
      ? ""
      : "No tasks yet. Ask the assistant to create one.";
  } catch (error) {
    boardStatus.classList.add("error");
    boardStatus.textContent =
      "Could not refresh tasks. Displayed tasks may be outdated.";
    console.error(error);
  } finally {
    refreshButton.disabled = false;
  }
}

function updateComposer() {
  messageInput.disabled = sending || pendingRequest !== null;
  sendButton.disabled = sending;
  cancelButton.hidden = !pendingRequest || sending;
  cancelButton.disabled = sending;
  sendButton.textContent = sending
    ? "Working…"
    : pendingRequest
      ? "Retry"
      : "Send";
}

function cancelPendingRequest() {
  if (sending || !pendingRequest) {
    return;
  }

  pendingRequest = null;
  sessionStorage.removeItem(PENDING_KEY);

  // Abandon the failed workflow checkpoint so a new message starts cleanly.
  threadId = crypto.randomUUID();
  sessionStorage.setItem(THREAD_KEY, threadId);

  addMessage("The previous request was cancelled. You can send a new instruction.", "assistant");
  saveTranscript();
  updateComposer();
  messageInput.focus();
}

async function submitPendingRequest() {
  if (sending || !pendingRequest) {
    return;
  }

  sending = true;
  updateComposer();

  const progress = addMessage("Working on your request…", "assistant");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pendingRequest),
    });

    const data = await response.json();

    if (!response.ok) {
      progress.textContent =
        typeof data.detail === "string"
          ? data.detail
          : "The server rejected this request. Check the API input.";

      // Retain the original request ID. A retry must not become a new mutation.
      return;
    }

    if (typeof data.reply !== "string") {
      throw new Error("Unexpected chat response");
    }

    progress.textContent = data.reply;

    pendingRequest = null;
    sessionStorage.removeItem(PENDING_KEY);

    messageInput.placeholder =
      data.status === "clarification"
        ? "Answer the question, or type cancel"
        : "Describe your next task instruction";
  } catch (error) {
    progress.textContent =
      "The response could not be received. Click Retry to resend the same " +
      "message ID. Do not submit the operation as a new message.";

    console.error(error);
  } finally {
    sending = false;
    saveTranscript();
    updateComposer();
    await loadTasks();

    if (!pendingRequest) {
      messageInput.focus();
    }
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (sending) {
    return;
  }

  if (pendingRequest) {
    await submitPendingRequest();
    return;
  }

  const text = messageInput.value.trim();

  if (!text) {
    messageInput.setCustomValidity("Enter a task instruction.");
    messageInput.reportValidity();
    return;
  }

  pendingRequest = {
    thread_id: threadId,
    message_id: crypto.randomUUID(),
    message: text,
  };

  sessionStorage.setItem(
    PENDING_KEY,
    JSON.stringify(pendingRequest)
  );

  addMessage(text, "user");
  saveTranscript();
  messageInput.value = "";

  await submitPendingRequest();
});

messageInput.addEventListener("input", () => {
  messageInput.setCustomValidity("");
});

cancelButton.addEventListener("click", cancelPendingRequest);
refreshButton.addEventListener("click", loadTasks);

for (const item of restoredTranscript) {
  if (typeof item.text === "string" && ["user", "assistant"].includes(item.role)) {
    addMessage(item.text, item.role);
  }
}

if (pendingRequest) {
  addMessage(
    "An unanswered submission was restored. Click Retry to retrieve or " +
    "complete that same request before sending another.",
    "assistant"
  );
} else if (restoredTranscript.length === 0) {
  addMessage(
    "Welcome to TaskFlow. Ask me to create, update, list, or search tasks. " +
    "If I need clarification, reply here.",
    "assistant"
  );
}

messageInput.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    event.keyCode !== 229
  ) {
    event.preventDefault();

    if (!sending && !sendButton.disabled) {
      chatForm.requestSubmit(sendButton);
    }
  }
});

updateComposer();
loadTasks();