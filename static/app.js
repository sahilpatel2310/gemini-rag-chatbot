const sessionStorageKey = "gemini-rag-session-id";
const indexedFilesKey = "gemini-rag-indexed-files";

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadFeedback = document.getElementById("uploadFeedback");
const fileList = document.getElementById("fileList");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const messages = document.getElementById("messages");
const chatStatus = document.getElementById("chatStatus");
const healthBadge = document.getElementById("healthBadge");
const sessionIdNode = document.getElementById("sessionId");
const messageTemplate = document.getElementById("messageTemplate");

let sessionId = localStorage.getItem(sessionStorageKey) || crypto.randomUUID();
localStorage.setItem(sessionStorageKey, sessionId);
if (sessionIdNode) sessionIdNode.textContent = sessionId;

renderIndexedFiles(JSON.parse(localStorage.getItem(indexedFilesKey) || "[]"));
checkHealth();

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    setUploadFeedback("Select at least one PDF or TXT file.", "error");
    return;
  }

  const button = uploadForm.querySelector("button");
  button.disabled = true;
  setUploadFeedback("Uploading and indexing documents...", "muted");

  const formData = new FormData();
  for (const file of fileInput.files) {
    formData.append("files", file);
  }

  try {
    const response = await fetch("/documents/upload", {
      method: "POST",
      body: formData,
    });

    const payload = await parseApiResponse(response, "Upload failed.");
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }

    const storedFiles = JSON.parse(localStorage.getItem(indexedFilesKey) || "[]");
    const merged = [...new Set([...storedFiles, ...payload.ingested_files])];
    localStorage.setItem(indexedFilesKey, JSON.stringify(merged));
    renderIndexedFiles(merged);

    setUploadFeedback(
      `Indexed ${payload.chunks_added} chunks from ${payload.ingested_files.join(", ")}. Total vectors: ${payload.total_indexed_documents}.`,
      "success"
    );
    fileInput.value = "";
  } catch (error) {
    setUploadFeedback(error.message, "error");
  } finally {
    button.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  appendMessage("user", question);
  questionInput.value = "";
  chatStatus.textContent = "Thinking...";
  const button = chatForm.querySelector("button");
  button.disabled = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    });

    const payload = await parseApiResponse(response, "Request failed.");
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    appendMessage("assistant", payload.answer, payload.sources);
    if (payload.session_id && payload.session_id !== sessionId) {
      sessionId = payload.session_id;
      localStorage.setItem(sessionStorageKey, sessionId);
      if (sessionIdNode) sessionIdNode.textContent = sessionId;
    }
    chatStatus.textContent = "Idle";
  } catch (error) {
    appendMessage("assistant", `Error: ${error.message}`);
    chatStatus.textContent = "Failed";
  } finally {
    button.disabled = false;
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const payload = await parseApiResponse(response, "Health check failed");
    if (!response.ok || !payload.service_ready) {
      throw new Error(payload.error || "Health check failed");
    }
    if (healthBadge) {
      healthBadge.textContent = "Online";
      healthBadge.className = "badge ok";
    }
  } catch {
    if (healthBadge) {
      healthBadge.textContent = "Offline";
      healthBadge.className = "badge error";
    }
  }
}

function setUploadFeedback(text, tone) {
  uploadFeedback.textContent = text;
  uploadFeedback.className = `status-box ${tone}`;
}

function renderIndexedFiles(files) {
  fileList.innerHTML = "";
  if (!files.length) {
    const item = document.createElement("li");
    item.textContent = "No indexed files recorded in this browser yet.";
    fileList.appendChild(item);
    return;
  }

  for (const file of files) {
    const item = document.createElement("li");
    item.textContent = file;
    fileList.appendChild(item);
  }
}

function appendMessage(role, text, sources = []) {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const roleNode = fragment.querySelector(".message-role");
  const bodyNode = fragment.querySelector(".message-body");
  const sourcesNode = fragment.querySelector(".sources");

  article.classList.add(role);
  roleNode.textContent = role === "user" ? "User" : "Assistant";

  if (role === "assistant" && typeof marked !== "undefined") {
    bodyNode.innerHTML = marked.parse(text);
  } else {
    bodyNode.textContent = text;
  }

  if (Array.isArray(sources) && sources.length) {
    for (const source of sources) {
      const item = document.createElement("li");
      item.className = "source-card";
      const title = document.createElement("span");
      title.className = "source-title";
      title.textContent = source.page ? `${source.source} - page ${source.page}` : source.source;
      const snippet = document.createElement("p");
      snippet.textContent = source.snippet;
      item.append(title, snippet);
      sourcesNode.appendChild(item);
    }
  } else {
    sourcesNode.remove();
  }

  messages.appendChild(fragment);
  messages.scrollTop = messages.scrollHeight;
}

async function parseApiResponse(response, fallbackMessage) {
  const raw = await response.text();

  if (!raw) {
    if (!response.ok) {
      throw new Error(response.statusText || fallbackMessage);
    }
    return {};
  }

  try {
    return JSON.parse(raw);
  } catch {
    if (!response.ok) {
      throw new Error(raw || response.statusText || fallbackMessage);
    }
    throw new Error(fallbackMessage);
  }
}
