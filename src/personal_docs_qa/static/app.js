const state = {
  config: null,
  indexBuiltWithEmbeddings: false,
  indexReady: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const elements = {
  openaiStatus: $("#openai-status"),
  embeddingModel: $("#embedding-model"),
  indexEmbeddings: $("#index-embeddings"),
  globalWarning: $("#global-warning"),
  indexForm: $("#index-form"),
  uploadButton: $("#upload-button"),
  fileUpload: $("#file-upload"),
  folderPath: $("#folder-path"),
  indexMode: $("#index-retrieval-mode"),
  askForm: $("#ask-form"),
  question: $("#question"),
  askMode: $("#ask-retrieval-mode"),
  answerMode: $("#answer-mode"),
  indexSummary: $("#index-summary"),
  answerSection: $("#answer-section"),
  answerTitle: $("#answer-title"),
  retrievalBadge: $("#retrieval-badge"),
  answerBadge: $("#answer-badge"),
  confidenceBadge: $("#confidence-badge"),
  fallbackSummary: $("#fallback-summary"),
  answerText: $("#answer-text"),
  warnings: $("#warnings"),
  sourceCount: $("#source-count"),
  sources: $("#sources"),
  sourceTemplate: $("#source-template"),
};

function titleCase(value) {
  if (!value) return "Unknown";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function setNotice(message, tone = "warning") {
  if (!message) {
    elements.globalWarning.classList.add("hidden");
    elements.globalWarning.textContent = "";
    return;
  }
  elements.globalWarning.className = `notice ${tone}`;
  elements.globalWarning.textContent = message;
}

function setStatusText(node, text, positive = false) {
  node.textContent = text;
  node.classList.toggle("ok", positive);
  node.classList.toggle("warn", !positive);
}

function setPipeline(activeSteps = []) {
  const active = new Set(activeSteps);
  $$(".pipeline-step").forEach((step) => {
    step.classList.toggle("active", active.has(step.dataset.step));
    step.classList.toggle("complete", active.has("done"));
  });
}

function setFlow(activeSteps = []) {
  const active = new Set(activeSteps);
  $$(".flow-step").forEach((step) => {
    step.classList.toggle("active", active.has(step.dataset.flow));
    step.classList.toggle("complete", active.has("done"));
  });
}

function animateSteps(steps, setter, delay = 260) {
  setter([]);
  steps.forEach((step, index) => {
    window.setTimeout(() => setter(steps.slice(0, index + 1)), delay * index);
  });
}

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || "Request failed.");
  }
  return body;
}

async function loadConfig() {
  try {
    const config = await fetch("/api/config").then(readJson);
    state.config = config;
    setStatusText(
      elements.openaiStatus,
      config.openai_available ? "Configured" : "Not configured",
      config.openai_available,
    );
    elements.embeddingModel.textContent = config.embedding_model || "Not set";
    elements.embeddingModel.classList.remove("warn");
    elements.embeddingModel.classList.add("ok");
    state.indexReady = Boolean(config.index_present);
    state.indexBuiltWithEmbeddings = Boolean(config.index_built_with_embeddings);
    setStatusText(
      elements.indexEmbeddings,
      !config.index_present ? "Not indexed" : config.index_built_with_embeddings ? "Yes" : "No",
      Boolean(config.index_built_with_embeddings),
    );
    if (config.index_present) {
      elements.indexSummary.innerHTML = `
        Current web index is available.
        Built with <strong>${config.index_retrieval_mode_built || "unknown"}</strong>.
      `;
    }
    if (!config.openai_available) {
      setNotice("OpenAI is not configured; auto mode will use local fallback.");
    }
  } catch (error) {
    setNotice(error.message || "Could not load app configuration.", "error");
  }
}

function updateIndexStatus(response) {
  state.indexReady = true;
  state.indexBuiltWithEmbeddings = Boolean(response.embeddings_created);
  setStatusText(
    elements.indexEmbeddings,
    response.embeddings_created ? "Yes" : "No",
    response.embeddings_created,
  );
  elements.indexSummary.innerHTML = `
    <strong>${response.document_count}</strong> documents,
    <strong>${response.chunk_count}</strong> chunks.
    Built with <strong>${response.retrieval_mode_built}</strong>
    after requesting <strong>${response.retrieval_mode_requested}</strong>.
  `;
  renderWarnings(response.warnings || [], elements.indexSummary);
}

function renderWarnings(warnings, target = elements.warnings) {
  if (!warnings.length) {
    if (target === elements.warnings) {
      target.classList.add("hidden");
      target.innerHTML = "";
    }
    return;
  }

  const list = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  const markup = `<ul>${list}</ul>`;
  if (target === elements.warnings) {
    target.classList.remove("hidden");
    target.innerHTML = markup;
  } else {
    target.insertAdjacentHTML("beforeend", `<div class="inline-warning">${markup}</div>`);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function ingestPath(event) {
  event.preventDefault();
  const retrievalMode = elements.indexMode.value;
  elements.indexSummary.textContent = "Indexing documents...";
  animateSteps(["load", "chunk", "tfidf", "embeddings"], setPipeline);

  try {
    const response = await fetch("/api/ingest-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder_path: elements.folderPath.value || "sample_docs",
        retrieval_mode: retrievalMode,
      }),
    }).then(readJson);
    setPipeline(["done"]);
    updateIndexStatus(response);
    if (!response.embeddings_created && ["embedding", "hybrid"].includes(retrievalMode)) {
      setNotice("This index was built without embeddings. Re-index with hybrid or embedding mode.", "warning");
    }
  } catch (error) {
    setPipeline([]);
    elements.indexSummary.textContent = error.message;
    setNotice(error.message, "error");
  }
}

async function uploadAndIndex() {
  const files = Array.from(elements.fileUpload.files || []);
  if (!files.length) {
    setNotice("Choose one or more files before uploading.", "warning");
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.webkitRelativePath || file.name));
  formData.append("retrieval_mode", elements.indexMode.value);
  elements.indexSummary.textContent = "Uploading and indexing files...";
  animateSteps(["load", "chunk", "tfidf", "embeddings"], setPipeline);

  try {
    const response = await fetch("/api/upload-and-index", {
      method: "POST",
      body: formData,
    }).then(readJson);
    setPipeline(["done"]);
    updateIndexStatus(response);
  } catch (error) {
    setPipeline([]);
    elements.indexSummary.textContent = error.message;
    setNotice(error.message, "error");
  }
}

function renderBadges(response) {
  const retrieval = titleCase(response.retrieval_mode_used);
  const answer = titleCase(response.answer_mode_used);
  const confidence = titleCase(response.confidence);

  elements.retrievalBadge.textContent = `Retrieval: ${retrieval}`;
  elements.answerBadge.textContent = `Answer: ${answer}`;
  elements.confidenceBadge.textContent = `Confidence: ${confidence}`;
  elements.retrievalBadge.dataset.tone = response.retrieval_mode_used || "tfidf";
  elements.answerBadge.dataset.tone = response.answer_mode_used || "local";
  elements.confidenceBadge.dataset.tone = response.confidence || "low";
}

function renderFallbacks(response) {
  const messages = [];
  if (response.retrieval_fallback_used) {
    messages.push(`Retrieval fell back from ${response.retrieval_mode_requested} to ${response.retrieval_mode_used}.`);
  }
  if (response.answer_fallback_used) {
    messages.push("OpenAI failed or was unavailable; local fallback was used.");
  }
  if (
    !state.indexBuiltWithEmbeddings &&
    ["auto", "embedding", "hybrid"].includes(response.retrieval_mode_requested) &&
    response.retrieval_mode_used === "tfidf"
  ) {
    messages.push("This index was built without embeddings. Re-index with hybrid or embedding mode.");
  }

  if (!messages.length) {
    elements.fallbackSummary.classList.add("hidden");
    elements.fallbackSummary.textContent = "";
    return;
  }

  elements.fallbackSummary.classList.remove("hidden");
  elements.fallbackSummary.innerHTML = messages.map((message) => `<p>${escapeHtml(message)}</p>`).join("");
}

function renderSources(sources = []) {
  elements.sources.innerHTML = "";
  elements.sourceCount.textContent = `${sources.length} source${sources.length === 1 ? "" : "s"}`;

  sources.forEach((source) => {
    const card = elements.sourceTemplate.content.firstElementChild.cloneNode(true);
    const toggle = card.querySelector(".source-toggle");
    const excerpt = card.querySelector(".source-excerpt");

    card.querySelector(".source-rank").textContent = `#${source.rank}`;
    card.querySelector(".source-file").textContent = source.file_name;
    card.querySelector(".source-score").textContent = `Score ${source.score_display || formatScore(source.score)}`;
    const sourceMode = card.querySelector(".source-mode");
    sourceMode.textContent = titleCase(source.retrieval_mode_used);
    sourceMode.dataset.tone = source.retrieval_mode_used || "tfidf";
    card.querySelector(".source-type").textContent = source.file_type || "FILE";
    card.querySelector(".source-page").textContent = source.page_label || "";
    card.querySelector(".source-tfidf").textContent = `TF-IDF ${source.score_tfidf_display || formatScore(source.score_tfidf)}`;
    card.querySelector(".source-embedding").textContent =
      `Embedding ${source.score_embedding_display || formatScore(source.score_embedding)}`;
    card.querySelector(".source-chunk").textContent = source.chunk_id || "";
    excerpt.innerHTML = source.excerpt_html || "";

    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      excerpt.classList.toggle("collapsed", expanded);
    });

    elements.sources.appendChild(card);
  });
}

async function askQuestion(event) {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) {
    setNotice("Enter a question first.", "warning");
    return;
  }

  elements.answerSection.classList.remove("hidden");
  elements.answerTitle.textContent = "Working...";
  elements.answerText.textContent = "Retrieving passages and preparing an answer...";
  elements.answerText.classList.add("loading");
  elements.sources.innerHTML = "";
  renderWarnings([]);
  animateSteps(["retrieve", "answer"], setPipeline);
  animateSteps(["query", "retrieval", "synthesis", "sources"], setFlow);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 5,
        retrieval_mode: elements.askMode.value,
        answer_mode: elements.answerMode.value,
      }),
    }).then(readJson);

    setPipeline(["done"]);
    setFlow(["done"]);
    elements.answerText.classList.remove("loading");
    elements.answerTitle.textContent = response.question;
    elements.answerText.textContent = response.answer;
    renderBadges(response);
    renderFallbacks(response);
    renderWarnings(response.warnings || []);
    renderSources(response.sources || []);
  } catch (error) {
    setPipeline([]);
    setFlow([]);
    elements.answerText.classList.remove("loading");
    elements.answerTitle.textContent = "Could not answer";
    elements.answerText.textContent = error.message;
    setNotice(error.message, "error");
  }
}

elements.indexForm?.addEventListener("submit", ingestPath);
elements.uploadButton?.addEventListener("click", uploadAndIndex);
elements.askForm?.addEventListener("submit", askQuestion);

loadConfig();
