const state = {
  sessionId: localStorage.getItem("knowledge-answer-session") || createSessionId(),
  messages: [],
  loading: false,
};

const elements = {
  chatMessages: document.getElementById("chatMessages"),
  emptyChat: document.getElementById("emptyChat"),
  queryInput: document.getElementById("queryInput"),
  sendButton: document.getElementById("sendButton"),
  charCount: document.getElementById("charCount"),
  evidenceBody: document.getElementById("evidenceBody"),
  evidenceCount: document.getElementById("evidenceCount"),
  evidenceSubtitle: document.getElementById("evidenceSubtitle"),
  nodeCount: document.getElementById("nodeCount"),
  relationCount: document.getElementById("relationCount"),
  latencyValue: document.getElementById("latencyValue"),
  miniStatusDot: document.getElementById("miniStatusDot"),
  miniStatusText: document.getElementById("miniStatusText"),
  sessionShortId: document.getElementById("sessionShortId"),
  toast: document.getElementById("toast"),
};

function createSessionId() {
  const randomPart = window.crypto && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  const sessionId = randomPart.replaceAll("-", "").slice(0, 16);
  localStorage.setItem("knowledge-answer-session", sessionId);
  return sessionId;
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value == null ? "" : String(value);
  return element.innerHTML;
}

function updateSessionLabel() {
  elements.sessionShortId.textContent = `会话 ${state.sessionId.slice(0, 8)}`;
}

function renderMessages() {
  if (!state.messages.length && !state.loading) {
    elements.chatMessages.innerHTML = elements.emptyChat.outerHTML;
    bindSuggestionCards();
    return;
  }
  elements.chatMessages.innerHTML = state.messages.map((message, index) => renderMessage(message, index)).join("");
  if (state.loading) {
    elements.chatMessages.insertAdjacentHTML("beforeend", `<div class="message"><div class="message-avatar">知</div><div class="message-content"><div class="message-role">知库智答 · 正在检索</div><div class="message-bubble typing-bubble"><i></i><i></i><i></i></div></div></div>`);
  }
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function renderMessage(message, index) {
  const isUser = message.role === "user";
  const role = isUser ? "你" : "知库智答";
  const meta = [];
  if (!isUser && message.intent && message.intent.label) meta.push(`<span class="intent-tag">${escapeHtml(message.intent.label)}</span>`);
  if (!isUser && message.grounded) meta.push(`<span class="grounded-tag">✓ 有据回答</span>`);
  const actions = !isUser ? `<div class="message-actions"><button class="message-action copy-action" data-index="${index}" type="button">复制答案</button><button class="message-action feedback-action" data-index="${index}" type="button">回答有帮助</button></div>` : "";
  return `<div class="message ${isUser ? "user" : "assistant"}"><div class="message-avatar">${isUser ? "我" : "知"}</div><div class="message-content"><div class="message-role">${role}${message.model && !isUser ? ` · ${escapeHtml(message.model)}` : ""}</div><div class="message-bubble">${escapeHtml(message.content)}</div>${meta.length ? `<div class="message-meta">${meta.join("")}<span>${message.latency_ms ? `${Number(message.latency_ms).toFixed(0)} ms` : ""}</span></div>` : ""}${actions}</div></div>`;
}

function renderEvidence(response) {
  if (typeof window.renderDocumentEvidence === "function") {
    window.renderDocumentEvidence(response);
    return;
  }
  const evidence = response.evidence || [];
  elements.evidenceCount.textContent = `${evidence.length} 条`;
  elements.evidenceSubtitle.textContent = evidence.length ? `来自 ${response.intent.label}` : "本次未检索到直接证据";
  if (!evidence.length) {
    elements.evidenceBody.innerHTML = `<div class="evidence-empty"><div class="evidence-empty-icon">⌁</div><p>本次检索没有命中直接知识<br>系统已阻止无依据扩展</p></div>`;
    return;
  }
  elements.evidenceBody.innerHTML = `<div class="evidence-list">${evidence.map((item, index) => `<div class="evidence-item"><div class="evidence-item-top"><span class="evidence-index">${String(index + 1).padStart(2, "0")}</span><span class="evidence-source">${escapeHtml(item.source || "medical_kg")}</span></div><div class="evidence-graph"><span class="graph-node">${escapeHtml(item.entity)}</span><span class="graph-relation">— ${escapeHtml(item.relation)} →</span><span class="graph-node">${escapeHtml(item.value)}</span></div>${item.description ? `<p class="evidence-description">${escapeHtml(item.description)}</p>` : ""}</div>`).join("")}</div>`;
}

async function sendQuery(queryOverride) {
  if (state.loading) return;
  const query = (queryOverride || elements.queryInput.value).trim();
  if (!query) {
    showToast("请先输入一个问题");
    elements.queryInput.focus();
    return;
  }
  state.loading = true;
  state.messages.push({ role: "user", content: query });
  const assistantMessage = { role: "assistant", content: "" };
  state.messages.push(assistantMessage);
  elements.queryInput.value = "";
  resizeInput();
  updateCharacterCount();
  toggleLoading(true);
  renderMessages();
  try {
    const response = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: state.sessionId }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "问答服务暂时不可用");
    }
    await consumeChatStream(response, assistantMessage);
  } catch (error) {
    assistantMessage.content = `抱歉，${error.message || "服务暂时不可用"}。`;
    showToast("请求未完成，请检查服务状态");
  } finally {
    state.loading = false;
    toggleLoading(false);
    updateSessionLabel();
    renderMessages();
  }
}

async function consumeChatStream(response, assistantMessage) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  const handleEvent = (block) => {
    const lines = block.split("\n");
    const eventName = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
    const dataLine = lines.find((line) => line.startsWith("data:"));
    if (!eventName || !dataLine) return;
    const payload = JSON.parse(dataLine.slice(5).trim());
    if (eventName === "meta") {
      state.sessionId = payload.session_id || state.sessionId;
      localStorage.setItem("knowledge-answer-session", state.sessionId);
    }
    if (eventName === "token") {
      assistantMessage.content += payload.text || "";
      renderMessages();
    }
    if (eventName === "done") {
      Object.assign(assistantMessage, payload, { role: "assistant", content: payload.answer || assistantMessage.content });
      renderEvidence(payload);
      updateStats();
      completed = true;
    }
  };
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    blocks.filter(Boolean).forEach(handleEvent);
    if (done) break;
  }
  if (buffer.trim()) handleEvent(buffer.trim());
  if (!completed) throw new Error("问答流意外结束");
}

function toggleLoading(loading) {
  elements.sendButton.disabled = loading;
  elements.queryInput.disabled = loading;
}

function updateCharacterCount() {
  elements.charCount.textContent = `${elements.queryInput.value.length} / 500`;
}

function resizeInput() {
  elements.queryInput.style.height = "auto";
  elements.queryInput.style.height = `${Math.min(elements.queryInput.scrollHeight, 110)}px`;
}

function bindSuggestionCards() {
  document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => sendQuery(card.dataset.query));
  });
}

async function loadHistory() {
  try {
    const response = await fetch(`/api/v1/history?session_id=${encodeURIComponent(state.sessionId)}`);
    if (!response.ok) return;
    const payload = await response.json();
    state.messages = (payload.messages || []).map((message) => ({
      role: message.role,
      content: message.content,
      grounded: message.grounded,
      intent: message.intent ? { label: message.intent } : null,
      evidence: message.evidence || [],
      latency_ms: message.latency_ms,
    }));
    if (state.messages.length) {
      const lastAssistant = [...state.messages].reverse().find((message) => message.role === "assistant");
      if (lastAssistant) renderEvidence({ evidence: lastAssistant.evidence, citations: lastAssistant.citations, intent: lastAssistant.intent || { label: "上次检索" } });
      renderMessages();
    }
  } catch (_) {
    renderMessages();
  }
}

async function updateStats() {
  try {
    const response = await fetch("/api/v1/knowledge/stats");
    if (!response.ok) return;
    const payload = await response.json();
    const graph = payload.graph || {};
    elements.nodeCount.textContent = formatNumber(graph.nodes);
    elements.relationCount.textContent = formatNumber(graph.relations);
    if (payload.metrics && payload.metrics.requests) elements.latencyValue.innerHTML = `${Number(payload.metrics.average_latency_ms).toFixed(0)}<span class="unit">ms</span>`;
  } catch (_) {
    elements.nodeCount.textContent = "—";
    elements.relationCount.textContent = "—";
  }
}

async function updateHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    const healthy = payload.status === "ok";
    elements.miniStatusDot.style.background = healthy ? "#22c89a" : "#efad55";
    elements.miniStatusText.textContent = healthy ? "API 与知识服务已连接" : "部分服务使用本地降级模式";
  } catch (_) {
    elements.miniStatusDot.style.background = "#e87878";
    elements.miniStatusText.textContent = "无法连接 API 服务";
  }
}

function formatNumber(value) {
  return typeof value === "number" ? value.toLocaleString("zh-CN") : "—";
}

function startNewChat() {
  state.sessionId = createSessionId();
  state.messages = [];
  renderEvidence({ evidence: [], intent: { label: "等待下一次检索" } });
  updateSessionLabel();
  renderMessages();
  showToast("已创建新会话");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 2300);
}

async function handleMessageAction(event) {
  const button = event.target.closest("button");
  if (!button) return;
  const index = Number(button.dataset.index);
  const message = state.messages[index];
  if (!message) return;
  if (button.classList.contains("copy-action")) {
    try {
      await navigator.clipboard.writeText(message.content);
      showToast("答案已复制");
    } catch (_) {
      showToast("复制失败，请手动选择文本");
    }
  }
  if (button.classList.contains("feedback-action")) {
    try {
      await fetch("/api/v1/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: state.sessionId, rating: 5 }) });
      button.textContent = "已感谢反馈";
      button.disabled = true;
      showToast("感谢你的反馈");
    } catch (_) {
      showToast("反馈暂时未提交");
    }
  }
}

elements.sendButton.addEventListener("click", () => sendQuery());
elements.queryInput.addEventListener("input", () => { updateCharacterCount(); resizeInput(); });
elements.queryInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendQuery();
  }
});
elements.chatMessages.addEventListener("click", handleMessageAction);
document.getElementById("newChatButton").addEventListener("click", startNewChat);
document.getElementById("knowledgeNav").addEventListener("click", () => document.getElementById("documentsPanel").scrollIntoView({ behavior: "smooth", block: "start" }));
document.getElementById("docsNav").addEventListener("click", () => window.open("/docs", "_blank", "noopener"));
document.getElementById("mobileMenu").addEventListener("click", () => document.getElementById("sidebar").classList.toggle("mobile-open"));

updateSessionLabel();
updateCharacterCount();
bindSuggestionCards();
loadHistory();
updateStats();
updateHealth();

