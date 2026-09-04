(function () {
  const documentState = { items: [], searchTimer: null };
  const ui = {
    panel: document.getElementById("documentsPanel"),
    token: document.getElementById("adminTokenInput"),
    department: document.getElementById("documentDepartmentInput"),
    category: document.getElementById("documentCategoryInput"),
    access: document.getElementById("documentAccessInput"),
    version: document.getElementById("documentVersionInput"),
    upload: document.getElementById("documentUploadInput"),
    uploadButton: document.getElementById("documentUploadButton"),
    refreshButton: document.getElementById("documentRefreshButton"),
    search: document.getElementById("documentSearchInput"),
    list: document.getElementById("documentList"),
    count: document.getElementById("documentCount"),
    chunks: document.getElementById("chunkCount"),
    provider: document.getElementById("retrievalProvider"),
    evidenceBody: document.getElementById("evidenceBody"),
    evidenceCount: document.getElementById("evidenceCount"),
    evidenceSubtitle: document.getElementById("evidenceSubtitle"),
  };

  function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = value == null ? "" : String(value);
    return element.innerHTML;
  }

  function notify(message) {
    if (typeof window.showToast === "function") {
      window.showToast(message);
      return;
    }
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    window.setTimeout(() => toast.classList.remove("show"), 2300);
  }

  function requestHeaders() {
    const headers = {};
    const token = ui.token.value.trim();
    if (token) headers["X-Admin-Token"] = token;
    return headers;
  }

  async function requestJson(url, options) {
    const requestOptions = options || {};
    const headers = new Headers(requestOptions.headers || {});
    Object.entries(requestHeaders()).forEach(([name, value]) => headers.set(name, value));
    const response = await fetch(url, { ...requestOptions, headers });
    const raw = await response.text();
    let payload = {};
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch (error) {
        payload = { detail: raw };
      }
    }
    if (!response.ok) throw new Error(payload.detail || `请求失败（${response.status}）`);
    return payload;
  }

  function formatNumber(value) {
    return typeof value === "number" ? value.toLocaleString("zh-CN") : "—";
  }

  function formatBytes(value) {
    const bytes = Number(value) || 0;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString("zh-CN");
  }

  function statusInfo(status) {
    return {
      active: { label: "已启用", className: "active" },
      processing: { label: "处理中", className: "processing" },
      archived: { label: "已归档", className: "archived" },
      failed: { label: "处理失败", className: "failed" },
    }[status] || { label: status || "未知", className: "failed" };
  }

  function renderDocuments() {
    if (!documentState.items.length) {
      ui.list.innerHTML = `<div class="document-empty"><span class="document-empty-icon">＋</span><div><strong>还没有上传企业文档</strong><p>支持 PDF、DOCX、XLSX、TXT、Markdown 和图片 OCR</p></div><button type="button" class="text-button" data-document-action="pick">立即上传</button></div>`;
      return;
    }
    ui.list.innerHTML = documentState.items.map((record) => {
      const status = statusInfo(record.status);
      const metadata = [
        record.department || "未设置部门",
        record.category || "未分类",
        `版本 ${record.version || "v1"}`,
        `${record.chunk_count || 0} 个切片`,
        formatBytes(record.size_bytes),
        formatDate(record.updated_at),
      ].filter(Boolean);
      return `<article class="document-item"><div class="document-icon">${escapeHtml((record.file_type || "doc").slice(0, 4).toUpperCase())}</div><div class="document-main"><div class="document-title-row"><strong title="${escapeHtml(record.filename)}">${escapeHtml(record.filename)}</strong><span class="document-status ${status.className}">${status.label}</span></div><div class="document-meta">${metadata.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}</div>${record.error ? `<p class="document-error">${escapeHtml(record.error)}</p>` : ""}</div><div class="document-item-actions">${record.status === "archived" ? `<button type="button" class="text-button" data-document-action="activate" data-document-id="${escapeHtml(record.document_id)}">切换为当前版本</button>` : ""}<button type="button" class="text-button" data-document-action="reindex" data-document-id="${escapeHtml(record.document_id)}" ${record.status === "processing" ? "disabled" : ""}>重索引</button><button type="button" class="text-button danger" data-document-action="delete" data-document-id="${escapeHtml(record.document_id)}">删除</button></div></article>`;
    }).join("");
  }

  async function loadStats() {
    try {
      const payload = await requestJson("/api/v1/documents/stats");
      ui.count.textContent = formatNumber(payload.documents);
      ui.chunks.textContent = formatNumber(payload.chunks);
      ui.provider.textContent = `${String(payload.sparse_provider || "bm25").toUpperCase()} · ${payload.dense_provider || "hash"}`;
    } catch (error) {
      ui.count.textContent = "—";
      ui.chunks.textContent = "—";
      ui.provider.textContent = "暂不可用";
    }
  }

  async function loadDocuments() {
    const query = ui.search.value.trim();
    const suffix = query ? `?search=${encodeURIComponent(query)}` : "";
    try {
      const payload = await requestJson(`/api/v1/documents${suffix}`);
      documentState.items = payload.items || [];
      renderDocuments();
    } catch (error) {
      ui.list.innerHTML = `<div class="document-empty"><span class="document-empty-icon">!</span><div><strong>文档列表暂时不可用</strong><p>${escapeHtml(error.message)}</p></div><button type="button" class="text-button" data-document-action="refresh">重试</button></div>`;
    }
  }

  function openPicker() {
    ui.upload.click();
  }

  async function uploadDocument() {
    const file = ui.upload.files && ui.upload.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("department", ui.department.value.trim());
    formData.append("category", ui.category.value.trim());
    formData.append("access_level", ui.access.value);
    formData.append("version", ui.version.value.trim());
    ui.uploadButton.disabled = true;
    try {
      await requestJson("/api/v1/documents/upload", { method: "POST", body: formData });
      ui.upload.value = "";
      ui.version.value = "";
      notify("文档已上传并建立索引");
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      notify(error.message || "文档上传失败");
    } finally {
      ui.uploadButton.disabled = false;
    }
  }

  async function reindexDocument(documentId) {
    try {
      await requestJson(`/api/v1/documents/${encodeURIComponent(documentId)}/reindex`, { method: "POST" });
      notify("文档已重新建立索引");
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      notify(error.message || "重索引失败");
    }
  }

  async function activateDocument(documentId) {
    try {
      await requestJson(`/api/v1/documents/${encodeURIComponent(documentId)}/activate`, { method: "POST" });
      notify("文档版本已切换");
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      notify(error.message || "版本切换失败");
    }
  }

  async function deleteDocument(documentId) {
    if (!window.confirm("确定删除这份文档及其检索切片吗？")) return;
    try {
      await requestJson(`/api/v1/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
      notify("文档及索引已删除");
      await Promise.all([loadDocuments(), loadStats()]);
    } catch (error) {
      notify(error.message || "删除失败");
    }
  }

  function renderEvidenceItem(item, index) {
    const isDocument = item.source === "document" || Boolean(item.document_id);
    if (isDocument) {
      const location = [
        item.version ? `版本 ${item.version}` : "",
        item.page !== null && item.page !== undefined ? `第 ${item.page} 页` : "",
        item.chapter || "",
        item.section || "",
      ].filter(Boolean).join(" · ");
      const preview = item.description || item.value || "检索命中文档片段";
      const score = Number(item.score);
      return `<div class="evidence-item document-evidence-item"><div class="evidence-item-top"><span class="evidence-index">${String(index + 1).padStart(2, "0")}</span><span class="evidence-source">企业文档</span>${score > 0 ? `<span class="evidence-score">${score.toFixed(2)}</span>` : ""}</div><strong class="evidence-document-title">${escapeHtml(item.filename || item.document || "未命名文档")}</strong>${location ? `<div class="evidence-location">${escapeHtml(location)}</div>` : ""}<p class="evidence-description">${escapeHtml(preview)}</p></div>`;
    }
    return `<div class="evidence-item"><div class="evidence-item-top"><span class="evidence-index">${String(index + 1).padStart(2, "0")}</span><span class="evidence-source">${escapeHtml(item.source || "medical_kg")}</span></div><div class="evidence-graph"><span class="graph-node">${escapeHtml(item.entity)}</span><span class="graph-relation">— ${escapeHtml(item.relation)} →</span><span class="graph-node">${escapeHtml(item.value)}</span></div>${item.description ? `<p class="evidence-description">${escapeHtml(item.description)}</p>` : ""}</div>`;
  }

  function renderDocumentEvidence(response) {
    const evidence = response.evidence && response.evidence.length ? response.evidence : (response.citations || []);
    ui.evidenceCount.textContent = `${evidence.length} 条`;
    ui.evidenceSubtitle.textContent = evidence.length ? `来自 ${(response.intent && response.intent.label) || "文档检索"}` : "本次未检索到直接证据";
    if (!evidence.length) {
      ui.evidenceBody.innerHTML = `<div class="evidence-empty"><div class="evidence-empty-icon">⌁</div><p>本次检索没有命中直接知识<br>系统已阻止无依据扩展</p></div>`;
    } else {
      ui.evidenceBody.innerHTML = `<div class="evidence-list">${evidence.map(renderEvidenceItem).join("")}</div>`;
    }
    const footer = document.querySelector(".evidence-footer strong");
    if (footer) {
      const label = evidence.some((item) => item.source === "document" || item.document_id) ? "Hybrid RAG" : "Knowledge Graph RAG";
      footer.textContent = "";
      const indicator = document.createElement("i");
      footer.append(indicator, ` ${label}`);
    }
  }

  window.renderDocumentEvidence = renderDocumentEvidence;
  ui.token.value = localStorage.getItem("knowledge-answer-admin-token") || "";
  ui.token.addEventListener("input", () => {
    const token = ui.token.value.trim();
    if (token) localStorage.setItem("knowledge-answer-admin-token", token);
    else localStorage.removeItem("knowledge-answer-admin-token");
  });
  ui.uploadButton.addEventListener("click", openPicker);
  ui.upload.addEventListener("change", uploadDocument);
  ui.refreshButton.addEventListener("click", () => Promise.all([loadDocuments(), loadStats()]));
  ui.search.addEventListener("input", () => {
    window.clearTimeout(documentState.searchTimer);
    documentState.searchTimer = window.setTimeout(loadDocuments, 250);
  });
  ui.list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-document-action]");
    if (!button || button.disabled) return;
    const action = button.dataset.documentAction;
    if (action === "pick") openPicker();
    if (action === "refresh") loadDocuments();
    if (action === "reindex") reindexDocument(button.dataset.documentId);
    if (action === "activate") activateDocument(button.dataset.documentId);
    if (action === "delete") deleteDocument(button.dataset.documentId);
  });
  loadDocuments();
  loadStats();
})();
