# 知库智答 · 企业级智能文档知识库与 Hybrid RAG

这是一个基于知识图谱的可解释问答 MVP，按 `README_project2_RAG_KG_requirements.md` 的主链路实现：

```text
问题 → 实体识别 → 意图路由 → 固定 Cypher/本地知识检索 → 证据上下文 → 受控回答
```

当前版本默认提供一套小规模医疗示例知识。知识库没有直接证据时，系统会明确拒答，不让模型自由编造。

## 环境

- Python 3.10+；Docker Desktop / Docker Compose
- Neo4j Community 5：知识图谱和 Cypher 检索
- MySQL `127.0.0.1:3306`：会话、消息、反馈持久化，默认数据库名为 `rag_kg`
- Redis `127.0.0.1:6379`：问答结果缓存，默认无密码
- 可选 LLM：Ollama、vLLM 或 OpenAI-compatible API；未配置时使用 grounded fallback

MySQL 数据库不存在时，应用会尝试用 root 账号自动创建 `rag_kg`。MySQL 或 Redis 暂时不可用时，应用仍可启动并进入降级模式，健康检查会显示具体状态。

## 启动方式

### Docker 启动 Neo4j，主机启动 API

```powershell
Copy-Item .env.example .env
docker compose up -d neo4j
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_kg.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 <http://localhost:8000>，API 文档在 <http://localhost:8000/docs>，Neo4j Browser 在 <http://localhost:7474>。

### API 和 Neo4j 一起用 Docker 启动

本机 MySQL、Redis 通过 `host.docker.internal` 访问：

```powershell
docker compose up -d --build
```

Compose 会读取 `.env` 中的 MySQL/Redis 账号配置；如果运行中的数据库凭据不同，请按实际环境修改 `MYSQL_USER`、`MYSQL_PASSWORD` 和 `MYSQL_DATABASE`，不要将真实凭据提交到 Git。如果 Docker Desktop 无法访问默认 Docker Hub 镜像源，项目默认使用 `public.ecr.aws/docker/library/python:3.11-slim`，也可以在构建时覆盖：`docker compose build --build-arg PYTHON_IMAGE=python:3.11-slim api`。

如果你的 Neo4j 已经占用 `7474` 或 `7687`，请先调整 `docker-compose.yml` 端口，或在 `.env` 中改用已有实例的连接信息。

## 可选 LLM 配置

编辑 `.env`：

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
```

不配置也可以完整演示 MVP；系统会用知识库事实生成回答。生产环境请使用独立密钥管理，不要把真实 Key 提交到仓库。

## API

- `POST /api/v1/chat`：问答，返回实体、意图、证据、耗时和 grounded 标记
- `GET /health`、`GET /api/v1/health`：组件健康检查
- `GET /api/v1/knowledge/stats`：节点、关系、后端和请求统计
- `GET /api/v1/history?session_id=...`：加载会话
- `POST /api/v1/feedback`：保存回答反馈
- `POST /api/v1/knowledge/import`：导入经过校验的实体和关系
- `POST /api/v1/knowledge/import-file`：上传 JSON/CSV 知识文件（实体或关系表）
- `POST /api/v1/reload`：重载种子知识

CSV 实体表使用 `name,type,description,aliases` 表头；CSV 关系表使用 `source,source_type,relation,target,target_type` 表头。文件导入上限为 5MB，管理接口在设置 `ADMIN_TOKEN` 后需要携带 `X-Admin-Token`。

管理接口可通过 `X-Admin-Token` 保护；设置 `ADMIN_TOKEN` 后请求必须携带对应值。

## 测试

```powershell
pytest
```

测试使用 SQLite、内存缓存和本地知识快照，不会修改你的 MySQL、Redis 或 Neo4j 数据。生产前建议另外执行：

```powershell
python -m compileall app scripts
docker compose config
```

## 目录

```text
app/       FastAPI、服务、数据库和知识图谱代码
data/      示例实体与关系
static/    响应式问答工作台
scripts/   知识图谱构建脚本
tests/     单元和接口测试
```

## 企业文档 Hybrid RAG

项目当前定位为企业级智能文档知识库与 Hybrid RAG 问答平台，主链路为：文档解析 → 清洗 → Structure-aware Chunking → Dense/BM25 双路召回 → Fusion → Reranker → Context Builder → LLM → Citation / No-answer。

### 依赖环境

- 必需：Python 3.10+、FastAPI、PyMuPDF、python-docx、openpyxl。
- 必需：MySQL 8（会话、消息和反馈），默认连接 `127.0.0.1:3306`。
- 必需：Redis 5+（问答缓存），默认连接 `127.0.0.1:6379`，可不设置密码。
- 必需：Neo4j 5（当前内置医疗示例和 Graph RAG 兼容能力）。
- 可选：FAISS、Sentence Transformers、PaddleOCR/PaddlePaddle，安装 `requirements-optional.txt` 启用本地高质量模型。
- 不需要：Milvus 不是当前 MVP 的必需依赖；向量索引使用进程内检索，配置 BGE-M3 且安装 FAISS 后自动使用 FAISS。

### 文档处理

- 支持 PDF、DOCX、XLSX、TXT、Markdown、JPG 和 PNG。
- PDF 优先使用 PyMuPDF 读取原生文本；无文本页面在安装 OCR 依赖后自动尝试 PaddleOCR。
- 文档切片保留 `document_id`、文件名、版本、页码、章节、小节、部门和访问级别。
- 默认切片参数为 600 tokens、80 tokens overlap，可通过 `.env` 中的 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 调整。
- 默认使用确定性的 Hash Dense + BM25 + 词法 Reranker，未下载模型时也能启动和演示。

### Hybrid Retrieval

- Dense Retrieval 负责语义召回，BM25 负责编号、标准号和专有名词的精确匹配。
- 支持 `rrf` 和归一化加权 `weighted` 两种 Fusion 策略，默认使用 RRF。
- 默认先召回约 20 条候选，再进行二阶段精排并构建 Top 5 上下文。
- Citation 元数据始终来自检索结果；上下文不足时返回 No-answer，不让 LLM 自由补充事实。
- 需要真实 BGE-M3 时，在 `.env` 配置 `EMBEDDING_MODEL=BAAI/bge-m3`；需要 Cross-Encoder 时配置 `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`。

### 文档生命周期

- 上传文档后自动解析、清洗、切片并建立索引，状态包括 `processing`、`active`、`archived` 和 `failed`。
- 同名新版本上传会自动归档当前活动版本，并保留旧版本记录。
- 管理员可以按文档重新索引，也可以删除原文件、切片和检索索引。
- 文档查询支持部门关键词搜索和访问级别过滤；生产环境请设置强随机 `ADMIN_TOKEN`。

### API

- `POST /api/v1/documents/upload`：上传并索引企业文档。
- `GET /api/v1/documents`、`GET /api/v1/documents/stats`：查询文档列表和索引统计。
- `POST /api/v1/documents/{document_id}/reindex`、`DELETE /api/v1/documents/{document_id}`：重建索引或删除文档。
- `POST /api/v1/retrieval/search`：直接查看 Hybrid Retrieval 结果和分数。
- `POST /api/v1/chat`：返回答案、证据、Citation、检索摘要和 No-answer 状态。
- `POST /api/v1/chat/stream`：通过 SSE 返回 `meta`、`token`、`citations` 和 `done` 事件。

SSE 响应示例：

```text
event: meta
event: token
event: citations
event: done
```

### 配置

基础运行依赖写在 `requirements.txt`；本地高质量 embedding、FAISS 和 OCR 写在 `requirements-optional.txt`，按需安装：

```powershell
pip install -r requirements-optional.txt
```

文档相关 `.env` 变量包括 `DOCUMENT_STORAGE_PATH`、`DOCUMENT_UPLOAD_DIR`、`MAX_UPLOAD_SIZE_BYTES`、`EMBEDDING_MODEL`、`RERANKER_MODEL`、`DENSE_TOP_K`、`BM25_TOP_K`、`RETRIEVAL_TOP_K`、`FUSION_STRATEGY`、`FUSION_ALPHA`、`MAX_CONTEXT_TOKENS`、`NO_ANSWER_THRESHOLD` 和 `OCR_ENABLED`。真实 API Key 只放在本地 `.env`，不要提交到 Git。

### 评测

使用 `data/evaluation/sample_questions.json` 作为格式样例运行离线检索评测：

```powershell
python scripts/evaluate_retrieval.py
```

脚本输出查询数量、Recall@5、Recall@10、MRR、NDCG@5 和 NDCG@10；样例结果不代表生产数据，生产环境应使用真实标注集替换样例。

### Docker
