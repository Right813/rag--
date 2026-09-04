# 知库智答 · 企业知识库问答 MVP

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
