> **企业级医疗智能知识库与 Hybrid RAG 问答平台**

核心路线：

**文档知识库 → 文档解析 → Chunk → BGE-M3 → Milvus Dense + BM25 → Hybrid Retrieval → Reranker → Context Builder → LLM → Citation / No-answer → Evaluation → FastAPI → Docker**

Milvus 本身已经支持 Dense + Sparse/BM25 的 Hybrid Search、RRF/Weighted Ranker，也支持 BGE-M3 集成，所以这套架构是比较自然的。

# 二、最终版本应该变成什么

我建议最终项目名称：

> **医疗智能知识库与企业级 Hybrid RAG 问答平台**

如果以后想弱化医疗领域，也可以改成：

> **企业级智能文档知识库与 Hybrid RAG 问答平台**

但你现在这个仓库是医疗项目，所以第一种更自然。

最终架构：

```
                医疗知识来源
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    医疗指南       药品说明书      医学文献
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                文档上传服务
                     │
                     ↓
              Document Parser
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      原生文本                 OCR
   PDF/DOCX/XLSX              图片PDF
          │                     │
          └──────────┬──────────┘
                     ↓
                 Text Cleaner
                     ↓
             Structure Parser
                     ↓
           Structure-aware Chunk
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       BGE-M3                 BM25
          ↓                     ↓
      Dense Vector          Sparse Vector
          │                     │
          └──────────┬──────────┘
                     ↓
                  Milvus
                     │
             Hybrid Retrieval
                     ↓
              Top 20 / Top 30
                     ↓
                 Reranker
                     ↓
                  Top 5
                     ↓
              Context Builder
                     ↓
                   LLM
              ┌──────┴──────┐
              ↓             ↓
            Answer       Citation
              │
              ↓
          No-answer检测
              │
              ↓
             API
              │
        ┌─────┴─────┐
        ↓           ↓
      Web UI      业务系统
```

# 三、Milvus 怎么设计

这里是这次升级的重点。

不要：

```
BGE-M3
 ↓
FAISS
```

改成：

```
BGE-M3
 ↓
Milvus
 ├── Dense Vector
 ├── Sparse/BM25
 └── Metadata
```

Milvus 当前文档已经支持 Hybrid Search，可以把 Dense 和 Sparse 两路搜索结果进行融合；BM25 也可以直接通过 Milvus 的全文检索能力实现。

------

# 四、Milvus Collection 怎么设计

建议建立：

```
medical_knowledge_chunks
```

核心字段：

```
id
document_id
document_version
content
title
department
category
source
page
section
chunk_index

dense_vector
sparse_vector

created_at
updated_at
status
access_level
```

例如：

```
id:
medical_000001

document_id:
clinical_guideline_001

document_version:
v3

title:
2型糖尿病诊疗指南

section:
第三章 / 药物治疗

page:
28

content:
对于2型糖尿病患者，应根据患者年龄、
病程、并发症以及低血糖风险选择治疗方案。

dense_vector:
[1024维]

sparse_vector:
BM25 sparse vector
```

------

# 五、为什么我推荐 BGE-M3

你之前已经提过 BGE-M3，这里非常适合。

BGE-M3 本身就支持多语言、Dense Retrieval 和 Sparse Retrieval，并且 Milvus 官方提供了 BGE-M3 的集成方式。

所以你的项目可以形成：

```
                BGE-M3
                  │
       ┌──────────┴──────────┐
       ↓                     ↓
 Dense Embedding        Sparse Retrieval
       ↓                     ↓
   Dense Vector             BM25
       ↓                     ↓
       └──────────┬──────────┘
                  ↓
                Milvus
```

这比：

```
Embedding
 ↓
FAISS
```

更符合现在企业级 RAG 的工程思路。

# 六、Milvus 不只是“存向量”

这是你面试时非常重要的一点。

不要说：

> “我们用了 Milvus，因为它是向量数据库。”

应该说：

> **Milvus 在项目中同时承担 Dense Semantic Retrieval 和 Sparse Lexical Retrieval，并通过 Hybrid Search 对两路召回结果进行融合，从而解决纯向量检索对医学术语、疾病名称、药品名称、剂量等精确关键词匹配能力不足的问题。**

例如用户问：

> “阿司匹林100mg每天一次适用于什么情况？”

Dense Retrieval：

可能找到：

```
抗血小板治疗
心血管疾病
冠心病
```

BM25：

会重点匹配：

```
阿司匹林
100mg
每天一次
```

最终：

```
Dense
   +
BM25
   ↓
Hybrid
   ↓
Reranker
```

这就是医疗 RAG 很合理的设计。

------

# 七、Hybrid Retrieval 怎么实现

第一阶段：

```
Query
 │
 ├──────────────┐
 ↓              ↓
BGE-M3         BM25
 ↓              ↓
Dense Search   Sparse Search
 ↓              ↓
Top 30         Top 30
 └───────┬──────┘
         ↓
      Fusion
         ↓
      Top 20
```

Milvus 官方已经提供 Hybrid Search 和 RRF / Weighted Ranker。

我建议：

### MVP

使用：

```
RRFRanker
```

先把 Dense 和 BM25 的排名融合。

例如：

```
Dense:

A
B
C
D
E

BM25:

C
A
F
B
G
```

RRF：

```
A
C
B
F
D
```

然后：

```
Top 20
   ↓
Reranker
   ↓
Top 5
```

------

# 八、一定要增加 Reranker

这是你现在项目和真正企业级 RAG 差距非常大的一个地方。

不要：

```
Milvus
 ↓
Top5
 ↓
LLM
```

应该：

```
Milvus
 ↓
Top30
 ↓
Reranker
 ↓
Top5
 ↓
LLM
```

例如：

```
Query

“糖尿病患者使用二甲双胍有什么禁忌？”

Milvus Top30
       ↓
Reranker
       ↓
1. 二甲双胍禁忌症
2. 二甲双胍药品说明书
3. 肾功能不全用药指南
4. 糖尿病治疗指南
5. ...
```

这样 LLM 拿到的 Context 才真正有价值。

------

# 九、Reranker 不要和 Milvus 的 RRF 混淆

这里你面试非常容易被问。

实际上是两个层级：

```
Dense Search ──────┐
                    │
                    ↓
                 RRF融合
                    ↓
                 Top20~30
                    ↓
              Cross Encoder
                 Reranker
                    ↓
                  Top5
                    ↓
                   LLM
```

Milvus 的 RRF / Weighted Ranker 是**召回结果融合**；真正的 Reranker 则进一步对：

```
Query + Document
```

进行相关性判断。

所以你可以说：

> 第一阶段利用 Milvus Hybrid Search 完成多路召回和 RRF 融合，第二阶段使用 Reranker 对 Query-Document Pair 进行精排。

这句话非常适合面试。

# 十、文档解析必须重新设计

现在项目不能只依赖：

```
Neo4j
```

你要增加：

```
Document Ingestion Pipeline
```

支持：

```
PDF
DOCX
XLSX
TXT
Markdown
PNG/JPG
```

------

# 十一、PDF解析

不要直接全部 OCR。

应该：

```
PDF
 │
 ↓
PyMuPDF
 │
 ├── 有文本
 │      ↓
 │   直接解析
 │
 └── 无文本/扫描件
        ↓
      OCR
        ↓
    PaddleOCR
```

这样更符合工程实践。

------

# 十二、DOCX

需要解析：

```
标题
正文
表格
段落
章节
```

最终变成：

```
{
  "document_id": "doc_001",
  "title": "糖尿病诊疗指南",
  "section": "药物治疗",
  "content": "......",
  "page": 28
}
```

------

# 十三、Excel不能简单转TXT

例如：

| 药品     | 剂量  | 禁忌症     |
| -------- | ----- | ---------- |
| 二甲双胍 | 500mg | 肾功能异常 |
| 阿司匹林 | 100mg | 活动性出血 |

不要直接：

```
二甲双胍 500mg 肾功能异常
阿司匹林 100mg 活动性出血
```

应该保留结构：

```
药品：二甲双胍
剂量：500mg
禁忌症：肾功能异常
```

这样 RAG 检索质量会明显更稳定。

------

# 十四、Chunking 必须升级

不要简单：

```
text[i:i+500]
```

最终应该：

```
Document
 ↓
Title
 ↓
Chapter
 ↓
Section
 ↓
Paragraph
 ↓
Chunk
```

推荐初始参数：

```
Chunk Size:
400~700 tokens

Overlap:
50~100 tokens
```

但最终不能说：

> “我用了500 tokens，所以效果最好。”

应该：

> “初始使用500 tokens，并通过离线评估集对 Chunk Size 和 Overlap 进行消融实验。”

------

# 十五、Metadata一定要做好

每个 Chunk 都应该有：

```
{
    "document_id": "doc_001",
    "document_name": "糖尿病诊疗指南.pdf",
    "version": "v3",
    "page": 28,
    "chapter": "第三章",
    "section": "药物治疗",
    "chunk_id": "chunk_00028",
    "department": "医学",
    "category": "clinical_guideline",
    "content": "..."
}
```

因为后面的：

```
Citation
权限过滤
版本管理
删除
更新
检索过滤
```

全部依赖这些 metadata。

------

# 十六、一定增加版本管理

企业级知识库不能：

```
上传新文件
 ↓
覆盖旧文件
```

应该：

```
document_id
      │
      ├── v1
      ├── v2
      └── v3 ← Active
```

例如：

```
糖尿病诊疗指南

v1 旧版本
v2 旧版本
v3 当前有效
```

检索时：

```
status = active
```

这样用户不会问到旧政策。

# 十七、增加完整的知识库生命周期

最终：

```
Upload
   ↓
Parsing
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Indexing
   ↓
Active
   ↓
Update
   ↓
New Version
   ↓
Old Version Archive
```

同时支持：

```
删除
重新索引
版本切换
批量导入
增量更新
```

------

# 十八、RAG Pipeline 重新设计

最终核心类建议：

```
RAGPipeline
```

内部：

```
Query
 ↓
QueryNormalizer
 ↓
QueryRewrite
 ↓
HybridRetriever
 ↓
Reranker
 ↓
ContextBuilder
 ↓
PromptBuilder
 ↓
LLM
 ↓
CitationBuilder
 ↓
AnswerValidator
 ↓
Response
```

------

# 十九、增加 Query Rewrite

例如：

第一轮：

> “二甲双胍有哪些禁忌？”

第二轮：

> “那肾功能不全呢？”

第二句话本身信息不足。

Query Rewrite：

```
原始：
那肾功能不全呢？

改写：
二甲双胍对于肾功能不全患者有哪些使用禁忌？
```

再进入：

```
Hybrid Retrieval
```

这个能力会让你的 RAG 从单轮 QA 开始接近真正的知识库产品。

------

# 二十、增加 Context Builder

不要把 Milvus Top5 原封不动塞给 LLM。

应该：

```
Top20
 ↓
去重
 ↓
过滤低相关
 ↓
按照 Reranker Score 排序
 ↓
截取 Top5
 ↓
Token Budget
 ↓
Context
```

例如：

```
context = context_builder.build(
    documents=reranked_docs,
    max_tokens=6000
)
```

------

# 二十一、必须有 Citation

最终答案：

> 二甲双胍主要用于2型糖尿病患者的血糖控制。对于存在严重肾功能损害的患者，应根据肾功能情况谨慎使用或避免使用。

下面：

```
来源：

[1] 2型糖尿病诊疗指南.pdf
    第28页 / 药物治疗

[2] 二甲双胍药品说明书.pdf
    第6页 / 禁忌症
```

**Citation不能让LLM自己编。**

应该来自：

```
Milvus Metadata
```

即：

```
chunk
 ↓
document_id
page
section
source
 ↓
CitationBuilder
```

------

# 二十二、必须有 No-answer

这是企业 RAG 非常重要的能力。

用户问：

> “2026年某医院内部某项尚未发布政策是什么？”

知识库：

```
没有
```

不能让 LLM 硬回答。

应该：

```
Query
 ↓
Retrieval
 ↓
Reranker
 ↓
相关性检测
 ↓
低于阈值
 ↓
No Answer
```

返回：

> 当前知识库中没有检索到足够的信息，无法基于现有知识库可靠回答该问题。
>
> # 二十三、增加权限控制
>
> 如果你想真正叫“企业级”，这一块最好加入。
>
> 例如：
>
> ```
> 用户
>  ↓
> 部门
>  ↓
> 权限
>  ↓
> Milvus metadata filter
> ```
>
> 例如：
>
> ```
> access_level = medical
> ```
>
> 或者：
>
> ```
> department = cardiology
> ```
>
> 查询时：
>
> ```
> department == user's_department
> ```
>
> 这样：
>
> ```
> 普通员工
> ```
>
> 不能检索：
>
> ```
> 管理员内部文档
> ```
>
> ------
>
> # 二十四、增加 Evaluation，这是整个项目最重要的升级
>
> 你现在最缺的不是功能，而是：
>
> > **证明 RAG 好不好。**
>
> 建立：
>
> ```
> evaluation/
> ```
>
> 准备：
>
> ```
> 100~200条问题
> ```
>
> 类型：
>
> ```
> 事实型
> 流程型
> 条件型
> 多文档
> 相似问题
> 无答案问题
> 多轮问题
> ```
>
> ------
>
> # 二十五、检索指标
>
> 至少：
>
> ```
> Recall@5
> Recall@10
> MRR
> NDCG@5
> ```
>
> 例如：
>
> ```
> Dense Only
> 
> Recall@5 = 0.72
> 
> BM25 Only
> 
> Recall@5 = 0.65
> 
> Dense + BM25
> 
> Recall@5 = 0.81
> 
> Dense + BM25 + Reranker
> 
> Recall@5 = 0.86
> ```
>
> **注意：这些数字这里只是示例，不能写进你的简历作为实际结果，除非你真的跑出来。**
>
> ------
>
> # 二十六、一定做 Ablation
>
> 这是非常重要的面试材料。
>
> 实验：
>
> ```
> Experiment A
> Dense
> 
> Experiment B
> BM25
> 
> Experiment C
> Dense + BM25
> 
> Experiment D
> Dense + BM25 + Reranker
> 
> Experiment E
> Dense + BM25 + Reranker + Query Rewrite
> ```
>
> 然后：
>
> ```
> Recall@5
> MRR
> NDCG
> Faithfulness
> Answer Relevance
> Citation Accuracy
> ```
>
> 形成实验表。
>
> 这时候你就不是：
>
> > “我搭了一个 RAG。”
>
> 而是：
>
> > **“我负责 RAG 检索链路设计和效果优化，并通过消融实验验证各模块贡献。”**
>
> 这个差距非常大。
>
> ------
>
> # 二十七、生成质量也要评估
>
> 至少：
>
> ```
> Faithfulness
> Answer Relevance
> Answer Completeness
> Citation Accuracy
> No-answer Accuracy
> ```
>
> 最终：
>
> ```
> Retrieval Evaluation
>         +
> Generation Evaluation
> ```
>
> 构成完整 RAG Evaluation。
>
> ------
>
> # 二十八、增加 API 服务
>
> 不要继续以：
>
> ```
> Streamlit
> ```
>
> 作为核心产品。
>
> Streamlit 可以保留作为 Demo。
>
> 真正后端：
>
> ```
> FastAPI
> ```
>
> 接口：
>
> ```
> GET    /health
> 
> POST   /api/v1/documents/upload
> 
> GET    /api/v1/documents
> 
> GET    /api/v1/documents/{id}
> 
> DELETE /api/v1/documents/{id}
> 
> POST   /api/v1/documents/{id}/reindex
> 
> POST   /api/v1/retrieval/search
> 
> POST   /api/v1/chat
> 
> POST   /api/v1/chat/stream
> ```
>
> ------
>
> # 二十九、SSE
>
> 最终：
>
> ```
> Frontend
>    ↓
> POST /chat/stream
>    ↓
> FastAPI
>    ↓
> RAG
>    ↓
> LLM Streaming
>    ↓
> SSE
>    ↓
> Frontend
> ```
>
> 用户可以看到：
>
> ```
> 正在检索知识库...
> 
> 正在分析相关资料...
> 
> 根据《糖尿病诊疗指南》...
> ```
>
> 而不是等30秒一次性返回。
>
> # 三十、Docker化
>
> 最终：
>
> ```
> docker-compose.yml
> ```
>
> 至少：
>
> ```
> rag-api
> milvus
> etcd
> minio
> ```
>
> 如果使用 Milvus Standalone，按照官方部署方式组织即可。
>
> 最终：
>
> ```
>                     ┌──────────────┐
>                     │   Frontend   │
>                     └──────┬───────┘
>                            │
>                           SSE
>                            │
>                     ┌──────▼───────┐
>                     │   FastAPI    │
>                     └──────┬───────┘
>                            │
>              ┌─────────────┼─────────────┐
>              ↓             ↓             ↓
>           RAG Core       LLM         Knowledge
>              │                           │
>              ↓                           ↓
>          Retriever                     Milvus
> ```
>
> ------
>
> # 三十一、代码目录我建议直接改成这样
>
> ```
> rag--
> │
> ├── README.md
> ├── requirements.txt
> ├── .env.example
> ├── docker-compose.yml
> ├── Dockerfile
> │
> ├── configs/
> │   ├── model.yaml
> │   ├── milvus.yaml
> │   ├── retrieval.yaml
> │   └── rag.yaml
> │
> ├── data/
> │   ├── raw/
> │   ├── parsed/
> │   ├── chunks/
> │   └── evaluation/
> │
> ├── models/
> │   ├── bge-m3/
> │   └── reranker/
> │
> ├── src/
> │
> │   ├── document/
> │   │   ├── parser.py
> │   │   ├── pdf_parser.py
> │   │   ├── docx_parser.py
> │   │   ├── excel_parser.py
> │   │   └── ocr.py
> │   │
> │   ├── preprocessing/
> │   │   ├── cleaner.py
> │   │   ├── structure.py
> │   │   ├── chunker.py
> │   │   └── metadata.py
> │   │
> │   ├── embedding/
> │   │   └── bge_m3.py
> │   │
> │   ├── vectorstore/
> │   │   ├── milvus_client.py
> │   │   ├── collection.py
> │   │   ├── index.py
> │   │   └── filter.py
> │   │
> │   ├── retrieval/
> │   │   ├── dense.py
> │   │   ├── bm25.py
> │   │   ├── hybrid.py
> │   │   ├── fusion.py
> │   │   └── reranker.py
> │   │
> │   ├── query/
> │   │   ├── normalizer.py
> │   │   └── rewrite.py
> │   │
> │   ├── rag/
> │   │   ├── pipeline.py
> │   │   ├── context.py
> │   │   ├── prompt.py
> │   │   ├── generator.py
> │   │   ├── citation.py
> │   │   └── no_answer.py
> │   │
> │   ├── evaluation/
> │   │   ├── dataset.py
> │   │   ├── retrieval.py
> │   │   ├── generation.py
> │   │   ├── metrics.py
> │   │   └── ablation.py
> │   │
> │   ├── api/
> │   │   ├── main.py
> │   │   ├── routes.py
> │   │   ├── schemas.py
> │   │   └── dependencies.py
> │   │
> │   └── security/
> │       ├── auth.py
> │       ├── permission.py
> │       └── file_validation.py
> │
> ├── scripts/
> │   ├── ingest.py
> │   ├── build_index.py
> │   ├── update_document.py
> │   ├── evaluate_retrieval.py
> │   └── evaluate_rag.py
> │
> └── tests/
>     ├── test_parser.py
>     ├── test_chunker.py
>     ├── test_milvus.py
>     ├── test_retrieval.py
>     ├── test_reranker.py
>     ├── test_rag.py
>     └── test_api.py
> ```
>
> ------
>
> # 三十二、Neo4j 怎么办？
>
> **不要删。**
>
> 但我建议把它从 P2 的核心链路拿出来。
>
> 变成：
>
> ```
>                   Query
>                     │
>           ┌─────────┴─────────┐
>           ↓                   ↓
>       Hybrid RAG          Graph Retrieval
>           │                   │
>        Milvus               Neo4j
>           │                   │
>           └─────────┬─────────┘
>                     ↓
>                   Fusion
>                     ↓
>                   Rerank
>                     ↓
>                    LLM
> ```
>
> 但是：
>
> > **第一版企业级项目不要一上来就做 Graph + Vector + Agent。**
>
> 否则项目会变得非常乱。
>
> P2 的主线应该是：
>
> ```
> RAG
>  ↓
> Hybrid Retrieval
>  ↓
> Reranker
>  ↓
> Evaluation
> ```
>
> Neo4j 是后续增强。
>
> ------
>
> # 三十三、最终项目的技术栈
>
> 我建议最终固定成：
>
> | 层               | 技术                                 |
> | ---------------- | ------------------------------------ |
> | Language         | Python                               |
> | API              | FastAPI                              |
> | Document         | PyMuPDF / python-docx / openpyxl     |
> | OCR              | PaddleOCR                            |
> | Embedding        | BGE-M3                               |
> | Vector DB        | **Milvus**                           |
> | Sparse Retrieval | **BM25**                             |
> | Hybrid           | Milvus Hybrid Search                 |
> | Fusion           | RRF                                  |
> | Reranker         | BGE Reranker                         |
> | LLM              | Qwen / DeepSeek / MiniMax 等         |
> | Streaming        | SSE                                  |
> | Cache            | Redis                                |
> | Database         | PostgreSQL                           |
> | Container        | Docker                               |
> | Evaluation       | Recall@K / MRR / NDCG / Faithfulness |
> | UI               | React / Streamlit                    |
> | Optional Graph   | Neo4j                                |
>
> Milvus 官方目前已经支持 Dense + BM25 Hybrid Search，并提供 RRF、Weighted Ranker 等结果融合机制，因此这个技术组合是合理的。
>
> # 三十四、最重要：不要一口气乱改
>
> 我建议按照这个顺序开发。
>
> ## Phase 1：重构数据层
>
> ```
> 医疗文档
>  ↓
> Parser
>  ↓
> Cleaner
>  ↓
> Metadata
>  ↓
> Chunk
> ```
>
> 先把知识库做好。
>
> ------
>
> ## Phase 2：Milvus
>
> ```
> BGE-M3
>  ↓
> Dense
>  ↓
> Milvus
> ```
>
> 先完成纯 Dense Retrieval。
>
> 测试：
>
> ```
> Recall@5
> Recall@10
> ```
>
> ------
>
> ## Phase 3：BM25
>
> ```
> Document
>  ↓
> BM25
>  ↓
> Milvus Sparse
> ```
>
> 然后：
>
> ```
> Dense
> +
> BM25
> ```
>
> ------
>
> ## Phase 4：Hybrid
>
> ```
> Dense
>    +
> BM25
>    ↓
> RRF
>    ↓
> Top20
> ```
>
> ------
>
> ## Phase 5：Reranker
>
> ```
> Hybrid
>  ↓
> Top20~30
>  ↓
> Reranker
>  ↓
> Top5
> ```
>
> ------
>
> ## Phase 6：RAG
>
> ```
> Top5
>  ↓
> Context Builder
>  ↓
> Prompt
>  ↓
> LLM
> ```
>
> ------
>
> ## Phase 7：可靠性
>
> 加入：
>
> ```
> Citation
> No-answer
> Query Rewrite
> Multi-turn
> ```
>
> ------
>
> ## Phase 8：Evaluation
>
> 建立：
>
> ```
> 100~200 QA
> ```
>
> 然后：
>
> ```
> Dense
> ↓
> BM25
> ↓
> Hybrid
> ↓
> Hybrid + Reranker
> ↓
> Hybrid + Reranker + Rewrite
> ```
>
> 做完整对比。
>
> ------
>
> ## Phase 9：企业服务化
>
> ```
> FastAPI
> +
> SSE
> +
> PostgreSQL
> +
> Redis
> +
> Docker
> ```
>
> ------
>
> ## Phase 10：企业能力
>
> 最后增加：
>
> ```
> 权限
> 版本
> 增量更新
> 删除
> 重新索引
> 日志
> 监控
> 异常处理
> ```
>
> 这样项目才真正闭环。
>
> ------
>
> # 三十五、最终项目应该达到这个状态
>
> ```
>                          ┌──────────────┐
>                          │ 医疗知识文档 │
>                          └──────┬───────┘
>                                 ↓
>                     ┌────────────────────┐
>                     │ Document Ingestion │
>                     └─────────┬──────────┘
>                               ↓
>                     ┌──────────────────┐
>                     │ Parser / OCR     │
>                     └────────┬─────────┘
>                              ↓
>                     ┌──────────────────┐
>                     │ Cleaning / Chunk  │
>                     └────────┬─────────┘
>                              ↓
>                        ┌─────┴─────┐
>                        ↓           ↓
>                     BGE-M3       BM25
>                        ↓           ↓
>                    Dense Vec    Sparse Vec
>                        │           │
>                        └─────┬─────┘
>                              ↓
>                          ┌───────┐
>                          │Milvus │
>                          └───┬───┘
>                              ↓
>                        Hybrid Search
>                              ↓
>                            RRF
>                              ↓
>                           Top 30
>                              ↓
>                          Reranker
>                              ↓
>                            Top 5
>                              ↓
>                       Context Builder
>                              ↓
>                             LLM
>                        ┌─────┴─────┐
>                        ↓           ↓
>                     Answer      Citation
>                        │
>                        ↓
>                    No-answer
>                        │
>                        ↓
>                     FastAPI
>                        │
>                  ┌─────┴─────┐
>                  ↓           ↓
>               Web UI      Enterprise
> ```



## 最终审计结果

如果按照上面的方案真正实现并完成测试，P2 就不再是：

> “一个医疗知识图谱问答 Demo”

而可以定位为：

> **企业级医疗智能知识库与 Hybrid RAG 问答平台**

核心能力形成：

**Document AI → Knowledge Base → Dense Retrieval → BM25 → Hybrid Retrieval → Reranker → RAG → Citation → No-answer → Evaluation → API → Docker**

这条链已经足够支撑你面试 **RAG 算法工程师 / 大模型应用算法工程师 / LLM 应用开发工程师**。

另外，Milvus 官方当前推荐使用 `MilvusClient`