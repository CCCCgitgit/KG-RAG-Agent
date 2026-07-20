# KG-RAG Agent

KG-RAG Agent 是一个基于 **LangGraph** 的知识图谱增强问答系统。项目采用 `src-layout` 和分层架构，将 Web API、Service、Agent、Graph 编排、实体解析、知识图谱检索、向量检索、答案生成、Memory、工具和运行时依赖分开管理。

当前项目适合作为以下场景的工程基础：

- 实体检索、关系查询、邻居扩展和多跳路径推理；
- 知识图谱与向量召回结合的混合检索；
- 基于 Evidence、Reasoning 和 Citation 的可解释回答；
- 支持会话、项目和用户隔离的受控 Memory；
- FastAPI、CLI、批处理和评估等多入口复用同一套 Agent 流程；
- 后续接入 MCP Client、MCP Server 和外部工具服务。

---

## 1. 核心能力

### 1.1 LangGraph 工作流

系统通过统一 `AgentState` 驱动节点执行，Node 之间不直接调用，所有状态通过 Graph 传递。

```text
START
  ↓
memory_load
  ↓
query_router
  ├── direct_llm
  │      ↓
  │   direct_llm
  │      ↓
  │   memory_write
  │      ↓
  │     END
  │
  ├── clarify
  │      ↓
  │   generation
  │      ↓
  │   memory_write
  │      ↓
  │     END
  │
  └── kg_rag
         ↓
      mention_extraction
         ↓
      entity_linking
         ↓
      entity_grounding
         ↓
      kg_retrieval
         ↓
      semantic_scoring
         ↓
      reasoning
         ↓
      generation
         ↓
      memory_write
         ↓
        END
```

当 Memory 未启用时，`memory_load` 和 `memory_write` 会保持安全降级，不影响原有 KG-RAG 流程。

### 1.2 实体解析与知识图谱检索

项目已将实体处理拆分为独立模块：

```text
Mention Extraction
    ↓
Normalization
    ↓
Candidate Generation
    ↓
Entity Linking
    ↓
Grounding Verification
```

知识图谱检索支持：

- 关系查询；
- 多跳路径查询；
- 邻居查询；
- 可选子图查询；
- Evidence 构建与去重；
- 基于分数和语义相关性的结果筛选。

### 1.3 混合检索与回答生成

检索层支持实体召回、向量召回、关键词召回和重排。回答层统一处理：

- Evidence 选择；
- 推理步骤组织；
- Claim 与 Citation 对齐；
- 不可回答与不确定状态；
- 最终答案组合。

### 1.4 受控 Memory

Memory 模块支持：

- 会话近期消息；
- 会话摘要；
- 用户长期 Memory；
- `user_id`、`project_id`、`session_id` 隔离；
- Token Budget 和最大消息数限制；
- Memory 状态摘要；
- 内存存储与 JSON 持久化存储；
- 写入、遗忘和 supersede 操作。

Memory 默认关闭，启用前需要明确配置读写策略、命名空间和持久化方式。

### 1.5 多入口复用

所有外部入口最终都通过 `AgentService → KGRAGAgent → LangGraph` 调用，不维护第二套业务流程：

```text
CLI / FastAPI / Evaluation / Future MCP
                  ↓
             AgentService
                  ↓
              KGRAGAgent
                  ↓
        LangGraph CompiledGraph
                  ↓
       Domain Components + Runtime
```

---

## 2. 项目结构

```text
kg_rag_agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
│
├── configs/                 # 稳定配置与功能开关
├── data/
│   ├── demo/                # 可直接运行的演示 Profile
│   └── production/          # 正式数据 Profile
├── frontend/                # Vue 3 轻量聊天界面
├── docs/                    # 架构与模块专项文档
├── outputs/                 # 日志、响应、Trace、Memory 和评估结果
├── scripts/                 # 轻量启动、构建和评估脚本
├── tests/                   # 单元、集成和端到端测试
│
└── src/
    └── kg_rag_agent/
        ├── app/             # FastAPI、请求响应 Schema、异常边界
        ├── services/        # 外部请求协调与统一服务入口
        ├── agents/          # Agent 门面与标准请求结果
        ├── runtime/         # 共享依赖创建和生命周期管理
        ├── graph/           # State、Node、Edge 和 Graph 编排
        ├── entity_resolution/
        ├── kg/
        ├── retrieval/
        ├── answering/
        ├── llm/
        ├── prompts/
        ├── memory/
        ├── tools/
        ├── evaluation/
        ├── data_pipeline/
        └── utils/
```

模块依赖原则：

```text
入口层 → Service → Agent → Graph → 领域能力 → Runtime / External Services
```

下层模块不得反向依赖上层入口；`app/` 不得直接拼装 KG、Retrieval、LLM 或 Graph Node。

完整结构约束见 [项目结构说明](docs/project_structure.md)。

---

## 3. 环境要求

- Python 3.10–3.13；
- 推荐使用独立虚拟环境；
- 使用真实 LLM 时需要对应 Provider 的 API Key；
- 使用向量检索时需要安装 Retrieval 可选依赖；
- 默认模型 Provider 为 DeepSeek。

创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

---

## 4. 安装

### 4.1 完整运行环境

`requirements.txt` 会基于 `pyproject.toml` 安装项目及全部运行依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.2 按功能安装

仅安装核心依赖：

```bash
python -m pip install -e .
```

安装 API 与检索能力：

```bash
python -m pip install -e ".[api,retrieval]"
```

安装开发与测试依赖：

```bash
python -m pip install -e ".[dev,test]"
```

---

## 5. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

默认使用 DeepSeek：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-chat
```

也可以切换为 OpenAI：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

配置边界：

- API Key、Token 和密码只放在 `.env` 或部署平台的 Secret 中；
- 模型、检索、图谱、工具和 Memory 参数放在 `configs/`；
- Prompt 正文放在 `src/kg_rag_agent/prompts/`；
- Graph 拓扑由 Python 代码定义；
- 每次请求只能覆盖经过白名单审核的参数；
- 外部请求不能设置 Memory Namespace、Store 路径或系统权限。

主要配置文件：

| 文件 | 作用 |
|---|---|
| `configs/model.yaml` | LLM Provider、模型名称、超时与重试 |
| `configs/graph.yaml` | 节点参数和流程开关 |
| `configs/kg.yaml` | 图谱路径、查询方式和 Evidence 参数 |
| `configs/retrieval.yaml` | Embedding、向量库、混合召回与重排 |
| `configs/prompt.yaml` | Prompt 名称、路径和加载策略 |
| `configs/tools.yaml` | ToolRegistry、权限和工具参数 |
| `configs/memory.yaml` | Memory 隔离、预算、读写和持久化 |
| `configs/evaluation.yaml` | 评估数据与输出参数 |
| `configs/mcp.yaml` | MCP 预留配置 |
| `configs/logging.yaml` | 日志级别与输出设置 |

---

## 6. 运行 CLI

安装后可直接使用命令：

```bash
kg-rag-agent -q "UNITED STATES 与 CHINA 之间有什么关系？"
```

也可以使用模块入口：

```bash
python -m kg_rag_agent -q "UNITED STATES 与 CHINA 之间有什么关系？"
```

交互式多轮会话：

```bash
python -m kg_rag_agent \
  --interactive \
  --user-id user_001 \
  --project-id demo_project \
  --session-id session_001 \
  --memory-status
```

Windows PowerShell 可写为一行：

```powershell
python -m kg_rag_agent --interactive --user-id user_001 --project-id demo_project --session-id session_001 --memory-status
```

JSON 输出：

```bash
python -m kg_rag_agent \
  -q "查询两个实体之间的路径" \
  --json \
  --include-identifiers \
  --memory-status
```

常用请求级参数：

```text
--retrieval-top-k
--path-max-depth
--temperature
--max-tokens
--language
--allowed-tool
--no-citations
```

查看全部参数：

```bash
python -m kg_rag_agent --help
```

---

## 7. 启动 FastAPI

启动服务：

```bash
python -m kg_rag_agent --serve --host 127.0.0.1 --port 8000
```

开发模式：

```bash
python -m kg_rag_agent --serve --reload
```

多进程部署：

```bash
python -m kg_rag_agent --serve --host 0.0.0.0 --port 8000 --workers 4
```

`--reload` 不能与多个 Worker 同时使用。

默认地址：

```text
Swagger UI: http://127.0.0.1:8000/docs
ReDoc:      http://127.0.0.1:8000/redoc
OpenAPI:    http://127.0.0.1:8000/openapi.json
```

主要接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 单次标准问答 |
| `POST` | `/api/chat/batch` | 同一隔离边界内的批量问答 |
| `POST` | `/api/chat/stream` | NDJSON LangGraph 事件流 |
| `GET` | `/api/chat/info` | AgentService 非敏感摘要 |
| `GET` | `/api/health` | 进程健康状态 |
| `GET` | `/api/ready` | Runtime 和 Service 就绪状态 |
| `GET` | `/api/status` | API 基础状态 |

单次问答示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "UNITED STATES 与 CHINA 之间有什么关系？",
    "user_id": "user_001",
    "project_id": "demo_project",
    "session_id": "session_001",
    "include_memory_status": true,
    "options": {
      "retrieval_top_k": 8,
      "path_max_depth": 4,
      "include_citations": true
    }
  }'
```

流式接口返回 `application/x-ndjson`。事件中的 Runtime 对象、Memory 正文和敏感字段会被过滤。

默认禁止 API 返回完整 `raw_state`。只有在可信环境中显式设置以下变量后，API 才允许请求该字段：

```dotenv
KG_RAG_API_ALLOW_RAW_STATE=true
```

---


## 8. 前端

轻量前端位于 `frontend/`，使用 Vue 3、Vite 和 TypeScript。先启动后端，再运行：

```bash
cd frontend
npm ci
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。开发服务器会把 `/api` 代理到 `http://127.0.0.1:8000`。

也可以一次启动前后端：

```bash
docker compose up --build -d
```

此时前端默认地址为 `http://127.0.0.1:5173`，后端 Swagger 为 `http://127.0.0.1:8000/docs`。

---

## 9. 启用 Memory

Memory 默认配置为关闭：

```yaml
enabled: false
write_enabled: false
long_term_enabled: false
```

启用受控会话与长期 Memory 时，可在 `configs/memory.yaml` 中配置：

```yaml
enabled: true
namespace_prefix: kg_rag_agent

max_messages: 20
max_summary_tokens: 1200
max_retrieved_items: 8
max_context_tokens: 4000

write_enabled: true
long_term_enabled: true

store_type: json
store_path: outputs/memory/memories.json
```

同时可以通过环境变量覆盖总开关：

```dotenv
MEMORY_ENABLED=true
```

使用 Memory 时建议始终传入：

```text
user_id
project_id
session_id
```

三者的职责：

- `user_id`：隔离不同用户的长期信息；
- `project_id`：隔离不同业务项目或知识空间；
- `session_id`：维持一次连续对话的近期消息和摘要。

Memory 正文不会通过标准 API 响应直接返回。`include_memory_status=true` 只返回读取数量、写入数量、Token 估计等非敏感状态。

详细设计见 [Memory 设计](docs/memory_design.md)。

---

## 10. 数据与离线构建

项目的数据处理流程负责生成：

```text
原始实体与关系
    ↓
标准化实体、关系和三元组
    ↓
NetworkX MultiDiGraph
    ↓
实体索引与 Alias
    ↓
Embedding 与 Chroma 向量库
    ↓
构建清单与统计信息
```

一键构建：

```bash
python scripts/build_all.py --profile demo
```

本地没有 Embedding 模型时，可以允许下载：

```bash
python scripts/build_all.py --profile demo --allow-download
```

只构建图谱、不构建向量库：

```bash
python scripts/build_all.py --profile demo --skip-vector-store
```

数据和构建产物必须来自同一批次，避免图谱、实体索引和向量库交叉混用。

详细说明见 [数据流水线](docs/data_pipeline.md)。

---

## 11. 评估

运行示例评估：

```bash
python scripts/evaluate.py \
  --input data/demo/examples/demo_questions.json \
  --output-dir outputs/evaluation
```

快速限制样例数量：

```bash
python scripts/evaluate.py \
  --input data/demo/examples/demo_questions.json \
  --limit 10
```

评估层会复用正式 `AgentService`，并输出汇总指标和逐样例结果。

详细说明见 [评估设计](docs/evaluation.md)。

---

## 12. 测试与代码质量

运行全部测试：

```bash
pytest
```

按标记运行：

```bash
pytest -m memory
pytest -m api
pytest -m retrieval
pytest -m integration
```

覆盖率：

```bash
pytest --cov=kg_rag_agent --cov-report=term-missing
```

代码检查：

```bash
ruff check src tests
ruff format --check src tests
mypy -p kg_rag_agent
```

构建 Wheel：

```bash
python -m build
```

---

## 13. 开发约束

新增功能时应遵循以下规则：

1. Node 只完成一个原子步骤，不直接调用其他 Node；
2. Node 之间只通过 `AgentState` 传递请求状态；
3. Runtime、LLM、Graph Store、Vector Store、Memory 和 ToolRegistry 不得放入 `AgentState`；
4. API、CLI、Evaluation 和 MCP 不得各自实现一套 KG-RAG 流程；
5. `kg/`、`retrieval/`、`llm/` 和 `memory/` 不得依赖 `app/` 或 `services/`；
6. Prompt 正文不得散落在 Node 文件中；
7. 外部请求只能覆盖白名单参数；
8. 日志、Trace 和 API 响应不得泄露密钥、Memory 正文和内部连接对象；
9. 大体积图谱、向量库、模型文件和运行时 Memory 不提交到 Git；
10. 任何新协议适配都应复用现有 Service、Tools 和 Runtime。

---

## 14. 文档导航

- [文档总览](docs/README.md)
- [系统架构](docs/architecture.md)
- [项目结构](docs/project_structure.md)
- [Graph 设计](docs/graph_design.md)
- [API 说明](docs/api.md)
- [Memory 设计](docs/memory_design.md)
- [Prompt 设计](docs/prompt_design.md)
- [Tools 设计](docs/tools.md)
- [MCP 规划](docs/mcp.md)
- [数据流水线](docs/data_pipeline.md)
- [评估设计](docs/evaluation.md)
- [开发规范](docs/development.md)
- [部署说明](docs/deployment.md)

---

## 15. 当前状态

当前已完成：

- 分层项目结构；
- Runtime 依赖容器；
- LangGraph KG-RAG 主流程；
- 实体解析、KG 查询、混合检索和回答生成；
- ToolRegistry 与工具权限；
- Memory 模块及 Graph/API/CLI 接入；
- FastAPI、CLI、批量、流式和评估入口；
- Vue 3 轻量前端及前后端 Docker Compose；
- 数据构建和测试框架。

当前默认关闭或仍需继续完善：

- Memory 的生产级持久化、鉴权和管理接口；
- MCP Client、MCP Server 与协议适配；
- 面向真实生产环境的身份认证、限流、监控和分布式部署；
- 针对正式数据集的系统化评测与性能基线。
