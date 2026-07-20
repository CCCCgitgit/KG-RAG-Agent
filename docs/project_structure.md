# KG-RAG Agent 项目结构说明

本文档规定 KG-RAG Agent 的目录结构、模块职责、依赖方向和扩展边界。专项算法、字段和协议细节分别由其他 `docs/` 文档维护。

## 1. 当前项目结构

```text
kg_rag_agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
│
├── configs/
│   ├── model.yaml
│   ├── graph.yaml
│   ├── retrieval.yaml
│   ├── kg.yaml
│   ├── prompt.yaml
│   ├── tools.yaml
│   ├── memory.yaml
│   ├── evaluation.yaml
│   ├── mcp.yaml
│   └── logging.yaml
│
├── src/kg_rag_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── app/
│   ├── services/
│   ├── agents/
│   ├── runtime/
│   ├── graph/
│   ├── entity_resolution/
│   ├── kg/
│   ├── retrieval/
│   ├── answering/
│   ├── llm/
│   ├── prompts/
│   ├── memory/
│   ├── tools/
│   ├── evaluation/
│   ├── data_pipeline/
│   └── utils/
│
├── data/
│   ├── demo/                              # 演示 Profile
│   └── production/                        # 正式 Profile
├── frontend/                              # Vue 3 / Vite / TypeScript
├── outputs/
├── logs/
├── scripts/
├── tests/
└── docs/
```

`frontend/` 已实现为独立 Vue 工程；`mcp/` 尚未创建，只有开始真实协议接入时才新增。

## 2. 分层与依赖方向

```text
入口层：frontend / app / scripts / future mcp
                         ↓
服务层：services
                         ↓
Agent 门面：agents
                         ↓
编排层：graph
                         ↓
领域能力：entity_resolution / kg / retrieval / answering / memory / llm / tools
                         ↓
运行时与数据：runtime / data / external services
```

基本规则：

1. 上层可以调用下层，下层不得反向依赖上层。
2. `app/`、CLI、Evaluation 和未来 MCP 必须复用 `AgentService`。
3. Node 之间只通过 `AgentState` 传递请求状态，不能互相直接调用。
4. Client、Store、Manager、Registry 和连接对象只放入 `RuntimeContext`。
5. 同一核心能力只能有一个正式实现位置；Tool 和 MCP 只能适配，不得复制算法。
6. 请求级覆盖必须采用白名单，不能修改凭据、路径、Memory Namespace 或系统权限。

## 3. 顶层目录职责

### `configs/`

保存稳定参数和功能开关。密钥只使用环境变量；完整 Prompt 正文放在 `src/kg_rag_agent/prompts/`；Graph 拓扑由 Python 定义。

### `data/`

保存原始数据、标准化产物、图谱和向量库。Demo 与 Production 的图谱、索引和向量库必须来自同一 Build Manifest。

### `outputs/` 与 `logs/`

保存运行时生成内容，包括评估、Memory、缓存和日志，不保存源码或固定配置。默认不提交 Git。

### `scripts/`

只做参数解析和正式模块调用。`run_cli.py`、`run_api.py`、`run_server.py` 都转发到统一 `kg_rag_agent.main`，不得维护第二套启动逻辑。

### `tests/`

按 `unit`、`integration`、`e2e` 和 Marker 组织。Memory、API、KG、Retrieval 等能力必须有对应测试。

### `docs/`

保存架构和开发规范。`project_structure.md` 是总纲，专项文档负责具体协议。

## 4. 源码模块职责

| 模块 | 职责 | 禁止事项 |
|---|---|---|
| `app/` | FastAPI、Schema、HTTP 错误、生命周期 | 直接实现 KG-RAG 算法 |
| `services/` | 外部请求校验、选项白名单、业务协调 | 实现 Node 或检索算法 |
| `agents/` | 构造 State、调用 Graph、标准化结果 | 持有第二套业务流程 |
| `runtime/` | 创建并复用 LLM、Graph、Vector、Prompt、Tool、Memory | 将运行时对象写入 State |
| `graph/` | State、Node、Edge、编译和流程控制 | 放置通用领域算法 |
| `entity_resolution/` | Mention、规范化、候选、链接、Grounding | 生成最终回答 |
| `kg/` | 图加载、关系/路径/邻居/子图查询、Evidence | 决定 HTTP 或工作流协议 |
| `retrieval/` | Embedding、向量召回、混合召回和重排 | 生成最终答案 |
| `answering/` | Evidence 选择、Reasoning、Citation、Composer | 绕过证据约束 |
| `llm/` | Provider Client、异常和 PromptManager | 决定业务路由 |
| `prompts/` | Prompt 正文模板 | 保存密钥或用户数据 |
| `memory/` | Memory 模型、策略、检索、写入、摘要和 Store | 直接暴露原始 Memory |
| `tools/` | ToolRegistry、Schema、权限和能力适配 | 重新实现 KG/检索算法 |
| `evaluation/` | 数据集、指标、报告和 Manifest | 维护第二套 Agent |
| `data_pipeline/` | 离线标准化、图谱/向量构建和 Manifest | 在线请求中重建产物 |
| `utils/` | 通用配置、路径、日志和异常辅助 | 承载业务流程 |

## 5. 正式调用链

```text
Frontend / CLI / API / Evaluation / Future MCP
                 ↓
            AgentService
                 ↓
             KGRAGAgent
                 ↓
         Compiled LangGraph
                 ↓
       Domain Components
                 ↓
          RuntimeContext
```

## 6. 命名与兼容规则

- 正式类名使用 `KGRAGAgent`；`KGRAgent` 只作为旧名称兼容别名。
- 旧 `config_overrides` 只在 Service 迁移层保留；新调用使用 `request_options` 或 API 的 `options`。
- 公共包入口采用懒加载，普通 `import kg_rag_agent` 不初始化 LLM、图谱或 Memory。
- 删除兼容接口前必须先完成调用点搜索、迁移测试和文档更新。

## 7. 后续新增模块

### `frontend/`

当前前端是独立 Vue 3 工程，通过 HTTP 调用 FastAPI，不导入 Python 领域模块。开发模式由 Vite 代理 `/api`，容器模式由 Nginx 代理到 Compose 中的 `api` 服务。

### `mcp/`

MCP 层只能连接 Service、ToolRegistry、PromptManager 和受控 Resource Provider。不得绕过权限系统或直接暴露 Runtime 对象。
