# KG-RAG Agent 文档总览

本目录是 KG-RAG Agent 的架构、模块边界和开发规范入口。文档已按当前代码同步：Runtime、Entity Resolution、Answering、ToolRegistry、Memory、FastAPI、CLI、Data Pipeline 和 Evaluation 已落地；MCP 与独立前端仍属于后续能力。

## 文档导航

| 文档 | 主要内容 | 当前状态 |
|---|---|---|
| [project_structure.md](project_structure.md) | 目录结构、职责和依赖方向 | 已实现结构总纲 |
| [architecture.md](architecture.md) | 系统分层、调用链和运行时边界 | 已实现架构 |
| [graph_design.md](graph_design.md) | AgentState、11 个节点和条件边 | 已接入 Memory |
| [memory_design.md](memory_design.md) | 会话/项目/用户 Memory、策略和 Store | 已实现基础版本 |
| [api.md](api.md) | FastAPI 请求、响应、流式接口和安全边界 | 已实现 |
| [prompt_design.md](prompt_design.md) | PromptManager、模板和 Memory 注入安全 | 已实现基础规范 |
| [tools.md](tools.md) | ToolRegistry、权限和内置工具 | 已实现基础版本 |
| [data_pipeline.md](data_pipeline.md) | Raw、Processed、KG、Vector 和 Manifest | 已实现 |
| [evaluation.md](evaluation.md) | 数据集、指标、报告和 Run Manifest | 已实现 |
| [deployment.md](deployment.md) | 本地、Docker 和生产部署边界 | 已实现基础部署 |
| [development.md](development.md) | docs-first、测试和提交规范 | 当前开发规范 |
| [mcp.md](mcp.md) | MCP Client/Server 接入设计 | 仅配置与规划 |

推荐阅读顺序：

```text
project_structure.md
        ↓
architecture.md
        ↓
graph_design.md / memory_design.md / api.md
        ↓
当前开发模块的专项文档
        ↓
development.md
```

## 当前能力边界

已具备：

```text
CLI 与 FastAPI
AgentService 与 KGRAGAgent
LangGraph KG-RAG 主流程
实体解析、KG 检索、混合检索
Evidence、Reasoning、Citation 和 Answer Composer
RuntimeContext
ToolRegistry
会话、项目和用户隔离 Memory
离线数据构建与评估
Docker 基础部署
```

尚未具备：

```text
独立 Web 前端
生产级身份认证、限流和审计
分布式 Memory / Vector / Graph 服务
完整 MCP Client、MCP Server 和协议适配层
```


当前项目还包含独立的 `frontend/` 轻量 Vue 3 界面；部署方式见 [deployment.md](deployment.md)。
