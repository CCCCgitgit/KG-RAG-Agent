# KG-RAG Agent 系统架构

本文档描述当前已经落地的系统架构以及尚未实现的扩展边界。

## 1. 系统目标

KG-RAG Agent 面向结构化知识问答，重点保证：

```text
实体可定位
证据可追溯
回答可引用
流程可测试
依赖可替换
Memory 可隔离
入口可复用
```

## 2. 当前架构总览

```text
用户 / Vue 前端 / CLI / FastAPI
          ↓
      AgentService
          ↓
       KGRAGAgent
          ↓
      LangGraph
          ↓
┌────────────────────────────────────────────┐
│ entity_resolution / kg / retrieval /       │
│ answering / memory / llm / tools           │
└────────────────────────────────────────────┘
          ↓
      RuntimeContext
          ↓
LLM / Graph / Vector Store / JSON Memory Store
```

Runtime、Entity Resolution、Answering、Prompt 模板、ToolRegistry、Memory 和轻量 Vue 前端均已落地。MCP 尚未实现。

## 3. 入口层

### CLI

统一入口：

```bash
kg-rag-agent -q "问题"
python -m kg_rag_agent --interactive
```

兼容脚本只转发到 `kg_rag_agent.main`。

### FastAPI

`create_app()` 负责应用装配；Lifespan 负责 Runtime、Agent 和 Service 的创建与释放。API 路由只依赖 `AgentService`。

### Evaluation

评估通过 `EvaluationService` 或 Runner 复用正式 AgentService，不维护简化版 Agent。

## 4. 服务与 Agent

`AgentService` 负责：

```text
问题非空校验
请求选项白名单
user/project/session 标识透传
单轮、批量和流式调用
服务状态与关闭
```

`KGRAGAgent` 负责：

```text
构造初始 AgentState
生成 request_id / session_id
调用或流式执行 CompiledGraph
将最终 State 转换为 AgentResult
```

## 5. RuntimeContext

RuntimeContext 是应用级依赖容器，当前可以保存：

```text
RuntimeSettings
LLMClient
EmbeddingClient
GraphLoader
EntityVectorStore
VectorRetriever / HybridRetriever / Reranker
EntityLinker
PromptManager
ToolRegistry
MemoryManager
扩展依赖
```

规则：

- Runtime 由入口或 Agent 创建一次并复用。
- Runtime 不进入 AgentState。
- Runtime 关闭后不能继续使用。
- 测试可以注入 Fake Runtime 或单项依赖。

## 6. LangGraph 主链

```text
START → memory_load → query_router

query_router ─ direct_llm → direct_llm ┐
             ├ kg_rag → mention_extraction → entity_linking
             │          → entity_grounding → kg_retrieval
             │          → semantic_scoring → reasoning → generation
             └ clarify/error → generation

直接回答或生成完成 → memory_write → END
```

Memory 关闭时两个节点安全降级为空更新，不改变原有回答路径。

## 7. 领域模块

### Entity Resolution

将 Mention 抽取、规范化、候选生成、链接和 Grounding 分离，避免将所有逻辑塞入 Node。

### KG 与 Retrieval

`kg/` 提供关系、路径、邻居和子图查询；`retrieval/` 提供向量与混合检索。二者输出标准化结果，再由 Node 构造 Evidence。

### Answering

`answering/` 统一管理 Evidence 选择、Reasoning、Citation 和最终 Composer。Memory 不能替代 Evidence 或创建 Citation。

### Memory

Memory 提供会话缓冲、摘要、长期记录、策略、检索和 Store，并按 `user_id + project_id + session_id` 隔离。

### Tools

ToolRegistry 负责注册、Schema、权限、超时、调用次数、结果截断和统一错误。工具只适配已有能力。

## 8. 状态与运行时边界

AgentState 保存可序列化的请求数据和中间结果，例如：

```text
query / route / mentions / grounded_entities
evidence / reasoning / final_answer
memory_context / memory_text / memory_write_result
traces / warnings / errors
```

AgentState 不保存：

```text
RuntimeContext
LLMClient
GraphLoader
Vector Store
MemoryManager
ToolRegistry
文件句柄或网络连接
```

## 9. 安全边界

- API 默认禁止返回 `raw_state`。
- Memory 正文不进入标准 API 响应。
- Prompt 中的 Memory、工具结果和外部内容均按不可信数据处理。
- Tool 权限不能由普通请求或 Prompt 提升。
- 密钥只从环境变量或 Secret 系统读取。
- 日志和 Trace 不记录密钥、原始 Memory 或完整内部对象。

## 10. 当前未实现能力

```text
身份认证与租户授权
分布式 Runtime 和持久化服务
生产级审计、限流和监控
MCP Client / MCP Server
```
