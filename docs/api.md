# KG-RAG Agent API 文档

FastAPI 入口位于 `src/kg_rag_agent/app/`。API 只负责协议、校验、依赖注入和错误转换，业务调用统一进入 `AgentService`。

## 1. 启动

```bash
python -m kg_rag_agent --serve --host 127.0.0.1 --port 8000
```

默认：

```text
Swagger UI  /docs
ReDoc       /redoc
OpenAPI     /openapi.json
```

当前没有独立 Web 前端；这些页面仅用于接口调试。

## 2. 路由

| Method | Path | 作用 |
|---|---|---|
| GET | `/api/health` | 进程存活状态 |
| GET | `/api/ready` | Runtime 与 AgentService 就绪状态 |
| GET | `/api/status` | 简单 API 状态 |
| POST | `/api/chat` | 单轮问答 |
| POST | `/api/chat/batch` | 串行批量问答 |
| POST | `/api/chat/stream` | NDJSON LangGraph 事件流 |
| GET | `/api/chat/info` | 非敏感 Service 摘要 |

API 前缀可由 `KG_RAG_API_PREFIX` 修改。

## 3. 单轮请求

```json
{
  "query": "UNITED STATES 与 CHINA 有什么关系？",
  "user_id": "user_001",
  "project_id": "project_001",
  "session_id": "session_001",
  "request_id": "request_001",
  "messages": [],
  "chat_history": [],
  "metadata": {},
  "options": {
    "retrieval_top_k": 8,
    "path_max_depth": 4,
    "temperature": 0.2,
    "max_tokens": 1200,
    "language": "zh",
    "include_citations": true,
    "allowed_tools": ["kg.relation_search"]
  },
  "include_raw_state": false,
  "include_memory_status": true
}
```

重要字段：

| 字段 | 说明 |
|---|---|
| `user_id` | 用户长期 Memory 隔离 |
| `project_id` | 项目 Memory 隔离 |
| `session_id` | 会话标识；短期 Memory 使用 user/project/session 组合隔离，缺省由服务生成 |
| `options` | 已审核的请求级白名单 |
| `include_raw_state` | 默认禁止，需服务端显式启用 |
| `include_memory_status` | 只返回非敏感 Memory 状态 |

禁止外部设置：API Key、Provider URL、图谱/向量/Memory 路径、Memory Namespace、系统权限和 Prompt 文件路径。

## 4. 单轮响应

```json
{
  "answer": "……",
  "request_id": "request_001",
  "session_id": "session_001",
  "user_id": "user_001",
  "project_id": "project_001",
  "route": "kg_rag",
  "answerability": "answerable",
  "semantic_score": 0.86,
  "citations": [],
  "traces": [],
  "warnings": [],
  "has_error": false,
  "error_message": "",
  "memory_status": {
    "loaded": true,
    "written": true,
    "recent_message_count": 2,
    "retrieved_memory_count": 1,
    "estimated_tokens": 128,
    "summary_used": false,
    "written_count": 1,
    "skipped_count": 0
  },
  "raw_state": null
}
```

Memory 正文不会通过 `memory_status` 返回。

## 5. 批量接口

```json
{
  "queries": ["问题一", "问题二"],
  "user_id": "user_001",
  "project_id": "project_001",
  "session_id": "session_001",
  "metadata": {},
  "options": {},
  "include_raw_state": false,
  "include_memory_status": false
}
```

当前批量接口串行调用。最大数量由 `KG_RAG_API_MAX_BATCH_SIZE` 控制，默认 20，允许范围 1–100。

## 6. 流式接口

`POST /api/chat/stream` 返回：

```text
Content-Type: application/x-ndjson
```

每行是一个 JSON 事件。API 会过滤 Runtime、Store、Manager、原始 Memory 和其他不可公开字段。前端应按行增量解析，不应假设所有节点事件结构完全相同。

## 7. Health、Ready 与 Status

- `/health`：只表示 Web 进程存活。
- `/ready`：检查应用依赖是否已经创建并可用；未初始化或启动失败时返回 503。
- `/status`：简单运行提示，不替代完整监控。

## 8. Raw State

服务端默认：

```dotenv
KG_RAG_API_ALLOW_RAW_STATE=false
```

即使客户端请求 `include_raw_state=true`，服务端未开启时也会拒绝。生产环境通常应保持关闭。

## 9. CORS

主要环境变量：

```text
KG_RAG_API_CORS_ORIGINS
KG_RAG_API_CORS_ALLOW_CREDENTIALS
KG_RAG_API_CORS_ALLOW_METHODS
KG_RAG_API_CORS_ALLOW_HEADERS
```

不能将通配 Origin `*` 与 Credentials 同时启用。

## 10. 错误响应

```json
{
  "error_code": "invalid_agent_request",
  "message": "……",
  "request_id": "req_xxx",
  "details": {}
}
```

不得返回密钥、绝对路径、Prompt 正文、Memory 正文或内部堆栈。服务端完整异常只进入受控日志。

## 11. 生产缺口

当前 API 尚未内置：

```text
身份认证
租户授权
限流和配额
前端会话管理
生产级审计与指标系统
```
