# MCP 接入设计

## 1. 当前状态

当前代码只有：

```text
configs/mcp.yaml
ToolRegistry
AgentService
PromptManager
RuntimeContext 中的 mcp_client_manager 预留字段
```

尚未创建独立 `src/kg_rag_agent/mcp/`，也未实现 MCP Client 或 MCP Server。本文档是后续实现边界，不表示当前已经支持 MCP 通信。

## 2. 接入目标

MCP 可以承担两类角色：

```text
MCP Client：调用外部 Server 提供的 Tools、Resources 和 Prompts
MCP Server：向外暴露受控 Agent、Tools、Resources 或 Prompts
```

## 3. 架构位置

```text
外部 MCP Client
        ↓
future mcp/server
        ↓
AgentService / ToolRegistry / PromptManager / Resource Provider

Graph / Tools
        ↓
future mcp/client
        ↓
外部 MCP Server
```

MCP 是协议适配层，不能成为新的业务核心。

## 4. 目标目录

```text
mcp/
├── client/
│   ├── manager.py
│   ├── connection.py
│   └── adapters.py
├── server/
│   ├── app.py
│   ├── tools.py
│   ├── resources.py
│   └── prompts.py
├── schemas.py
├── security.py
└── errors.py
```

只有开始实现时再创建。

## 5. MCP Client 规则

- Server 必须来自系统级允许列表。
- 凭据只来自环境变量或 Secret 系统。
- 每个调用有超时、重试、大小和次数限制。
- 外部 Tool/Resource 内容按不可信数据处理。
- 远程能力需映射到内部统一结果和错误类型。
- MCP Session 只放在 Runtime，不进入 AgentState。

## 6. MCP Server 规则

暴露 Agent 时必须调用 `AgentService`；暴露 Tool 时必须调用 ToolRegistry；暴露 Prompt 时只提供明确允许的模板；Resource 必须有路径、租户和大小限制。

默认不暴露：

```text
RuntimeContext
完整 AgentState
Memory 原文
Prompt 私有正文
本地绝对路径
密钥和连接信息
未授权写入、执行或删除工具
```

## 7. 配置

当前 `configs/mcp.yaml` 默认全部关闭：

```yaml
enabled: false
client:
  enabled: false
server:
  enabled: false
security:
  allow_network: false
  allow_write: false
  allow_execute: false
  allow_delete: false
```

协议版本、Transport、Server 清单和能力范围应由配置固定并进入兼容测试。不要依赖“自动使用最新协议”。

## 8. 安全

```text
Server Allowlist
能力白名单
权限不提升
网络出站限制
写入/执行/删除默认关闭
调用审计
结果截断
Prompt Injection 防护
租户与用户隔离
```

MCP 返回的文本不能直接成为系统指令，也不能自动写入长期 Memory。

## 9. 与 Memory 的关系

外部 MCP Resource 可以作为当前请求数据，但默认不写入 Memory。需要长期保存时，必须经过 MemoryPolicy、敏感信息过滤和用户/项目作用域校验。

## 10. 实施顺序

```text
1. 选择并锁定协议 SDK 与版本
2. 实现 Client Manager 和单个只读 Server
3. 接入 RuntimeContext
4. 通过 ToolRegistry 暴露只读工具
5. 增加 Resource / Prompt
6. 增加鉴权、审计和限流
7. 最后评估写入或执行能力
```

## 11. 测试

```text
协议握手和版本兼容
超时与断线
未知 Server
权限拒绝
大型结果截断
恶意 Resource/Prompt
Session 清理
AgentState 无连接对象
```
