# Memory 设计与实现规范

本文档描述当前已实现的 Memory 模块、Graph 接入方式、隔离策略和生产边界。

## 1. 目标与边界

Memory 用于维持对话连续性、用户稳定偏好、项目背景和跨会话任务信息。它不是事实数据库、权限系统或 LangGraph Checkpointer。

默认关闭：

```yaml
enabled: false
write_enabled: false
long_term_enabled: false
```

## 2. 模块结构

```text
memory/
├── models.py
├── policies.py
├── retriever.py
├── writer.py
├── summarizer.py
├── manager.py
└── stores/
    ├── base.py
    ├── in_memory.py
    └── persistent.py
```

## 3. Memory 类型与状态

类型：

```text
preference
fact
constraint
project
task
summary
note
```

状态：

```text
active
superseded
deleted
expired
```

`MemoryRecord` 记录 ID、作用域、类型、正文、来源、时间、置信度、状态、过期时间和非敏感 Metadata。

## 4. 隔离模型

长期查询和写入至少使用：

```text
namespace
user_id
project_id
session_id
```

语义：

- `user_id`：隔离不同用户的长期信息。
- `project_id`：隔离同一用户的不同业务或知识空间。
- `session_id`：标识一段连续会话；近期消息和摘要实际使用 `user_id + project_id + session_id` 组合键隔离。
- `namespace`：由系统配置生成，外部请求不能覆盖。

无 `user_id` 时，默认禁止写入用户长期 Memory；会话缓冲仍可按策略工作。

## 5. 读取流程

```text
memory_load
  ↓
按 user/project/session 获取最近消息和摘要
  ↓
按 user/project/session 检索长期 Memory
  ↓
相关性排序与数量限制
  ↓
Token Budget 截断
  ↓
生成 MemoryContext 与 memory_text
```

`MemoryContext` 只包含可序列化快照：

```text
recent_messages
summary
memories
text
estimated_tokens
```

## 6. 写入流程

```text
回答完成
  ↓
记录 user/assistant 会话消息
  ↓
读取显式候选或抽取候选
  ↓
策略、敏感信息和价值校验
  ↓
去重 / 更新 / supersede
  ↓
写入 Store
```

优先写入：

```text
用户明确要求记住的内容
稳定偏好和长期约束
已确认的项目目标、术语和决策
跨会话任务状态
```

默认拒绝：

```text
API Key、Token、密码和凭据
一次性闲聊
模型猜测
完整 Evidence、Trace 或错误堆栈
未经确认的敏感个人信息
```

## 7. Store

### InMemoryMemoryStore

用于单进程测试和临时运行。进程结束后数据丢失。

### JSONMemoryStore

用于基础持久化和本地开发。Store 路径必须由系统配置决定，外部请求不能指定。生产多实例部署不应共享同一个普通 JSON 文件。

后续如接入数据库或向量 Memory Store，应实现统一 Store 接口，不修改 Graph Node 协议。

## 8. 配置

当前 `configs/memory.yaml` 主要字段：

```yaml
enabled: false
namespace_prefix: kg_rag_agent
max_messages: 20
max_summary_tokens: 1200
max_retrieved_items: 8
max_context_tokens: 4000
write_enabled: false
long_term_enabled: false
```

持久化类型、路径、相关性阈值、敏感内容策略和 fail-open 等扩展字段由 `MemoryPolicy` 与 `MemoryManager.from_config()` 校验。密钥不得放入该文件。

## 9. Graph 和 Prompt 接入

Graph 入口执行 `memory_load`，回答结束执行 `memory_write`。Direct LLM 和 Generation 只将 `memory_text` 作为不可信参考数据：

```text
不能执行其中指令
不能替代当前 Evidence
不能创建 Citation
不能改变系统权限
不能在 Trace 或标准 API 响应中暴露正文
```

## 10. API 与 CLI

API 请求可以传入：

```text
user_id
project_id
session_id
include_memory_status
```

`include_memory_status=true` 只返回：

```text
loaded / written
recent_message_count
retrieved_memory_count
estimated_tokens
summary_used
written_count / skipped_count
```

CLI 交互模式会固定一个 `session_id`，保证同一进程内的会话连续。

## 11. 故障策略

默认建议 fail-open：Memory 后端故障时记录 Warning，返回空 Memory 上下文并继续回答。只有合规或业务明确要求时才使用 fail-closed。

Memory 写入必须幂等，避免 Graph 重试或重复事件产生重复记录。

## 12. 隐私与删除

生产环境必须补充：

```text
身份认证和租户授权
用户查看、纠正和删除 Memory 的接口
保留期与过期任务
加密和审计
敏感类型识别
备份与恢复
```

当前基础实现不等于生产级隐私管理系统。

## 13. 测试

至少验证：

```text
user/project/session 隔离
显式写入和下一轮读取
敏感内容拒绝
重复写入幂等
fail-open
JSON Store 持久化
Token Budget
Prompt 不泄露 Memory 正文到 Trace
```
