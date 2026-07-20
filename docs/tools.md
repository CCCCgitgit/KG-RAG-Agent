# Tools 与 ToolRegistry 设计规范

## 1. 当前实现

```text
tools/
├── registry.py
├── schemas.py
├── permissions.py
├── errors.py
├── file_tools.py
├── kg_tools.py
└── vector_tools.py
```

ToolRegistry、统一 Schema 和集中权限策略已实现基础版本。

## 2. 定位

```text
Graph / Agent / Service / Future MCP
                 ↓
             ToolRegistry
                 ↓
             Tool Adapter
                 ↓
kg / retrieval / file / future memory capabilities
```

工具只适配已有领域能力，不实现第二套算法。

## 3. 工具定义

工具应具有稳定的：

```text
name
version
description
input_schema
output_schema
handler
permission
timeout
max_calls_per_request
side_effect
idempotent
enabled
```

建议命名：

```text
kg.relation_search
kg.path_search
kg.neighbor_search
retrieval.entity_search
retrieval.hybrid_search
file.read_text
```

## 4. 统一结果

工具结果统一包含：

```text
ok
data
error_code
error_message
metadata
```

Metadata 可以记录工具名、耗时、截断和来源，但不能包含底层连接或完整堆栈。

## 5. Registry 职责

```text
注册和去重
按名称发现
Schema 校验
权限判断
超时与调用次数
结果字符数限制
统一错误映射
Trace
```

Registry 不决定业务路由，也不允许模型自行注册新工具。

## 6. 权限

权限类型按当前模型区分只读、执行和潜在破坏性能力。`configs/tools.yaml` 当前默认：

```yaml
enabled: true
allowed_permissions: [read, execute]
allow_destructive: false
```

外部请求只能通过 `allowed_tools` 缩小工具集合，不能扩大系统级权限。

## 7. FileTools

文件工具必须遵守：

```text
base_dir 沙箱
默认拒绝绝对路径
阻止 ../ 逃逸
限制读取和写入字节数
写入与删除默认关闭
符号链接策略明确
```

Docker 或生产部署中还应配合只读挂载和 OS 权限。

## 8. KGTools 与 VectorTools

KGTools 只调用 `kg/` 的关系、路径、邻居、子图和 Evidence 能力；VectorTools 只调用 `retrieval/` 的 Embedding、Store、Retriever 和 Reranker。

二者均需限制：

```text
实体数量
路径深度
返回数量
结果字符数
超时
```

## 9. Memory Tool

当前没有对模型公开的 `memory_tools.py`。Memory 读写由 Graph 首尾节点和 MemoryManager 控制。后续如增加“查看/删除/纠正 Memory”工具，必须要求明确用户授权，并与普通自动写入权限分离。

## 10. AgentState

State 可以记录标准化 Tool Call 和 Tool Result 摘要，但不能保存 Tool 实例、Registry、文件句柄、数据库连接或 MCP Session。

## 11. MCP 关系

MCP Adapter 可以将已授权 ToolRegistry 工具映射为 MCP Tool，但不得绕过 Registry、Schema 或 PermissionPolicy。具体规划见 [mcp.md](mcp.md)。

## 12. 测试

```text
重复注册
未知工具
Schema 错误
权限拒绝
超时
结果截断
路径逃逸
副作用工具默认关闭
并发调用限制
```
