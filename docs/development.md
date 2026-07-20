# 开发与重构规范

## 1. docs-first

新增或重构模块前：

1. 在专项文档说明职责、输入输出和边界。
2. 确认正式实现目录和依赖方向。
3. 定义兼容与迁移策略。
4. 编写或更新测试。
5. 修改代码后同步文档。

文档应描述当前真实状态，并明确区分“已实现”和“计划”。

## 2. 新功能落位

- HTTP 协议：`app/`
- 外部业务协调：`services/`
- Agent 门面：`agents/`
- 流程节点：`graph/nodes/`
- 可复用算法：对应领域模块
- 共享 Client/Store：`runtime/`
- Prompt 正文：`src/kg_rag_agent/prompts/`
- 离线构建：`data_pipeline/`
- 评估：`evaluation/`

无法明确归属时，先修正文档和职责，不要先创建 `common.py`。

## 3. Node 规范

```text
一个 Node 一个原子步骤
输入 AgentState
返回部分字段更新 Mapping
不直接调用其他 Node
不创建重型共享对象
不在 State 中写 Runtime 对象
错误转换为标准字段和 Trace
```

## 4. Runtime 规范

RuntimeFactory 负责依赖创建；RuntimeContext 负责共享和关闭。测试可以注入 Fake，但生产代码不得在每个请求中重复创建 Graph、Embedding、Memory Store 或 ToolRegistry。

## 5. 请求选项

公开白名单当前包括：

```text
retrieval_top_k
path_max_depth
temperature
max_tokens
language
include_citations
allowed_tools
```

新增选项必须同时更新：

```text
Agent RequestOptions
AgentService 白名单
API ChatOptions
CLI 参数（如需要）
测试
API 和 README 文档
```

## 6. Memory 变更

修改 Memory 时必须验证：

```text
user/project/session 隔离
敏感内容过滤
幂等写入
fail-open 行为
Token Budget
API 不暴露正文
Prompt 不执行 Memory 指令
```

## 7. Prompt 变更

Prompt 语义变化需更新版本或 Checksum，并运行路由、实体、Evidence、Citation 和注入安全回归。不要只依靠人工观察一个示例。

## 8. Tool 变更

新工具必须先有领域实现，再添加 Adapter、Schema、权限、超时、结果限制和测试。写入、执行、删除默认关闭。

## 9. 兼容接口

兼容层必须：

```text
有明确旧名称
只转发到正式实现
不复制业务逻辑
有删除条件
有测试
```

当前示例包括 `KGRAgent` 别名和 `scripts/run_server.py` 转发。

## 10. 测试

```bash
pytest
pytest -m unit
pytest -m integration
pytest -m api
pytest -m memory
```

质量检查：

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy -p kg_rag_agent
python -m build
```

也可以使用：

```bash
make check
```

## 11. 提交前检查

```text
测试通过
新配置有默认值和校验
无密钥和本地绝对路径
无缓存、大图谱、模型和 Memory 文件
文档与代码一致
公开接口保持兼容或有迁移说明
```

## 12. 注释和类型

注释解释“为什么”和边界，不重复代码。公共函数和复杂 State 字段使用类型提示；`Any` 只用于真实外部边界或迁移兼容。

## 13. 错误处理

领域错误、Runtime 错误、Tool 错误、API 错误分层转换。用户响应不暴露内部堆栈；日志通过 request_id/trace_id 关联。

## 14. 前端与 MCP

未来前端和 MCP 都必须调用现有 API/Service/ToolRegistry，不得另建独立 Agent 工作流。
