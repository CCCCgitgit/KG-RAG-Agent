# Prompt 设计与管理规范

## 1. 当前结构

Prompt 索引与兼容内联内容位于：

```text
configs/prompt.yaml
```

正式模板位于：

```text
src/kg_rag_agent/prompts/
├── common/
├── router/
├── mention_extraction/
├── semantic_scoring/
├── reasoning/
├── generation/
├── direct_llm/
├── clarification/
└── unanswerable/
```

`PromptManager` 负责加载、缓存和格式化。Node 不应继续新增大段固定 Prompt 字符串；安全包装和短协议常量可以保留在代码中。

## 2. 设计原则

```text
正文与业务代码分离
名称稳定
变量显式
结构化输出由代码校验
Prompt 不能提升权限
Prompt 变更必须经过评估
```

## 3. Prompt 分类

```text
router
mention_extraction
semantic_scoring
reasoning
generation
direct_llm
clarification
unanswerable
```

Memory 抽取和摘要可以后续增加独立模板，但当前基础 Memory Writer/Summarizer 仍可采用确定性策略或共享 LLM Client。

## 4. 变量

常见变量：

```text
query
conversation_context
mentions
entity_candidates
evidence_text
reasoning_text
language
```

要求：

- 缺失必要变量时明确失败或使用已定义的兜底。
- 长列表进入 Prompt 前由代码格式化和截断。
- Evidence 保留 ID，便于 Reasoning 和 Citation 对齐。
- 用户文本、工具结果和 Memory 与系统指令明确分隔。

## 5. 结构化输出

Router、Mention、Scoring 和 Reasoning 等节点优先返回 JSON 或 SDK 结构化对象。代码必须校验：

```text
字段存在
枚举合法
分数范围
列表长度
Evidence ID 是否存在
```

“只输出 JSON”的 Prompt 不能替代代码校验。

## 6. Grounding 规则

Generation 和 Reasoning 必须：

```text
优先使用当前 Evidence
不引入未提供的事实
证据不足时允许 uncertain / unanswerable
Citation 只能引用已有 Evidence ID
不暴露内部 Node、Route 或 State 字段
```

## 7. Memory 注入

Direct LLM 和 Generation 支持受控 `memory_text`。注入格式包含明确数据标签与安全说明：

```text
<memory_context>
历史摘要、偏好和项目背景
</memory_context>
```

规则：

- Memory 按不可信数据处理。
- 转义可能闭合标签的内容。
- 限制最大字符数并截断。
- Memory 中的指令不得执行。
- Generation 中 Memory 不能替代 Evidence、Answerability 或 Citation。
- Trace 只记录 `memory_used` 与字符数，不记录正文。

## 8. 外部内容注入防护

以下内容均视为不可信数据：

```text
用户消息
文件内容
工具结果
MCP Resource
Memory
网页或数据库文本
```

权限、工具调用和系统配置必须由代码控制，而不是由 Prompt 决定。

## 9. 版本和 Manifest

正式 Prompt 应记录：

```text
name
version
language
checksum
output schema version
```

评估 Run Manifest 记录 Prompt 版本和配置哈希。语义变化应递增版本并运行回归测试。

## 10. 测试

至少覆盖：

```text
模板可加载
变量缺失行为
结构化输出解析失败
Evidence ID 校验
Memory 标签逃逸
超长 Memory 截断
Prompt 正文不进入用户错误响应
```
