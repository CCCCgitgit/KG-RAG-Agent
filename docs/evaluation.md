# Evaluation 设计规范

## 1. 目标

评估必须复用正式调用链：

```text
Dataset
  ↓
Evaluation Runner / EvaluationService
  ↓
AgentService
  ↓
KGRAGAgent
  ↓
LangGraph
  ↓
Metrics + Report + Run Manifest
```

不得为评估维护简化版实体链接、检索或回答流程。

## 2. 当前模块

```text
evaluation/
├── schemas.py
├── dataset_loader.py
├── test_cases.py
├── evaluator.py
├── metrics.py
├── reporter.py
├── manifest.py
└── runner.py
```

入口：

```text
scripts/evaluate.py
services/evaluation_service.py
```

## 3. 数据集

每条 Case 建议包含：

```text
case_id
query
expected_answer 或 expected_facts
expected_route
expected_entities
expected_evidence
metadata
```

数据集记录名称、版本和来源。评估不应修改正式 Memory 或共享生产会话。

## 4. 指标层次

### Routing

```text
route accuracy
clarification accuracy
error rate
```

### Entity Resolution

```text
mention precision/recall
candidate recall@k
linking accuracy
grounding success rate
```

### Retrieval

```text
Evidence recall@k
MRR / Hit@k
path correctness
neighbor relevance
```

### Answering

```text
answerability accuracy
fact correctness
citation precision/recall
unsupported claim rate
```

### Engineering

```text
latency
LLM/tool calls
timeout/error rate
result size
```

### Memory

```text
isolation violations
relevant memory retrieval
sensitive write rejection
duplicate write rate
```

## 5. Run Manifest

当前 Manifest 记录：

```text
run_id
dataset name/version/size/hash
start/finish time
Git revision
config hash
request options
model information
prompt version
graph checksum
vector store version
Python/platform
```

报告必须能够回溯本次运行的代码、配置和数据基线。

## 6. 输出

建议输出：

```text
summary.json
cases.jsonl 或 cases.csv
run_manifest.json
错误与分组统计
```

默认写入 `outputs/evaluation/`，不提交生产结果到 Git。

## 7. Memory 评估隔离

评估用固定测试前缀和独立 Store，或完全关闭长期 Memory。每个 Case 使用明确的用户、项目和会话作用域，避免前一 Case 污染后一 Case。

## 8. 常用命令

```bash
python scripts/evaluate.py \
  --input data/demo/examples/demo_questions.json \
  --output-dir outputs/evaluation
```

```bash
make evaluate INPUT=data/demo/examples/demo_questions.json LIMIT=10
```

## 9. 回归门槛

关键模块变更后至少比较：

```text
整体成功率
路由与实体链接
Evidence 与 Citation
错误率
延迟
Memory 隔离
```

不能只观察最终文本是否“看起来合理”。
