# 数据构建流程说明

## 1. 职责

`data_pipeline/` 负责离线标准化和产物构建；在线请求只能加载已构建产物，不能触发完整重建。

```text
data/<profile>/raw
  ↓
DataLoader
  ↓
data/<profile>/processed
  ↓
GraphBuilder
  ↓
data/<profile>/kg
  ↓
VectorBuilder
  ↓
data/<profile>/vector_store
  ↓
Build Manifest
```

## 2. 当前模块

```text
data_pipeline/
├── schemas.py
├── paths.py
├── data_loader.py
├── graph_builder.py
├── vector_builder.py
├── manifest.py
└── pipeline.py
```

脚本：

```text
scripts/build_data.py
scripts/build_kg.py
scripts/build_vector_store.py
scripts/build_entity_vector_store.py
scripts/build_all.py
```

## 3. 当前标准产物

```text
entities.json
relations.json
triples.csv
entity_index.json
relation_index.json
alias_map.json
data_loader_stats.json
graph.pkl
graph_stats.json
vector_store_stats.json
build_manifest.json
```

## 4. Profile

支持：

```text
demo
production
```

正式运行统一使用 `data/<profile>/`。脚本默认 `demo`；旧顶层路径仅用于迁移兼容，可通过 `python scripts/migrate_legacy_data.py` 复制到 Production Profile。

不同 Profile 或不同 Build ID 的图谱、索引和向量库不得交叉使用。

## 5. Raw 数据规则

- 尽量保留源格式和只读副本。
- 清洗规则写入代码，不直接手工修改生产产物。
- 记录来源、版本、许可证和时间。
- 大文件和敏感数据不提交 Git。

## 6. 标准化协议

实体至少包含稳定 ID、名称、别名和 Metadata；关系包含稳定 ID、名称和方向约定；三元组包含主语、关系、宾语和可选来源/置信度。

离线 Normalization 与在线 Entity Resolution 必须共享相同规范，否则会导致链接失败。

## 7. 图谱构建

当前生成 NetworkX 图和 `graph.pkl`。要求：

```text
图类型和方向明确
节点/边属性可序列化
重复边策略稳定
保存前完成校验
统计信息独立输出
```

`graph.pkl` 是 Python 运行产物，不是跨语言长期交换格式。

## 8. 向量库构建

实体文档构造、Embedding 模型、维度、Collection 和构建参数必须记录。生产环境不应使用不可复现的临时 Hash Embedding 作为正式基线。

默认优先本地模型；允许下载时显式使用：

```bash
python scripts/build_all.py --profile demo --allow-download
```

## 9. Build Manifest

Manifest 记录：

```text
schema_version
build_id
created_at
profile
options
stages
artifact path / size / sha256
```

Manifest 会写入主要产物目录。加载生产数据前可以验证文件存在和可选校验和。

## 10. 常用命令

```bash
python scripts/build_all.py --profile demo
python scripts/build_all.py --profile production
python scripts/build_all.py --profile demo --skip-vector-store
```

也可以使用 Makefile：

```bash
make build-data-demo
make build-data-production
make build-data-no-vector PROFILE=demo
```

## 11. 在线加载

RuntimeFactory 负责创建 GraphLoader、EntityVectorStore、Retriever 和 EntityLinker。在线层只读取配置指定的产物，不自行推断或混用路径。

## 12. 测试

```text
输入解析
ID 稳定性
重复三元组
图统计
Profile 隔离
Manifest 校验
向量文档去重
离线/在线 Normalization 一致性
```
