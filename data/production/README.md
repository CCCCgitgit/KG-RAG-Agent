# Production Data Profile

该目录保存正式数据构建产物，不与 `data/demo/` 混用。

将原始文件放入：

```text
data/production/raw/
├── ent2ids
├── relation2ids
└── path_graph
```

然后在项目根目录执行：

```bash
python scripts/build_all.py --profile production
```

生成的 `processed/`、`kg/`、`vector_store/` 和 `build_manifest.json` 必须来自同一次构建。
