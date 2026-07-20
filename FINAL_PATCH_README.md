# KG-RAG Agent 最终补丁说明

本压缩包只包含相对于 `Kg_rag_agent(11).zip` 新增或修改的文件，未改动文件没有重复打包。

## 覆盖方式

将压缩包内文件按原目录结构覆盖到项目根目录。覆盖后执行：

```bash
python scripts/migrate_legacy_data.py --cleanup
```

该命令会把旧版顶层 `data/raw` 和 Production 中间产物迁移到 `data/production/`，并删除旧的混合目录。`data/demo/` 已包含可直接运行的 Demo Graph、Alias、Vector Store 和 Manifest。

如果需要重建正式数据：

```bash
python scripts/build_all.py --profile production
```

## 最终验证

```bash
pytest -q
cd frontend && npm ci && npm run build
cd .. && docker compose config
```

本次验收结果：后端 39 项测试通过，Vue 生产构建通过，Wheel 构建通过，Demo Manifest 校验通过，Memory 用户/项目/会话隔离通过。

## 启动

本地开发：

```bash
python -m kg_rag_agent --serve --host 127.0.0.1 --port 8000
cd frontend
npm ci
npm run dev
```

Docker Compose：

```bash
docker compose up --build -d
```

前端默认地址：`http://127.0.0.1:5173`
后端文档：`http://127.0.0.1:8000/docs`
