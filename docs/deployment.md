# KG-RAG Agent 部署说明

## 1. 环境

支持 Python 3.10–3.13。推荐生产镜像使用 Python 3.12，并在升级依赖后重新运行完整测试和 Wheel 构建。

## 2. 安装

完整本地环境：

```bash
python -m pip install -r requirements.txt
```

按功能安装：

```bash
python -m pip install -e .
python -m pip install -e ".[api,retrieval]"
python -m pip install -e ".[dev,test]"
```

## 3. 配置与密钥

复制 `.env.example` 为 `.env`。密钥不提交 Git；生产环境使用部署平台 Secret。

详细业务参数位于 `configs/`。外部请求不能覆盖数据路径、Store 路径、Memory Namespace 或权限。

## 4. CLI

```bash
python -m kg_rag_agent -q "问题"
python -m kg_rag_agent --interactive --memory-status
```

## 5. FastAPI

```bash
python -m kg_rag_agent --serve --host 0.0.0.0 --port 8000
```

开发：

```bash
python -m kg_rag_agent --serve --reload
```

多进程：

```bash
python -m kg_rag_agent --serve --workers 4
```

`--reload` 与多 Worker 不同时使用。

## 6. Docker

构建：

```bash
docker build -t kg-rag-agent:latest .
```

Compose：

```bash
docker compose up --build -d
docker compose logs -f api frontend
docker compose down
```

当前 Dockerfile 使用多阶段构建和非 root 用户。Compose 挂载：

```text
configs/  → 只读
data/     → 图谱和向量产物
outputs/  → 评估、缓存和持久化 Memory
logs/     → 日志
```

宿主机目录必须对容器 UID/GID 可读写。

## 7. 健康检查

```text
/api/health  进程存活
/api/ready   Runtime 与 AgentService 就绪
/api/status  简单状态
```

Docker Healthcheck 当前访问 `/api/health`。生产流量切换应优先结合 `/api/ready`。

## 8. Memory 部署

- 单进程测试可以使用 InMemory Store。
- JSON Store 适合本地单实例持久化。
- 多 Worker 或多实例共享普通 JSON 文件存在并发风险，应改用数据库或受控服务。
- `outputs/memory/` 必须配置备份、权限和保留策略。
- 标识来自请求时必须配合真实身份认证，不能只相信客户端自报 `user_id`。

## 9. 数据产物

生产部署前固定同一 Build Manifest 的：

```text
Graph
Entity Index
Alias Map
Vector Store
Embedding 配置
```

不要在启动请求期间自动下载大型模型或重建向量库。

## 10. 日志与可观测性

日志不得包含：

```text
API Key
完整 Prompt
Memory 原文
完整 AgentState
内部连接串
```

生产环境应增加结构化日志、请求 ID、指标、Trace 采样、告警和集中收集。

## 11. 安全

当前基础 API 不包含身份认证和限流。暴露公网前必须增加：

```text
TLS
身份认证
租户授权
限流与配额
CORS 收紧
请求大小限制
Tool 权限审计
Memory 删除和保留策略
```

## 12. 前端

前端位于 `frontend/`。本地开发：

```bash
cd frontend
npm ci
npm run dev
```

生产构建：

```bash
cd frontend
npm run build
```

Compose 会同时启动 `api` 与 `frontend`。Nginx 将 `/api/` 代理到 `api:8000`，浏览器只需访问 `http://127.0.0.1:5173`。前端不与 Python Runtime 共享进程状态。
