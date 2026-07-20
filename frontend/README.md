# KG-RAG Agent Frontend

轻量 Vue 3 前端，只包含聊天、服务状态、会话隔离标识、Citation 和 Memory 状态展示。

## 本地运行

先启动后端：

```bash
python -m kg_rag_agent --serve --host 127.0.0.1 --port 8000
```

再启动前端：

```bash
npm ci
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`。需要修改时复制环境变量模板：

```bash
cp .env.example .env
```

## 生产构建

```bash
npm run build
npm run preview
```

构建结果位于 `dist/`。

## Docker Compose

在项目根目录执行：

```bash
docker compose up --build -d
```

访问 `http://127.0.0.1:5173`。`nginx.conf` 会将 `/api/` 转发到 Compose 中名为 `api` 的后端服务。

单独构建前端镜像：

```bash
docker build -t kg-rag-agent-frontend ./frontend
```
