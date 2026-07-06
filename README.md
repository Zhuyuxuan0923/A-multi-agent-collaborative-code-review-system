# 多 Agent 协作代码审查系统

基于 LangGraph 编排的三 Agent 协作代码审查平台 — Reviewer 审查代码、Researcher 补充上下文、Reporter 生成报告。支持代码粘贴和 GitHub PR 链接两种输入方式。

## 工作流程

1. **提交代码** — 粘贴源代码或提供 GitHub PR 地址
2. **多 Agent 协作分析** — Researcher 搜索相关上下文，Reviewer 发现问题和安全隐患，Reporter 汇总生成结构化报告
3. **审查报告** — 综合评分、按严重程度分类的问题列表、最佳实践建议、完整 Markdown 报告

## 系统架构

```
+--------------------+       +---------------------+
|   Next.js 前端      |       |   FastAPI 后端       |
|   (localhost:3000)  |<----->|   (localhost:8000)   |
|                     |  REST |                      |
|  +---------------+  |       |  +----------------+  |
|  | 代码输入      |  |       |  | /api/review    |  |
|  | PR URL 输入   |  |       |  | /api/task      |  |
|  | 报告展示      |  |       |  | /api/report    |  |
|  +---------------+  |       |  +----------------+  |
+--------------------+       +-------+---+----------+
                                      |   |
                          +-----------+   +-----------+
                          |                           |
                    +-----v------+             +------v------+
                    | 编排器      |             | 任务管理器   |
                    | Orchestrator|             | TaskManager |
                    +-----+------+             +------+------+
                          |                           |
              +-----------+-----------+       +-------v--------+
              |           |           |       |    SQLite       |
        +-----v---+ +----v----+ +---v------+ +----------------+
        |Reviewer | |Researcher| |Reporter  |
        |审查 Agent| |研究 Agent| |报告 Agent|
        +---------+ +----------+ +----------+
              |           |           |
              +-----------+-----------+
                          |
                    +-----v------+
                    | LLM 客户端  |
                    | (DeepSeek) |
                    +------------+
```

数据流：用户提交代码 -> Orchestrator 协调三 Agent -> 消息总线通信 -> Reporter 合并输出 -> 结构化报告

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 20+
- Poetry（Python 包管理器）
- DeepSeek API Key

### 1. 安装

```bash
git clone https://github.com/Zhuyuxuan0923/A-multi-agent-collaborative-code-review-system.git
cd A-multi-agent-collaborative-code-review-system
poetry install
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```ini
DEEPSEEK_API_KEY="sk-your-key"
LLM_PROVIDER="deepseek"
```

### 3. 启动后端

```bash
poetry run uvicorn study_agent.api.server:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问 http://localhost:8000/docs 查看 Swagger API 文档。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 http://localhost:3000

### 5. Docker 部署

```bash
docker compose up -d
```

后端 8000 端口，前端 3000 端口。详细部署说明见 `docs/deployment-guide.md`。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/review` | 提交代码审查 |
| `POST` | `/api/review/pr` | 提交 PR 审查 |
| `GET` | `/api/task/{id}` | 查询任务状态 |
| `GET` | `/api/report/{id}` | 获取审查报告 |
| `GET` | `/api/tasks` | 最近任务列表 |

## 项目结构

```
study-agent/
+-- src/study_agent/
|   +-- agent/                   # Agent 实现
|   |   +-- code_review_agents.py   # Reviewer / Researcher / Reporter 三 Agent
|   |   +-- bus_orchestrator.py     # 消息总线编排器
|   |   +-- message_bus.py          # 发布-订阅 + 点对点消息总线
|   |   +-- message_protocol.py     # AgentMessage 协议（10 个字段）
|   |   +-- conflict_resolver.py    # 冲突检测 + 投票 + 仲裁
|   |   +-- langgraph_router.py     # LangGraph 路由 Agent
|   |   +-- react_agent.py          # 手写 ReAct Agent
|   |   +-- plan_execute_agent.py   # Plan-Execute Agent
|   |   +-- agent_guard.py          # 安全守卫（输入/工具/循环三层检测）
|   |   +-- agent_evaluator.py      # Agent 评测框架
|   |   +-- trace.py                # OpenTelemetry 追踪
|   |   \-- state.py                # Agent 状态管理
|   +-- api/                     # FastAPI 后端
|   |   +-- server.py               # 应用工厂 + 路由注册
|   |   +-- models.py               # Pydantic 请求/响应模型（含安全校验）
|   |   +-- task_manager.py         # 异步任务生命周期管理
|   |   \-- database.py             # SQLite 建表 + CRUD
|   +-- github/                  # GitHub 集成
|   |   \-- diff_fetcher.py         # PR diff 拉取
|   +-- llm/                     # LLM 抽象层（5 家 provider 统一封装）
|   +-- prompt/                  # Prompt 模板 + 评测
|   +-- tools/                   # 工具系统（计算器、日期时间等）
|   +-- config/                  # Provider 配置中心
|   \-- memory/                  # Memory 实现（Buffer / Summary / Vector / Hybrid）
+-- frontend/                    # Next.js 16 前端
|   +-- src/
|   |   +-- app/
|   |   |   +-- page.tsx            # 首页：代码输入 + PR URL 双 Tab
|   |   |   +-- task/[id]/page.tsx  # 任务进度页（2 秒轮询）
|   |   |   \-- report/[id]/page.tsx # 报告页（评分 + 问题列表 + 建议）
|   |   \-- lib/api.ts              # API 客户端
|   \-- Dockerfile
+-- tests/                       # 测试（72 个用例）
+-- docs/
|   +-- deployment-guide.md         # 服务器部署指南
|   \-- superpowers/specs/          # 设计文档
+-- Dockerfile                   # 后端 Docker 镜像
+-- docker-compose.yml           # 双容器编排
\-- pyproject.toml               # 项目配置
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Next.js 16 + React 19 + TypeScript |
| Agent 编排 | LangGraph + 手写 ReAct + 消息总线 |
| LLM | DeepSeek（Chat + Reasoner） |
| 数据库 | SQLite（任务、报告持久化） |
| 测试 | pytest（72 个用例） |
| 代码质量 | Black + Ruff + MyPy + Pre-commit |
| 部署 | Docker Compose（双容器） |

## 安全设计

- **输入校验**：PR URL 使用 Pydantic `pattern` 正则校验，拒绝非 GitHub 地址
- **注入防护**：PR URL 仅用于提取 owner/repo/pr_number 调用 GitHub API，不进入 LLM Prompt
- **JSON 解析容错**：三级降级策略（标准解析 -> 正则提取 -> 哨兵值），应对 LLM 输出格式波动

## 运行测试

```bash
poetry run pytest tests/ -v
```

## License

MIT
