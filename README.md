# Multi-Agent Code Review System

A full-stack AI code review platform powered by three specialized agents working in concert — Reviewer, Researcher, and Reporter — orchestrated via LangGraph and a message bus.

## How It Works

1. **Submit code** — paste source code or provide a GitHub PR URL
2. **Multi-agent analysis** — three agents collaborate: Researcher gathers context, Reviewer finds issues, Reporter composes the report
3. **Structured report** — score, categorized issues by severity, best practice suggestions, and a full Markdown review

## Architecture

```
+--------------------+       +---------------------+
|   Next.js Frontend  |       |   FastAPI Backend    |
|   (localhost:3000)  |<----->|   (localhost:8000)   |
|                     |  REST |                      |
|  +---------------+  |       |  +----------------+  |
|  | Code Input    |  |       |  | /api/review    |  |
|  | PR URL Input  |  |       |  | /api/task      |  |
|  | Report View   |  |       |  | /api/report    |  |
|  +---------------+  |       |  +----------------+  |
+--------------------+       +-------+---+----------+
                                      |   |
                          +-----------+   +-----------+
                          |                           |
                    +-----v------+             +------v------+
                    | Orchestrator|             | TaskManager |
                    +-----+------+             +------+------+
                          |                           |
              +-----------+-----------+       +-------v--------+
              |           |           |       |    SQLite       |
        +-----v---+ +----v----+ +---v------+ +----------------+
        |Reviewer | |Researcher| |Reporter  |
        |Agent    | |Agent     | |Agent     |
        +---------+ +----------+ +----------+
              |           |           |
              +-----------+-----------+
                          |
                    +-----v------+
                    | LLM Client |
                    | (DeepSeek) |
                    +------------+
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Poetry
- DeepSeek API Key

### 1. Install and Configure

```bash
git clone https://github.com/Zhuyuxuan0923/A-multi-agent-collaborative-code-review-system.git
cd A-multi-agent-collaborative-code-review-system
poetry install
```

Create `.env` in the project root:

```ini
DEEPSEEK_API_KEY="sk-your-key"
LLM_PROVIDER="deepseek"
```

### 2. Start Backend

```bash
poetry run uvicorn study_agent.api.server:app --reload --host 0.0.0.0 --port 8000
```

API docs at http://localhost:8000/docs

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### 4. Docker Deployment

```bash
docker compose up -d
```

Backend on port 8000, frontend on port 3000. See `docs/deployment-guide.md` for details.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/review` | Submit code for review |
| `POST` | `/api/review/pr` | Submit PR URL for review |
| `GET` | `/api/task/{id}` | Get task status |
| `GET` | `/api/report/{id}` | Get review report |
| `GET` | `/api/tasks` | List recent tasks |

## Project Structure

```
study-agent/
+-- src/study_agent/
|   +-- agent/                   # Agent implementations
|   |   +-- code_review_agents.py   # Reviewer/Researcher/Reporter agents
|   |   +-- bus_orchestrator.py     # Message bus orchestration
|   |   +-- message_bus.py          # Pub-sub + P2P message bus
|   |   +-- message_protocol.py     # AgentMessage protocol (10 fields)
|   |   +-- conflict_resolver.py    # Conflict detection + voting + arbitration
|   |   +-- langgraph_router.py     # LangGraph-based routing agent
|   |   +-- react_agent.py          # Hand-written ReAct agent
|   |   +-- plan_execute_agent.py   # Plan-Execute agent
|   |   +-- agent_guard.py          # Security guard (input/tool/loop)
|   |   +-- agent_evaluator.py      # Agent evaluation framework
|   |   +-- trace.py / traced_agent.py  # OpenTelemetry tracing
|   |   \-- state.py                # Agent state management
|   +-- api/                     # FastAPI backend
|   |   +-- server.py               # App factory + routes
|   |   +-- models.py               # Pydantic request/response models
|   |   +-- task_manager.py         # Async task lifecycle
|   |   \-- database.py             # SQLite schema + CRUD
|   +-- github/                  # GitHub integration
|   |   \-- diff_fetcher.py         # PR diff fetching
|   +-- llm/                     # LLM abstraction layer
|   +-- prompt/                  # Prompt templates + evaluation
|   +-- tools/                   # Tool system (calculator, datetime, etc.)
|   +-- config/                  # Provider configuration
|   \-- memory/                  # Memory implementations (Buffer/Summary/Vector/Hybrid)
+-- frontend/                    # Next.js 16 frontend
|   +-- src/
|   |   +-- app/
|   |   |   +-- page.tsx            # Home: code input + PR URL tabs
|   |   |   +-- task/[id]/page.tsx  # Task progress with polling
|   |   |   \-- report/[id]/page.tsx # Review report with score/issues
|   |   \-- lib/api.ts              # API client
|   \-- Dockerfile
+-- tests/                       # Test suite (72 tests)
+-- docs/
|   +-- deployment-guide.md         # Server deployment guide
|   \-- superpowers/specs/          # Design specifications
+-- Dockerfile                   # Backend Docker image
+-- docker-compose.yml           # Two-container deployment
\-- pyproject.toml               # Project configuration
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 16 + React 19 + TypeScript |
| Agent Framework | LangGraph + hand-written ReAct |
| LLM | DeepSeek (Chat + Reasoner) |
| Database | SQLite (tasks, reviews) |
| Testing | pytest (72 tests) |
| Code Quality | Black + Ruff + MyPy + Pre-commit |
| Deployment | Docker Compose (2 containers) |

## Running Tests

```bash
poetry run pytest tests/ -v
```

## License

MIT
