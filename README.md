# agent-core

统一 LLM 调用层 —— 一行配置切换 OpenAI / Anthropic / DeepSeek / GLM-4 / Moonshot，自带重试机制与工具抽象。

## 这是什么

一个轻量级的 AI Agent 基础设施包。把调用大模型时最重复的三件事——**配置管理**、**API 重试**、**工具定义**——封装成三个独立模块，让你专注于写业务逻辑。

## 快速开始

```bash
# 安装依赖
poetry install
```

```python
from study_agent import LLMClient

# 创建客户端（API Key 从环境变量读取）
client = LLMClient(provider="deepseek")

# 非流式对话
reply = client.chat("什么是机器学习？")
print(reply)

# 流式对话（打字机效果）
for chunk in client.chat_stream("用三句话介绍 Python"):
    print(chunk, end="")

# 带 system prompt
reply = client.chat(
    user_message="解释什么是递归",
    system="你是一个小学老师，用最通俗的语言解释，不超过3句话。",
)
```

## 环境变量

在项目根目录创建 `.env` 文件：

```ini
# 至少配一家
DEEPSEEK_API_KEY="sk-your-key"
# 可选
OPENAI_API_KEY="sk-your-key"
ANTHROPIC_API_KEY="sk-your-key"
ZHIPU_API_KEY="your-key"
MOONSHOT_API_KEY="sk-your-key"

# 默认使用的厂商
LLM_PROVIDER="deepseek"
```

## 支持的 Provider

| Provider   | 模型示例              | SDK 类型   |
|------------|----------------------|-----------|
| openai     | gpt-4o               | OpenAI    |
| anthropic  | claude-sonnet-4-6    | Anthropic |
| deepseek   | deepseek-chat        | OpenAI 兼容 |
| zhipu      | glm-4-flash          | OpenAI 兼容 |
| moonshot   | moonshot-v1-8k       | OpenAI 兼容 |

新增一家厂商只需在 `config/settings.py` 的 `PROVIDER_CONFIGS` 中加 4 行配置。

## 包结构

```
src/study_agent/
├── config/          # 配置中心
│   └── settings.py  # Provider 配置、默认模型、LLMConfig
├── llm/             # LLM 调用层
│   ├── client.py    # LLMClient 统一客户端
│   └── retry.py     # 重试机制（指数退避、智能判断错误类型）
├── tools/           # 工具抽象层
│   └── base.py      # BaseTool / ToolDefinition / ToolParameter
└── __init__.py
```

**设计原则**：换模型只改 provider 字符串，业务代码完全不动。

## 重试机制

所有 API 调用自动带有智能重试：

- 网络抖动、被限流（HTTP 429）、服务器临时故障 → 自动重试
- API Key 错误（HTTP 401）、参数格式错误（HTTP 400） → 不重试，直接报错
- 等待策略：指数退避（1s → 2s → 4s → ...，最长 30s），避免压垮服务器

## 工具系统

为 Week 2 的 Tool Calling 做准备。定义一个工具只需继承 `BaseTool`：

```python
from study_agent.tools import BaseTool, ToolDefinition, ToolParameter

class SearchWebTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_web",
            description="搜索互联网获取实时信息",
            parameters=[
                ToolParameter(name="query", type="string", description="搜索关键词"),
            ],
        )

    def execute(self, query: str) -> str:
        # 实际的搜索逻辑
        return f"搜索结果：关于 '{query}' ..."
```

## 技术栈

- Python 3.11+
- OpenAI SDK（调用 OpenAI / DeepSeek / GLM-4 / Moonshot）
- Anthropic SDK（调用 Claude）
- tenacity（重试机制）
- FastAPI（API 服务，见 `src/study_agent/main.py`）

## 开发

```bash
poetry install          # 安装依赖
poetry run ruff check . # 代码规范检查
poetry run mypy src/    # 类型检查
```
