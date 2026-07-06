"""AutoGen 多 Agent 协作 —— 研究员 + 写手 双 Agent 团队。

用 AutoGen 0.7.x 实现:
  - Researcher Agent: 深度研究一个技术主题，收集和组织信息
  - Writer Agent:    将研究成果转化为面向初学者的科普文章

两个 Agent 通过 RoundRobinGroupChat 轮流发言，
用 TextMentionTermination + MaxMessageTermination 组合控制终止。

核心概念（按学习顺序）：
  1. AssistantAgent          —— AutoGen 的 Agent 类型，支持工具调用和系统提示
  2. OpenAIChatCompletionClient —— 模型客户端（兼容 OpenAI 协议）
  3. RoundRobinGroupChat     —— 多 Agent 轮流发言的对话管理器
  4. TextMentionTermination  —— 当输出中出现特定文字时终止对话
  5. MaxMessageTermination   —— 消息数量达到上限时终止（安全网）
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ═══════════════════════════════════════════════════════════════════
# 1. 模型客户端
# ═══════════════════════════════════════════════════════════════════


def _build_model_client() -> OpenAIChatCompletionClient:
    """构建 DeepSeek 模型客户端。

    关键参数 model_info:
      AutoGen 内置了 OpenAI 模型的能力表（哪个模型支持 vision/function calling/json）。
      DeepSeek 不在这个表里，必须手动提供 ModelInfo，否则会报 ValueError。
      这是接入非 OpenAI 模型的必要步骤。
    """
    from autogen_core.models import ModelFamily, ModelInfo

    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    model_info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family=ModelFamily.ANY,
        structured_output=True,
    )

    return OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model_info=model_info,
        temperature=0.7,
        max_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════════════
# 2. Agent 定义 —— 系统提示是 AutoGen Agent 的"角色说明书"
# ═══════════════════════════════════════════════════════════════════
#
# AutoGen 的 Agent 通过 system_message 定义自己的行为。
# 这和 CrewAI 用 role+goal+backstory 不同。
#
# AutoGen 方式：
#   "你是一个研究员。你的职责是...输出格式要求..."
#   -> 直接、指令式，适合需要精确控制行为的场景
#
# CrewAI 方式：
#   role="Senior Researcher"
#   goal="Uncover insights about {topic}"
#   backstory="Veteran researcher with 15 years..."
#   -> 角色扮演式，LLM 更容易"入戏"，输出风格更稳定

RESEARCHER_SYSTEM_PROMPT = """你是一位资深技术研究员。

你的职责：
1. 深入调研给定的技术主题
2. 从多个维度组织信息：定义、核心概念、工作原理、使用场景、优缺点
3. 提供具体代码示例（如果有的话）
4. 标注信息的确定性（确定的事实 vs 需要验证的观点）

输出格式要求：
- 用清晰的分段标题组织内容
- 代码示例用 markdown 代码块
- 在信息末尾注明哪些是确定的事实，哪些是你推断的

重要：你的输出会直接传递给写手，请不要添加"我研究了..."这类元叙述。
研究报告写完后不需要等待回复，写手会直接基于你的报告写文章。
"""

WRITER_SYSTEM_PROMPT = """你是一位技术写手，擅长把复杂的技术概念转化为初学者能看懂的文章。

你的职责：
1. 接收研究员的调研结果
2. 写一篇面向编程初学者的科普文章（目标读者：学过 Python 基础但不懂该主题的人）
3. 用生活类比解释抽象概念
4. 文章结构：标题 -> 引出问题 -> 核心概念解释 -> 代码示例 -> 总结

写作要求：
- 标题要有吸引力
- 开头用一个生活场景引出"为什么需要这个技术"
- 每个概念都配上类比
- 代码示例要逐行解释
- 字数控制在 1200 字以内
- 用中文写作

当你完成文章后，必须单独一行输出: TERMINATE
不要多说任何其他内容，输出 TERMINATE 后对话会自动结束。
"""


def create_researcher(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """创建研究员 Agent。"""
    return AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message=RESEARCHER_SYSTEM_PROMPT,
        description="技术研究员，负责深入调研技术主题并输出结构化调研报告",
    )


def create_writer(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """创建写手 Agent。"""
    return AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=WRITER_SYSTEM_PROMPT,
        description="技术写手，负责将调研结果转化为面向初学者的科普文章",
    )


# ═══════════════════════════════════════════════════════════════════
# 3. 构建团队 —— 关键：终止条件的设计
# ═══════════════════════════════════════════════════════════════════


def build_research_team(
    max_messages: int = 8,
) -> RoundRobinGroupChat:
    """构建研究员+写手协作团队。

    终止条件设计（这是 AutoGen 最需要小心的地方）：

      AutoGen 的 Agent 会持续对话直到满足终止条件。
      如果没有终止条件或条件设计不当，Agent 会陷入"聊天循环"——
      互相感谢、互相确认、没完没了。

      所以需要两层保护：
        1. TextMentionTermination("TERMINATE") —— 正常完成时主动终止
        2. MaxMessageTermination(N)          —— 安全网，防止无限对话

    """
    model_client = _build_model_client()

    researcher = create_researcher(model_client)
    writer = create_writer(model_client)

    # 组合终止条件：任一满足即停止
    # | 是 Python 3.11+ 的 Union 类型语法，在这里表示"或"关系
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages)

    team = RoundRobinGroupChat(
        participants=[researcher, writer],
        termination_condition=termination,
        max_turns=10,
    )

    return team


# ═══════════════════════════════════════════════════════════════════
# 4. 运行接口
# ═══════════════════════════════════════════════════════════════════


async def run_research(task: str) -> str:
    """运行研究+写作任务，返回最终结果。"""
    team = build_research_team()
    result = await team.run(task=task)
    return str(result)


async def run_and_print(task: str) -> None:
    """运行任务并逐条打印 Agent 消息 —— 用于演示。

    每条消息都会标注来源（researcher / writer / user），
    让你清楚看到协作过程中谁在什么时候说了什么。
    """
    team = build_research_team()

    print(f"任务: {task}")
    print("=" * 60)

    message_count = 0
    async for message in team.run_stream(task=task):
        message_count += 1
        source = getattr(message, "source", "unknown")
        content = getattr(message, "content", str(message))

        # 跳过 user 自己的消息（就是 task 本身）
        if source == "user":
            continue

        print(f"\n--- [{source}] (消息 #{message_count}) ---")
        if isinstance(content, str) and len(content) > 1500:
            print(content[:1500] + "\n... (内容过长，已截断)")
        else:
            print(content)

    print("\n" + "=" * 60)
    print(f"[OK] 任务完成，共 {message_count} 条消息")
    print()


# ═══════════════════════════════════════════════════════════════════
# 5. 命令行入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    topic = "Python 异步编程（asyncio）—— 协程、事件循环、await 的工作原理"
    asyncio.run(run_and_print(topic))
