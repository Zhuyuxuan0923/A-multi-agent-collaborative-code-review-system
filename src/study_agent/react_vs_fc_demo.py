"""ReAct vs Function Calling 对比演示。

用同一个研究任务，分别通过 ReAct Agent 和 Function Calling (ToolCallLoop)
来执行，对比两者的推理过程、效率和可调试性。

这是 Week 5 Day 2 的核心实验。
"""

from __future__ import annotations

import logging
import sys

from study_agent.config.settings import get_config
from study_agent.llm.client import LLMClient
from study_agent.tools.tool_loop import ToolCallLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("react_vs_fc")

# ═══════════════════════════════════════════════════════════
# 演示前的准备工作
# ═══════════════════════════════════════════════════════════


def print_separator(title: str) -> None:
    """打印分隔线。"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_react_vs_fc():
    """核心对比实验。"""

    config = get_config()
    client = LLMClient(provider=config.provider, model=config.model)

    # ── 测试问题 ──
    question = "搜索 React 19 的新特性，计算如果 bundle size 减小 20% 后原来是 150KB 的话现在是多大，最后用一句话总结 React 19 最重要的改进。"

    print_separator("Part 1: ReAct Agent (文本推理)")
    print(f"[问题] {question}")
    print()

    from study_agent.agent.react_agent import ReactAgent

    react_agent = ReactAgent(client, max_rounds=8, parse_method="regex")
    react_answer = react_agent.run(question)

    print()
    print(f"[ReAct 最终答案] {react_answer}")

    # ── 对比：Function Calling ──
    print_separator("Part 2: Function Calling (ToolCallLoop)")

    from study_agent.tools.builtin_tools import CalculatorTool, TextStatsTool

    fc_loop = ToolCallLoop(client, tools=[CalculatorTool(), TextStatsTool()])

    # 注意：ToolCallLoop 没有搜索工具，所以我们提一个纯计算的问题
    fc_question = "计算从 1 加到 100 的和，然后告诉我这个结果的平方根是多少。"
    print(f"[问题] {fc_question}")
    print()

    fc_answer = fc_loop.run(fc_question)
    print(f"[FC 最终答案] {fc_answer}")

    # ── 对比总结 ──
    print_separator("Part 3: 对比分析")

    print(
        """
[ReAct 的特点]
  1. 推理过程完全可见 —— Thought 行告诉你 LLM "为什么"做这个决策
  2. 不依赖模型原生 tool_calls 能力 —— 任何文本模型都能用
  3. 每轮只能调一个工具 —— 因为 Action 格式是单行的
  4. 调试友好 —— 你能直接读到 LLM 的"内心独白"
  5. 解析依赖正则 —— 模型输出格式不稳定时容易失败

[Function Calling 的特点]
  1. 推理过程不可见 —— LLM 内部决定调什么工具，你看不到为什么
  2. 依赖模型原生 tool_calls 支持 —— 需要 OpenAI/Anthropic 等特定模型
  3. 可以同时调用多个工具 —— 一次返回多个 tool_calls
  4. 格式可靠 —— JSON 解析比文本解析稳定得多
  5. 更省 token —— 不需要 Thought/Action/Observation 这些格式文本

[什么时候用 ReAct？]
  - 你需要审计/调试 Agent 的决策过程
  - 你用的模型不支持原生 tool_calls（比如开源小模型）
  - 推理过程本身有价值（比如教育场景、解释型 Agent）

[什么时候用 Function Calling？]
  - 模型支持原生 tool_calls（DeepSeek, GPT, Claude 都支持）
  - 追求效率和稳定性
  - 生产环境，不需要暴露推理过程
"""
    )


# ═══════════════════════════════════════════════════════════
# 补充实验：看看解析器选型的影响
# ═══════════════════════════════════════════════════════════


def demo_parser_comparison():
    """对比三种 Action 解析方式的优劣。"""

    print_separator("Bonus: 三种解析器对比")

    from study_agent.agent.react_agent import (
        parse_action_json,
        parse_action_naive,
        parse_action_regex,
    )

    # 构造一些 LLM 可能输出的 Action 文本
    test_cases = [
        # 标准格式
        'Action: search(query="React 19")',
        # 多余空格
        '   Action:  calculator(expression="3*5")  ',
        # 无参数
        "Action: current_time()",
        # 大小写不一致 (naive 会挂，regex 不会)
        "action: search(query=test)",
        # LLM 乱写
        '我想用 search 工具搜一下 "AI Agent"',
        # JSON 格式
        '```json\n{"tool": "search", "params": {"query": "AI"}}\n```',
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n  测试 {i}: {test_input[:60]}...")
        naive = parse_action_naive(test_input)
        regex = parse_action_regex(test_input)
        json_r = parse_action_json(test_input)

        print(f"    naive: {naive}")
        print(f"    regex: {regex}")
        print(f"    json:  {json_r}")


# ═══════════════════════════════════════════════════════════
# LangChain 版 vs 手写版对比
# ═══════════════════════════════════════════════════════════


def demo_langchain_vs_handwritten():
    """用 LangChain 的 create_react_agent 运行同一个任务，和手写版对比。"""

    print_separator("Bonus: LangChain 版 ReAct Agent")

    config = get_config()
    client = LLMClient(provider=config.provider, model=config.model)

    try:
        from study_agent.agent.react_agent import create_langchain_react_agent

        lc_executor = create_langchain_react_agent(client)
        question = "搜索 React 19 的新特性，然后计算 150 减去 20% 等于多少。"

        print(f"[问题] {question}")
        print()

        result = lc_executor.invoke({"input": question})
        print(f"\n[LangChain 版答案] {result['output']}")

    except Exception as e:
        print(f"LangChain 版运行失败: {e}")
        print("[提示] 这个错误是学习的一部分 —— LangChain 的封装层有时比手写代码更难调试")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "parser":
        demo_parser_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "langchain":
        demo_langchain_vs_handwritten()
    else:
        demo_react_vs_fc()
