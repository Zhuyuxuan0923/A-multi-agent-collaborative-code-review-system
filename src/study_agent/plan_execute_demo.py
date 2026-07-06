"""Plan-Execute vs ReAct 对比演示。

用同一个任务，分别通过 PlanExecuteAgent 和 ReactAgent 来执行，
对比两种范式的规划方式、执行效率和成本差异。

这是 Week 5 Day 3 的核心实验。
"""

from __future__ import annotations

import logging
import sys
import time

from study_agent.config.settings import get_config
from study_agent.llm.client import LLMClient

logging.basicConfig(
    level=logging.WARNING,  # 降低日志噪音，只显示 WARNING 以上
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def print_section(title: str) -> None:
    """打印章节分隔。"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_plan_execute_basic():
    """演示 1: Plan-Execute 基础流程。

    用一个结构化问题展示 Plan-Execute 的完整流程：
      规划 -> 逐步执行 -> 校验 -> 汇总
    """
    from study_agent.agent.plan_execute_agent import PlanExecuteAgent

    print_section("演示 1: Plan-Execute 基础流程")

    config = get_config()
    client = LLMClient(provider=config.provider, model=config.model)

    agent = PlanExecuteAgent(client, max_replans=2, verbose=True)

    # 一个需要多步骤的结构化问题
    # 预期计划: 搜索 -> 搜索 -> 计算 -> 汇总
    question = (
        "请帮我做三件事：\n"
        "1. 搜索 React 19 的新特性\n"
        "2. 搜索 AI Agent 的架构\n"
        "3. 计算 (150 + 250) * 0.8 的结果\n"
        "然后告诉我这三件事的结果。"
    )

    print(f"\n[问题] {question}")

    start = time.time()
    answer = agent.run(question)
    elapsed = time.time() - start

    print(f"\n[最终答案]\n{answer}")
    print(f"\n[耗时] {elapsed:.1f} 秒")


def demo_plan_execute_replan():
    """演示 2: Plan-Execute 的 Replan 机制。

    故意制造一个会失败的场景：
      LLM 计划了一个工具但用了错误的参数 -> 步骤失败 -> 触发 replan

    注意：这个演示依赖于 LLM 实际输出的计划，
    不保证一定触发 replan（取决于 LLM 规划的质量）。
    """
    from study_agent.agent.plan_execute_agent import PlanExecuteAgent

    print_section("演示 2: Plan-Execute — Replan 机制")

    config = get_config()
    client = LLMClient(provider=config.provider, model=config.model)

    agent = PlanExecuteAgent(client, max_replans=2, verbose=True)

    # 使用一个较为模糊的问题 —— LLM 可能规划不完美
    question = "我想了解 LangChain 的 Agent 模块，请搜索相关信息并用中文总结关键点。"

    print(f"\n[问题] {question}")

    answer = agent.run(question)

    print(f"\n[最终答案]\n{answer}")

    # 打印 trace 信息
    trace = agent.last_trace
    for phase in trace.get("phases", []):
        if phase["phase"] == "execution":
            replan_count = phase.get("replan_count", 0)
            print(f"\n[Replan 统计] 共触发 {replan_count} 次 replan")
            results = phase.get("results", [])
            success_count = sum(1 for r in results if r.success)
            print(f"[步骤统计] {success_count}/{len(results)} 个步骤成功")


def demo_comparison():
    """演示 3: ReAct vs Plan-Execute 对比实验。

    用同一个问题分别跑两个 Agent，对比：
      - 执行方式（ReAct 逐步思考 vs Plan-Execute 先计划后执行）
      - 效率（LLM 调用次数、耗时）

    注意：为了避免输出过长，这里关闭 verbose 模式，
    只比较最终结果和执行统计。
    """
    from study_agent.agent.plan_execute_agent import PlanExecuteAgent
    from study_agent.agent.react_agent import ReactAgent

    print_section("演示 3: ReAct vs Plan-Execute 对比")

    config = get_config()
    client_via_config = LLMClient(provider=config.provider, model=config.model)

    question = (
        "搜索 FastAPI 的信息，计算 2025 - 2018（FastAPI 从发布到现在多少年了），然后总结搜索结果。"
    )

    print(f"\n[问题] {question}")

    # ── ReAct ──
    print("\n--- ReAct Agent ---")
    react_agent = ReactAgent(client_via_config, max_rounds=6, parse_method="regex")
    # 复制一个 client 避免状态污染
    client_for_react = LLMClient(provider=config.provider, model=config.model)

    start = time.time()
    react_answer = react_agent.run(question)
    react_time = time.time() - start

    print(f"\n[ReAct 答案] {react_answer[:200]}...")
    print(f"[ReAct 耗时] {react_time:.1f} 秒")

    # ── Plan-Execute ──
    print("\n--- PlanExecute Agent ---")
    client_for_pe = LLMClient(provider=config.provider, model=config.model)
    pe_agent = PlanExecuteAgent(client_for_pe, max_replans=2, verbose=False)

    start = time.time()
    pe_answer = pe_agent.run(question)
    pe_time = time.time() - start

    print(f"\n[Plan-Execute 答案] {pe_answer[:200]}...")
    print(f"[Plan-Execute 耗时] {pe_time:.1f} 秒")

    # ── 对比总结 ──
    print("\n--- 对比总结 ---")
    print(f"  ReAct:         耗时 {react_time:.1f}s")
    print(f"  Plan-Execute:  耗时 {pe_time:.1f}s")
    print()
    print("  关键差异：")
    print("    - ReAct 每步都要 LLM 思考，更灵活但更慢")
    print("    - Plan-Execute 一次性规划，机械执行，更高效但刚性")
    print("    - 对于步骤可预知的任务，Plan-Execute 更省 token")
    print("    - 对于探索性任务，ReAct 的动态决策更有优势")


def demo_plan_failure():
    """演示 4: 计划失败时的降级策略。

    当 LLM 输出的 JSON 无法解析时，PlanExecuteAgent 会
    自动降级为一个单步骤计划（用搜索工具直接搜问题）。

    这个演示中我们模拟一个"LLM 输出格式错误"的场景。
    """
    from study_agent.agent.plan_execute_agent import PlanExecuteAgent

    print_section("演示 4: 计划解析的容错机制")

    config = get_config()
    client = LLMClient(provider=config.provider, model=config.model)

    agent = PlanExecuteAgent(client, verbose=True)

    question = "什么是 Python 的 GIL？"

    print(f"\n[问题] {question}")
    print("\n（观察 Phase 1 中 LLM 输出的原始 JSON 计划）")

    answer = agent.run(question)
    print(f"\n[最终答案]\n{answer}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plan-Execute Agent 演示脚本")
    parser.add_argument(
        "demo",
        nargs="?",
        default="basic",
        choices=["basic", "replan", "compare", "fail"],
        help="选择演示: basic(基础流程), replan(重规划), compare(对比ReAct), fail(容错)",
    )
    args = parser.parse_args()

    demos = {
        "basic": demo_plan_execute_basic,
        "replan": demo_plan_execute_replan,
        "compare": demo_comparison,
        "fail": demo_plan_failure,
    }

    print("Plan-Execute Agent 演示")
    print(f"Provider: {get_config().provider}, Model: {get_config().model}")

    demos[args.demo]()
