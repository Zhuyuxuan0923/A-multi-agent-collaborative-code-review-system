"""Agent 评测框架演示脚本。

用 AgentEvaluator 对比评测 ReAct Agent 和 Plan-Execute Agent，
在同一组测试用例上跑，自动打分，生成 Markdown 对比报告。

运行方式：
  python -m study_agent.agent_evaluation_demo          # 快速模式（4条用例）
  python -m study_agent.agent_evaluation_demo --full   # 完整模式（12条用例）
  python -m study_agent.agent_evaluation_demo --dry-run # 干跑模式（不调LLM，演示框架结构）

学习目标：
  1. 理解"评估维度"是什么，为什么不能只看"答案对不对"
  2. 看懂评测报告的各项指标
  3. 能自己添加测试用例来评测你编写的 Agent
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ========================================================================
# 快速模式 vs 完整模式
# ========================================================================
# 快速模式：用 4 条精选用例（覆盖 easy/medium/hard），约 8-12 次 LLM 调用
# 完整模式：用全部 12 条用例，约 24-36 次 LLM 调用
# 干跑模式：不调 LLM，验证框架代码能正常运行
# ========================================================================

QUICK_MODE_IDS = {"E01", "E03", "E07", "E09"}


def print_section(title: str) -> None:
    """打印章节分隔线。"""
    print()
    print("=" * 58)
    print(f"  {title}")
    print("=" * 58)


# ========================================================================
# 核心演示函数
# ========================================================================


def run_evaluation(full: bool = False) -> str:
    """执行 Agent 评测，返回报告文件路径。

    参数：
      full -> True 用全部 12 条用例，False 用 4 条快速用例
    """
    from study_agent.agent.agent_eval_cases import BUILTIN_TEST_CASES, get_case_summary
    from study_agent.agent.agent_evaluator import AgentEvaluator
    from study_agent.agent.plan_execute_agent import PlanExecuteAgent
    from study_agent.agent.react_agent import ReactAgent
    from study_agent.config.settings import get_config
    from study_agent.llm.client import LLMClient

    # ── 1. 准备测试用例 ────────────────────────────────────
    if full:
        cases = BUILTIN_TEST_CASES
        mode_name = "完整模式 (12 条用例)"
    else:
        cases = [c for c in BUILTIN_TEST_CASES if c.id in QUICK_MODE_IDS]
        mode_name = "快速模式 (4 条用例)"

    print_section(f"Agent 评测框架演示 —— {mode_name}")
    print()
    print(get_case_summary() if full else _quick_summary(cases))

    # ── 2. 初始化评测引擎 ──────────────────────────────────
    # 权重说明：
    #   准确性 50% —— 答案有没有覆盖关键信息？这是最重要的
    #   工具正确率 30% —— Agent 有没有选对工具？如果你在开发工具型Agent，这个很重要
    #   效率 20% —— 用了多少次 LLM 调用？在生产环境中成本很关键
    evaluator = AgentEvaluator(
        accuracy_weight=0.5,
        tool_weight=0.3,
        efficiency_weight=0.2,
        verbose=True,
    )
    evaluator.load_test_cases(cases)

    # ── 3. 创建 Agent ──────────────────────────────────────
    config = get_config()
    print(f"\nLLM Provider: {config.provider}, Model: {config.model}")

    # ReAct Agent —— 用 regex 解析（比 naive 更健壮）
    react_agent = ReactAgent(
        LLMClient(provider=config.provider, model=config.model),
        max_rounds=8,
        parse_method="regex",
    )

    # Plan-Execute Agent —— 最多 2 次 replan
    pe_agent = PlanExecuteAgent(
        LLMClient(provider=config.provider, model=config.model),
        max_replans=2,
        verbose=False,  # 评测时关闭详细日志，避免输出太乱
    )

    # ── 4. 执行评测 ────────────────────────────────────────
    print_section("开始评测")

    all_results: dict[str, list] = {}

    # 评测 ReAct
    print("\n>>> 评测 ReAct Agent...")
    t0 = time.time()
    react_results = evaluator.evaluate(react_agent, "ReAct Agent")
    print(f"ReAct 评测完成, 耗时 {time.time() - t0:.1f}s")

    # 评测 Plan-Execute
    print("\n>>> 评测 Plan-Execute Agent...")
    t0 = time.time()
    pe_results = evaluator.evaluate(pe_agent, "Plan-Execute Agent")
    print(f"Plan-Execute 评测完成, 耗时 {time.time() - t0:.1f}s")

    all_results["ReAct Agent"] = react_results
    all_results["Plan-Execute Agent"] = pe_results

    # ── 5. 生成报告 ────────────────────────────────────────
    print_section("生成评测报告")
    report = evaluator.generate_report(all_results)
    print(report)

    # ── 6. 保存报告 ────────────────────────────────────────
    output_dir = Path("docs/week6")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "agent_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {report_path}")

    return str(report_path)


def dry_run_demo():
    """干跑演示：不调用 LLM，展示评测框架的结构和工作原理。

    这个模式适合：
      - 第一次运行时先了解框架结构
      - API key 不可用时验证代码能正常运行
      - 快速理解评分逻辑
    """
    from study_agent.agent.agent_eval_cases import BUILTIN_TEST_CASES, get_case_summary

    print_section("Agent 评测框架 —— 干跑演示")
    print()
    print("干跑模式：不调用 LLM，用模拟数据展示评测框架的结构和评分逻辑。")
    print()

    # ── 1. 展示测试用例 ──
    print("一、测试用例一览")
    print()
    print(get_case_summary())
    print()

    for case in BUILTIN_TEST_CASES:
        print(f"  {case.id} [{case.difficulty}/{case.category}] {case.question}")
        print(f"      期望关键词: {case.expected_keywords}")
        print(f"      期望工具: {case.expected_tools}")
        if case.note:
            print(f"      考察点: {case.note}")
    print()

    # ── 2. 展示评分逻辑（用模拟数据） ──
    print("二、评分维度演示")
    print()

    from study_agent.agent.agent_evaluator import (
        score_accuracy,
        score_efficiency,
        score_tool_usage,
    )

    # 模拟一条好的答案和一条差的答案
    good_answer = (
        "React 19 于 2024 年 12 月发布，主要新特性包括 Server Components 稳定版、"
        "Actions 机制、Document Metadata 原生支持等。"
        "客户端 bundle 比 React 18 减小了约 20%。"
    )
    bad_answer = "React 19 是一个前端框架。"

    keywords = ["2024", "12月", "Server Component", "Actions", "20%"]

    print("  场景 A: 好答案 vs 场景 B: 差答案")
    print(f"  关键词: {keywords}")
    print()

    score_good = score_accuracy(good_answer, keywords)
    score_bad = score_accuracy(bad_answer, keywords)

    print(f"  好答案: 得分 {score_good.score}, {score_good.details}")
    print(f"  差答案: 得分 {score_bad.score}, {score_bad.details}")
    print()

    # 工具评估演示
    print("  工具正确率演示:")
    tools_used_good = ["search", "summarize"]
    tools_used_bad = ["calculator"]
    expected = ["search", "summarize"]

    score_tool_good = score_tool_usage(tools_used_good, expected)
    score_tool_bad = score_tool_usage(tools_used_bad, expected)

    print(f"  选对工具: 得分 {score_tool_good.score}, {score_tool_good.details}")
    print(f"  选错工具: 得分 {score_tool_bad.score}, {score_tool_bad.details}")
    print()

    # 效率评估演示
    print("  效率演示:")
    score_fast = score_efficiency(llm_calls=2, tool_calls=2, duration_ms=1500, expected_min_calls=2)
    score_slow = score_efficiency(llm_calls=8, tool_calls=6, duration_ms=8000, expected_min_calls=2)
    print(f"  高效: 得分 {score_fast.score}, {score_fast.details}")
    print(f"  低效: 得分 {score_slow.score}, {score_slow.details}")
    print()

    # ── 3. 展示报告结构 ──
    print("三、评测报告结构")
    print()
    print("  Agent 评测对比报告")
    print("  ├── 评测概览（Agent 数量、用例数、权重配置）")
    print("  ├── 一、综合得分对比")
    print("  ├── 二、各维度详细得分")
    print("  ├── 三、各用例对比详情")
    print("  ├── 四、按任务类别分析")
    print("  ├── 五、按难度分析")
    print("  └── 六、选型建议")
    print()

    # ── 4. 添加自定义用例的模板 ──
    print("四、如何添加你自己的测试用例？")
    print()
    print(
        """
  from study_agent.agent.agent_evaluator import AgentTestCase

  # 创建一条自定义用例
  my_case = AgentTestCase(
      id="M01",                          # 用例编号
      question="你的问题...",             # 给Agent的问题
      difficulty="medium",               # easy / medium / hard
      category="multi_step",             # factual / calculation / multi_step / comparison / time
      expected_keywords=["关键词1", "关键词2"],  # 答案应该包含的关键词
      expected_tools=["search", "calculator"],  # 预期使用的工具
      min_steps=2,                       # 预期最少步骤数
      note="这道题考察xxx能力",            # 出题备注
  )

  # 加到评测中
  evaluator.add_test_case(my_case)
"""
    )

    print_section("干跑演示完成")
    print()
    print("[提示] 运行 python -m study_agent.agent_evaluation_demo 开始真实评测")
    print("[提示] 运行 python -m study_agent.agent_evaluation_demo --full 完整评测(12题)")


# ========================================================================
# 辅助函数
# ========================================================================


def _quick_summary(cases) -> str:
    """生成快速模式的用例概览。"""
    lines = [f"快速评测用例 ({len(cases)} 条):", ""]
    for c in cases:
        lines.append(f"  {c.id} [{c.difficulty}/{c.category}] {c.question[:50]}...")
    return "\n".join(lines)


# ========================================================================
# 入口
# ========================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent 评测框架演示")
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整模式：使用全部 12 条测试用例（否则用 4 条快速模式）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：不调用 LLM，展示框架结构和工作原理",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run_demo()
    else:
        report_path = run_evaluation(full=args.full)
        print(f"\n[OK] 评测完成! 报告: {report_path}")
