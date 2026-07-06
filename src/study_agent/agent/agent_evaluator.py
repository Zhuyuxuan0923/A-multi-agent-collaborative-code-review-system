"""Agent 评估框架 —— 定义评测维度、构建测试集、自动打分。

为什么需要评估框架？
  没有评估时：你写了 ReAct 和 Plan-Execute 两个 Agent，
  但没法回答"哪个更好"——凭感觉选 = 玄学，不是工程。
  有评估后：跑同一组测试，每个维度自动打分，量化对比。

三个核心概念（先理解这三个，后面的代码就很好懂）：

  1. 评测维度 (Dimension)：从什么角度打分？
     - 准确性 (Accuracy)：答案对不对？有没有覆盖关键信息？
     - 效率 (Efficiency)：用了多少次 LLM 调用？耗时多久？
     - 工具正确率 (Tool Correctness)：是否选对了工具？

  2. 测试用例 (Test Case)：用什么题来考 Agent？
     每条用例包含：问题 + 期望答案关键词 + 期望用到的工具 + 难度等级

  3. 自动打分 (Scoring)：怎么把"好/坏"量化成分数？
     每个维度返回 0.0~1.0 的分数，最终加权求和得到总分。

评测流程（ASCII 流程图）：

  测试集 (12条用例)
      |
      v
  +---------+     +------------------+
  | ReAct   |     | Plan-Execute     |   <-- 两个 Agent 跑同一组题
  +---------+     +------------------+
      |                  |
      v                  v
  +----------------------------------+
  |        AgentEvaluator            |   <-- 对每份答案打分
  |  - 准确性：关键词命中率          |
  |  - 效率：LLM调用次数 / 耗时      |
  |  - 工具正确率：期望工具命中率    |
  +----------------------------------+
      |
      v
  Markdown 对比报告
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# ========================================================================
# 第 1 部分：数据结构 —— 测试用例长什么样？评分结果长什么样？
# ========================================================================
# 在写评测逻辑之前，先定义"数据长什么样"。
# dataclass  = 一种省代码的数据容器，Python 自动帮你生成 __init__。
# 如果你忘了 dataclass 是什么，回头看 plan_execute_agent.py 里的注释。
# ========================================================================


@dataclass
class AgentTestCase:
    """一条 Agent 评测用例。

    字段说明：
      id              -> 用例编号（如 "E01"），方便在报告里引用
      question        -> 给 Agent 的问题
      difficulty      -> 难度：easy / medium / hard
      category        -> 类别：factual / calculation / multi_step / comparison / time
      expected_keywords -> 好的答案应该包含这些关键词（用于自动评分）
      expected_tools  -> 好的方案应该用到这些工具（用于评估工具选择是否正确）
      min_steps       -> 最少需要几步（用于判断 Agent 是否有合理的执行计划）
      note            -> 出题人的备注（为什么出这道题？考察什么能力？）

    注意：
      expected_keywords 不是唯一正确答案——Agent 可能用不同表述给出正确答案。
      这是一种"近似评估"，比人工逐条看快，但不如人工精准。
      工程上，当测试集足够大时（50+），关键词法的统计结论是可靠的。
    """

    id: str
    question: str
    difficulty: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    min_steps: int = 1
    note: str = ""


@dataclass
class AgentEvalScore:
    """单个维度的评分。

    字段说明：
      dimension  -> 维度名称（accuracy / efficiency / tool_correctness）
      score      -> 实际得分（0.0 ~ max_score）
      max_score  -> 该维度的满分
      details    -> 评分详情（为什么得这个分？方便调试）
    """

    dimension: str
    score: float
    max_score: float
    details: str

    @property
    def percentage(self) -> float:
        """得分百分比。"""
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


@dataclass
class AgentEvalResult:
    """一条用例的完整评测结果。

    字段说明：
      test_case    -> 用的是哪条测试用例
      agent_name   -> 被评测的 Agent 名称（如 "ReAct"）
      answer       -> Agent 给出的最终答案
      scores       -> 各维度评分列表
      total_score  -> 加权总分
      llm_calls    -> LLM 调用次数
      tool_calls   -> 工具调用次数
      duration_ms  -> 执行耗时（毫秒）
      agent_trace  -> Agent 的运行 trace（调试用）
    """

    test_case: AgentTestCase
    agent_name: str
    answer: str
    scores: list[AgentEvalScore] = field(default_factory=list)
    total_score: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    agent_trace: dict[str, Any] = field(default_factory=dict)


# ========================================================================
# 第 2 部分：评分函数 —— 每个维度怎么算分？
# ========================================================================
# 这里把每个维度的评分逻辑拆成独立函数。
# 好处：要加新维度时，只需加一个新函数 + 在 evaluate_one 里调用它。
# ========================================================================


def score_accuracy(answer: str, expected_keywords: list[str]) -> AgentEvalScore:
    """评分维度 1：准确性 —— 答案是否包含期望的关键信息？

    评分算法（简单但实用）：
      - 统计 expected_keywords 中有多少个出现在 answer 中
      - 分数 = 命中数 / 总关键词数
      - 忽略大小写
      - 对于中文关键词，用直接包含匹配；英文用词边界匹配

    为什么用关键词而不是让 LLM 评判？
      - 速度快（毫秒级 vs 秒级）
      - 成本为零（不需要额外的 LLM 调用）
      - 结果可复现（相同的输入永远得到相同的分数）
      - 缺点：Agent 用同义词回答可能被误判（如说了"减速"但关键词是"变慢"）

    工程权衡：测试用例越多，关键词法的统计偏差越小。
    """
    if not expected_keywords:
        return AgentEvalScore(
            dimension="准确性",
            score=0.0,
            max_score=0.0,
            details="没有设置期望关键词，跳过准确性评估",
        )

    answer_lower = answer.lower()
    hit_count = 0
    hit_details: list[str] = []

    for kw in expected_keywords:
        kw_lower = kw.lower()
        # 英文关键词用单词边界匹配（避免 "react" 匹配到 "reactive"）
        if re.search(r"[a-zA-Z]", kw_lower):
            matched = bool(re.search(rf"\b{re.escape(kw_lower)}\b", answer_lower))
        else:
            # 中文关键词用直接包含匹配
            matched = kw_lower in answer_lower

        if matched:
            hit_count += 1
            hit_details.append(f"[命中] {kw}")
        else:
            hit_details.append(f"[丢失] {kw}")

    total = len(expected_keywords)
    score = hit_count / total

    return AgentEvalScore(
        dimension="准确性",
        score=round(score, 2),
        max_score=1.0,
        details=f"命中 {hit_count}/{total} 个关键词: " + "; ".join(hit_details),
    )


def score_tool_usage(tools_used: list[str], expected_tools: list[str]) -> AgentEvalScore:
    """评分维度 2：工具正确率 —— Agent 是否选择了正确的工具？

    评分算法：
      - 统计 expected_tools 中有多少个被 Agent 实际使用了
      - 分数 = 命中数 / 期望工具数
      - 注意：Agent 多用了额外的工具不扣分（只要核心工具用对了）

    这个维度的意义：
      - 选对工具 = Agent 理解了问题的结构
      - 例如"计算 React 19 比 18 快多少%"应该先用 search 再用 calculator
      - 如果 Agent 只用 search 然后让 LLM 心算，工具使用不算最优
    """
    if not expected_tools:
        return AgentEvalScore(
            dimension="工具正确率",
            score=0.0,
            max_score=0.0,
            details="没有设置期望工具，跳过工具评估",
        )

    hit_count = 0
    hit_details: list[str] = []
    tools_used_lower = [t.lower() for t in tools_used]

    for tool in expected_tools:
        tool_lower = tool.lower()
        matched = any(tool_lower in used for used in tools_used_lower)
        if matched:
            hit_count += 1
            hit_details.append(f"[命中] {tool}")
        else:
            hit_details.append(f"[丢失] {tool}")

    total = len(expected_tools)
    score = hit_count / total

    return AgentEvalScore(
        dimension="工具正确率",
        score=round(score, 2),
        max_score=1.0,
        details=f"使用工具: {tools_used}; 命中 {hit_count}/{total}: " + "; ".join(hit_details),
    )


def score_efficiency(
    llm_calls: int,
    tool_calls: int,
    duration_ms: float,
    expected_min_calls: int = 1,
) -> AgentEvalScore:
    """评分维度 3：效率 —— Agent 用了多少资源完成任务？

    评分算法：
      - 基础分 1.0，每超过预期 LLM 调用次数扣 0.15 分
      - 例如：预期最少 2 次 LLM 调用（规划+汇总），实际用了 4 次
              扣 (4-2) * 0.15 = 0.3，得分 0.7

    为什么用 LLM 调用次数而不是耗时？
      - LLM 调用次数 = 成本（每次调用都花钱）
      - 耗时受网络波动影响大，不稳定
      - 但报告里会同时展示耗时作为参考

    这个维度的意义：
      - Plan-Execute 通常比 ReAct 更省 LLM 调用（计划一次，执行多次）
      - 但如果 Replan 过多，Plan-Execute 也会变贵
      - 效率维度的分数直接反映"完成同样任务谁更省钱"
    """
    penalty_per_extra_call = 0.15
    extra_calls = max(0, llm_calls - expected_min_calls)
    penalty = extra_calls * penalty_per_extra_call
    score = max(0.0, 1.0 - penalty)

    details = f"LLM调用 {llm_calls} 次, 工具调用 {tool_calls} 次, " f"耗时 {duration_ms:.0f}ms; "
    if extra_calls > 0:
        details += f"超过预期 {expected_min_calls} 次 LLM 调用, 扣 {penalty:.2f} 分"
    else:
        details += "效率优秀"

    return AgentEvalScore(
        dimension="效率",
        score=round(score, 2),
        max_score=1.0,
        details=details,
    )


# ========================================================================
# 第 3 部分：AgentEvaluator —— 评测引擎核心
# ========================================================================


class AgentEvaluator:
    """Agent 评测引擎 —— 用测试集跑 Agent，自动打分，生成报告。

    使用方式：

        evaluator = AgentEvaluator()
        evaluator.load_test_cases(builtin_cases)

        # 评测 ReAct
        react_results = evaluator.evaluate(react_agent, "ReAct Agent")

        # 评测 Plan-Execute
        pe_results = evaluator.evaluate(pe_agent, "Plan-Execute Agent")

        # 生成对比报告
        report = evaluator.generate_report({
            "ReAct Agent": react_results,
            "Plan-Execute Agent": pe_results,
        })
        print(report)

    构造参数：
      accuracy_weight   -> 准确性在总分中的权重（默认 0.5）
      tool_weight       -> 工具正确率在总分中的权重（默认 0.3）
      efficiency_weight -> 效率在总分中的权重（默认 0.2）
      verbose           -> 是否打印详细日志

    权重的意义：
      不同场景下，各维度的重要性不同。
      - 客服机器人：准确性最重要（0.7），效率无所谓
      - 实时对话系统：效率很重要（0.4），用户不想等
      - 开发辅助工具：工具正确率最关键（0.5），错了可能导致 bug
      - 这里的默认值（5:3:2）是通用比例，适合学习阶段
    """

    def __init__(
        self,
        accuracy_weight: float = 0.5,
        tool_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        verbose: bool = True,
    ):
        # 权重之和应该为 1.0，这里做简单的校验
        total = accuracy_weight + tool_weight + efficiency_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"权重之和应为 1.0，当前为 {total}。"
                f"请调整 accuracy_weight / tool_weight / efficiency_weight。"
            )

        self.accuracy_weight = accuracy_weight
        self.tool_weight = tool_weight
        self.efficiency_weight = efficiency_weight
        self.verbose = verbose

        self.test_cases: list[AgentTestCase] = []

    # ── 测试集管理 ──────────────────────────────────────────

    def load_test_cases(self, cases: list[AgentTestCase]) -> None:
        """加载测试用例集。"""
        self.test_cases = cases
        if self.verbose:
            print(f"[Evaluator] 加载了 {len(cases)} 条测试用例")
            by_difficulty: dict[str, int] = {}
            for c in cases:
                by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
            for diff, count in sorted(by_difficulty.items()):
                print(f"  {diff}: {count} 条")

    def add_test_case(self, case: AgentTestCase) -> None:
        """添加单条测试用例。"""
        self.test_cases.append(case)

    # ── 核心评测逻辑 ────────────────────────────────────────

    def evaluate(
        self,
        agent: Any,
        agent_name: str,
    ) -> list[AgentEvalResult]:
        """用全部测试用例评测一个 Agent，返回每条用例的结果列表。

        参数：
          agent      -> 一个 Agent 实例（必须有 .run(question) -> str 方法）
          agent_name -> Agent 的名字（用于报告中显示）

        评测过程：
          for each test_case:
              1. 记录开始时间
              2. 调用 agent.run(question)
              3. 记录结束时间
              4. 从 agent trace 中提取 LLM 调用次数和工具调用列表
              5. 三个维度分别打分
              6. 计算加权总分
        """
        if not self.test_cases:
            raise ValueError("没有加载测试用例。请先调用 load_test_cases()。")

        results: list[AgentEvalResult] = []

        for i, case in enumerate(self.test_cases, 1):
            if self.verbose:
                print(
                    f"\n[{i}/{len(self.test_cases)}] "
                    f"{case.id} [{case.difficulty}] {case.question[:60]}..."
                )

            # 步骤 1+2: 计时执行
            t0 = time.time()
            try:
                answer = agent.run(case.question)
            except Exception as e:
                answer = f"[ERR] Agent 运行异常: {type(e).__name__}: {e}"
            duration_ms = (time.time() - t0) * 1000

            # 步骤 3: 从 agent trace 提取调试信息
            agent_trace = self._extract_trace(agent)
            llm_calls = self._estimate_llm_calls(agent_trace, agent_name)
            tools_used = self._extract_tools_used(agent_trace)

            if self.verbose:
                print(f"  答案({len(answer)}字): {answer[:100]}...")
                print(
                    f"  耗时: {duration_ms:.0f}ms, LLM调用: {llm_calls}, "
                    f"工具调用: {len(tools_used)}"
                )

            # 步骤 4: 三维度打分
            accuracy = score_accuracy(answer, case.expected_keywords)
            tool_score = score_tool_usage(tools_used, case.expected_tools)
            efficiency = score_efficiency(llm_calls, len(tools_used), duration_ms, case.min_steps)

            # 步骤 5: 加权总分
            total = (
                accuracy.score * self.accuracy_weight
                + tool_score.score * self.tool_weight
                + efficiency.score * self.efficiency_weight
            )

            result = AgentEvalResult(
                test_case=case,
                agent_name=agent_name,
                answer=answer,
                scores=[accuracy, tool_score, efficiency],
                total_score=round(total, 2),
                llm_calls=llm_calls,
                tool_calls=len(tools_used),
                duration_ms=duration_ms,
                agent_trace=agent_trace,
            )
            results.append(result)

        if self.verbose:
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_accuracy = sum(r.scores[0].score for r in results) / len(results)
            print(f"\n{'=' * 50}")
            print(
                f"[{agent_name}] 评测完成: "
                f"平均总分 {avg_total:.2f}, 平均准确性 {avg_accuracy:.2%}"
            )

        return results

    # ── 报告生成 ────────────────────────────────────────────

    def generate_report(
        self,
        all_results: dict[str, list[AgentEvalResult]],
    ) -> str:
        """生成 Markdown 格式的对比评测报告。

        参数：
          all_results -> {"Agent名称": [AgentEvalResult, ...], ...}

        报告结构：
          一、评测概览（Agent 数量、用例数、权重配置）
          二、综合得分对比（总分、各维度平均分）
          三、各用例详细对比（逐题对比答案和得分）
          四、按类别/难度分析（哪类题谁更强？）
          五、选型建议
        """
        if not all_results:
            return "没有评测数据，请先调用 evaluate()。"

        lines: list[str] = [
            "# Agent 评测对比报告",
            "",
            f"评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"参评 Agent 数: {len(all_results)}",
            f"测试用例数: {len(self.test_cases)}",
            "",
            f"评分权重: 准确性={self.accuracy_weight}, "
            f"工具正确率={self.tool_weight}, 效率={self.efficiency_weight}",
            "",
            "---",
            "",
        ]

        # ── 一、综合得分对比 ──
        lines += self._section_overview(all_results)

        # ── 二、各维度详细分析 ──
        lines += self._section_dimension_detail(all_results)

        # ── 三、逐用例对比 ──
        lines += self._section_per_case(all_results)

        # ── 四、按类别分析 ──
        lines += self._section_by_category(all_results)

        # ── 五、按难度分析 ──
        lines += self._section_by_difficulty(all_results)

        # ── 六、选型建议 ──
        lines += self._section_recommendations(all_results)

        return "\n".join(lines)

    # ── 报告各部分 ──────────────────────────────────────────

    def _section_overview(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """生成综合得分对比表。"""
        lines = [
            "## 一、综合得分对比",
            "",
            "| Agent | 总分 | 准确性 | 工具正确率 | 效率 | " "平均LLM调用 | 平均耗时 |",
            "|-------|------|--------|-----------|------|" "------------|---------|",
        ]

        for name, results in all_results.items():
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_acc = sum(r.scores[0].score for r in results) / len(results)
            avg_tool = sum(r.scores[1].score for r in results) / len(results)
            avg_eff = sum(r.scores[2].score for r in results) / len(results)
            avg_llm = sum(r.llm_calls for r in results) / len(results)
            avg_time = sum(r.duration_ms for r in results) / len(results)

            lines.append(
                f"| **{name}** | {avg_total:.2f} | {avg_acc:.0%} | "
                f"{avg_tool:.0%} | {avg_eff:.0%} | "
                f"{avg_llm:.1f} | {avg_time:.0f}ms |"
            )

        # 找出各维度最优
        best_total_name = max(
            all_results,
            key=lambda n: sum(r.total_score for r in all_results[n]) / len(all_results[n]),
        )
        best_acc_name = max(
            all_results,
            key=lambda n: sum(r.scores[0].score for r in all_results[n]) / len(all_results[n]),
        )
        best_eff_name = max(
            all_results,
            key=lambda n: sum(r.scores[2].score for r in all_results[n]) / len(all_results[n]),
        )

        lines += [
            "",
            f"- 综合最优: **{best_total_name}**",
            f"- 准确性最优: **{best_acc_name}**",
            f"- 效率最优: **{best_eff_name}**",
            "",
            "---",
            "",
        ]
        return lines

    def _section_dimension_detail(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """生成各维度详细的逐用例得分表。"""
        lines = [
            "## 二、各维度详细得分",
            "",
        ]

        for dimension_name, idx in [("准确性", 0), ("工具正确率", 1), ("效率", 2)]:
            lines.append(f"### {dimension_name}")
            lines.append("")

            # 表头
            case_ids = [c.id for c in self.test_cases]
            header = "| Agent | " + " | ".join(case_ids) + " | 平均 |"
            sep = "|-------|" + "|".join(["-" * 6 for _ in case_ids]) + "|------|"
            lines.append(header)
            lines.append(sep)

            for name, results in all_results.items():
                scores = [f"{r.scores[idx].percentage:.0f}%" for r in results]
                avg = sum(r.scores[idx].score for r in results) / len(results)
                row = f"| {name} | " + " | ".join(scores) + f" | {avg:.0%} |"
                lines.append(row)

            lines.append("")

        lines += ["---", ""]
        return lines

    def _section_per_case(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """生成逐用例对比（只展示关键信息，不全量输出答案）。"""
        lines = [
            "## 三、各用例对比详情",
            "",
        ]

        for i, case in enumerate(self.test_cases):
            lines.append(f"### {case.id}: {case.question}")
            lines.append(f"  难度: {case.difficulty} | 类别: {case.category}")
            if case.note:
                lines.append(f"  考察点: {case.note}")
            lines.append("")

            # 对比表
            lines.append("| Agent | 总分 | 准确性 | 工具 | 效率 | " "LLM调用 | 耗时 | 关键发现 |")
            lines.append("|-------|------|--------|------|------|" "---------|------|---------|")

            for name, results in all_results.items():
                r = results[i]
                accuracy_pct = r.scores[0].percentage
                tool_pct = r.scores[1].percentage
                eff_pct = r.scores[2].percentage

                # 提取关键发现：丢了哪些关键词
                acc_detail = r.scores[0].details
                lost_kw = [
                    kw.split("] ")[1] for kw in acc_detail.split("; ") if kw.startswith("[丢失]")
                ]
                finding = f"丢失关键词: {', '.join(lost_kw)}" if lost_kw else "全部命中"

                lines.append(
                    f"| {name} | {r.total_score:.2f} | {accuracy_pct:.0f}% | "
                    f"{tool_pct:.0f}% | {eff_pct:.0f}% | "
                    f"{r.llm_calls} | {r.duration_ms:.0f}ms | {finding} |"
                )

            lines.append("")

        lines += ["---", ""]
        return lines

    def _section_by_category(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """按类别分析：哪个 Agent 在什么类型的任务上更强？"""
        # 收集所有类别
        categories = list(dict.fromkeys(c.category for c in self.test_cases))

        lines = [
            "## 四、按任务类别分析",
            "",
            "| 类别 | " + " | ".join(all_results.keys()) + " | 最优 |",
            "|------|" + "|".join(["-" * 8 for _ in all_results]) + "|------|",
        ]

        for cat in categories:
            cat_results: dict[str, float] = {}
            for name, results in all_results.items():
                cat_scores = [r.total_score for r in results if r.test_case.category == cat]
                cat_results[name] = sum(cat_scores) / len(cat_scores) if cat_scores else 0

            best = max(cat_results, key=cat_results.get)  # type: ignore[arg-type]
            scores_str = " | ".join(f"{cat_results[n]:.2f}" for n in all_results)
            lines.append(f"| {cat} | {scores_str} | **{best}** |")

        lines += ["", "---", ""]
        return lines

    def _section_by_difficulty(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """按难度分析：哪个 Agent 在难题上更有优势？"""
        difficulties = ["easy", "medium", "hard"]

        lines = [
            "## 五、按难度分析",
            "",
            "| 难度 | " + " | ".join(all_results.keys()) + " | 最优 |",
            "|------|" + "|".join(["-" * 8 for _ in all_results]) + "|------|",
        ]

        for diff in difficulties:
            diff_results: dict[str, float] = {}
            for name, results in all_results.items():
                diff_scores = [r.total_score for r in results if r.test_case.difficulty == diff]
                diff_results[name] = sum(diff_scores) / len(diff_scores) if diff_scores else 0

            if all(v == 0 for v in diff_results.values()):
                continue

            best = max(diff_results, key=diff_results.get)  # type: ignore[arg-type]
            scores_str = " | ".join(f"{diff_results[n]:.2f}" for n in all_results)
            lines.append(f"| {diff} | {scores_str} | **{best}** |")

        lines += ["", "---", ""]
        return lines

    def _section_recommendations(self, all_results: dict[str, list[AgentEvalResult]]) -> list[str]:
        """生成选型建议 —— 基于评测数据的推荐。"""
        lines = [
            "## 六、选型建议",
            "",
        ]

        # 计算各 Agent 的关键指标
        for name, results in all_results.items():
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_llm = sum(r.llm_calls for r in results) / len(results)
            avg_time = sum(r.duration_ms for r in results) / len(results)

            # 按类别分
            by_cat: dict[str, float] = {}
            for r in results:
                cat = r.test_case.category
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(r.total_score)
            strengths = sorted(by_cat.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- 综合得分: {avg_total:.2f}")
            lines.append(f"- 平均 LLM 调用: {avg_llm:.1f} 次")
            lines.append(f"- 平均耗时: {avg_time:.0f}ms")
            if strengths:
                lines.append(
                    f"- 最擅长的任务类型: **{strengths[0][0]}** "
                    f"(平均分 {sum(strengths[0][1])/len(strengths[0][1]):.2f})"
                )
            lines.append("")

        lines += [
            "### 综合建议",
            "",
            "- **事实查询类任务**: 两种 Agent 表现相当，选择更省成本的即可",
            "- **多步骤推理任务**: 观察哪个 Agent 的工具选择更准确",
            "- **计算密集型任务**: Plan-Execute 通常更高效（计划一次，批量执行）",
            "- **探索性任务**: ReAct 更灵活（每步根据上一步结果调整方向）",
            "",
            "> 注意：本报告基于模拟工具（mock_search）的测试结果。",
            "> 在真实场景中（真实搜索 API、更大知识库），",
            "> Agent 行为可能不同，建议用真实数据重新评测。",
            "",
            "---",
            "",
            f"*报告由 AgentEvaluator 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        ]
        return lines

    # ── Trace 提取辅助方法 ──────────────────────────────────

    @staticmethod
    def _extract_trace(agent: Any) -> dict[str, Any]:
        """从 Agent 实例中提取运行 trace。

        不同 Agent 的 trace 格式不同：
          - ReactAgent: get_trace() 返回 {"system_prompt": ...}
                       last_round_count 属性记录最终轮数
          - PlanExecuteAgent: last_trace 属性，包含 phases
        """
        trace: dict[str, Any] = {}

        # 尝试获取轮数（ReactAgent）
        if hasattr(agent, "last_round_count"):
            trace["last_round_count"] = agent.last_round_count

        # 尝试 get_trace() 方法
        if hasattr(agent, "get_trace"):
            try:
                t = agent.get_trace()
                if isinstance(t, dict):
                    trace.update(t)
            except Exception:
                pass

        # 尝试 last_trace 属性（PlanExecuteAgent）
        if hasattr(agent, "last_trace"):
            try:
                plan_trace = agent.last_trace
                if isinstance(plan_trace, dict):
                    trace.update(plan_trace)
            except Exception:
                pass

        return trace

    @staticmethod
    def _estimate_llm_calls(trace: dict[str, Any], agent_name: str) -> int:
        """从 trace 中估算 LLM 调用次数。

        ReactAgent: 每轮 ReAct 循环 = 1 次 LLM 调用
                   优先使用 agent.last_round_count，降级到经验估算
        PlanExecuteAgent: planning(1) + replan(每次1) + final_summary(1)
        """
        # ReactAgent 的轮数（如果有的话）
        last_round_count = trace.get("last_round_count")
        if last_round_count is not None and isinstance(last_round_count, int):
            return last_round_count

        # PlanExecuteAgent 的 trace 包含 phases
        phases = trace.get("phases", [])
        if phases:
            calls = 1  # 初始 planning
            for phase in phases:
                if phase.get("phase") == "execution":
                    calls += phase.get("replan_count", 0)
            calls += 1  # final summary
            return calls

        # 降级策略：基于经验估算
        return 3  # 默认估算

    @staticmethod
    def _extract_tools_used(trace: dict[str, Any]) -> list[str]:
        """从 trace 中提取实际使用的工具列表。"""
        tools: list[str] = []

        phases = trace.get("phases", [])
        for phase in phases:
            if phase.get("phase") == "execution":
                results = phase.get("results", [])
                for r in results:
                    step = r.get("step") if isinstance(r, dict) else getattr(r, "step", None)
                    if step:
                        tool = (
                            step.get("tool")
                            if isinstance(step, dict)
                            else getattr(step, "tool", "")
                        )
                        if tool and tool not in ("final_summary", "none", ""):
                            tools.append(tool)

        return list(dict.fromkeys(tools))  # 去重，保持顺序
