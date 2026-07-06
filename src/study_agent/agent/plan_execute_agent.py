"""Plan-Execute Agent -- "先计划，再执行，出错就调整"的 Agent 范式。

和 ReAct 的核心区别（一个生活类比）：
  ReAct    = 走迷宫时每到一个路口都停下来想"接下来往哪走"
  Plan-Execute = 先在入口把整条路线画好，一口气走完，走错了就重新画后半段

为什么需要 Plan-Execute？
  问题：ReAct 每一步都要一次 LLM 调用（Thought -> Action），LLM 调用 = 钱 + 时间。
       如果任务需要 5 步，ReAct 至少要 5 次 LLM 调用。
  解决：Plan-Execute 把"决策"集中到一次 LLM 调用（生成计划），
        步骤执行不需要 LLM 参与（直接用代码执行工具），
        只在步骤失败时才重新调用 LLM（replan）。

Plan-Execute 的四个阶段：
  1. Planning:    LLM 分析问题 -> 输出 JSON 格式的执行计划
  2. Execution:   按顺序执行每个步骤，代码直接调工具（不需要 LLM）
  3. Validation:  每步执行后校验结果（非空？无错误？有关键词？）
  4. Replanning:  某步校验失败时，把当前进度 + 失败信息发给 LLM，
                  让 LLM 重新规划剩余步骤

适合什么场景？
  - 步骤可预知的结构化任务（如"搜索 X -> 计算 Y -> 总结"）
  - 需要控制成本的场景（减少 LLM 调用次数）
  - 需要审计执行计划的场景（计划本身就是可审查的文档）

不适合什么场景？
  - 探索性任务（你不知道需要几步才能找到答案）
  - 每步结果都会极大改变方向的任务（ReAct 的灵活性更有价值）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from study_agent.agent.research_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ========================================================================
# 数据结构：用 dataclass 定义 Plan 和 Step
# ========================================================================
# dataclass 是什么？
#   它是 Python 的一种"数据容器"。你只需要声明字段和类型，
#   Python 自动帮你生成 __init__、__repr__ 等方法。
#   不用手写 self.x = x 这样的重复代码。
#
# 对比普通 class：
#   class PlanStep:                    @dataclass
#       def __init__(self, ...):       class PlanStep:
#           self.id = id                   id: int
#           self.description = desc        description: str
#           ...                            ...
#   两种写法等价，但 dataclass 省掉 80% 的样板代码。
# ========================================================================


@dataclass
class PlanStep:
    """计划中的一个步骤。

    字段说明：
      id              -> 步骤编号（1, 2, 3...）
      description     -> 这一步要做什么（给人看的描述）
      tool            -> 要用的工具名，必须是 TOOL_REGISTRY 中注册的
      tool_params     -> 传给工具的参数（dict）
      expected_result -> 期望得到什么结果（用于校验，可选）
    """

    id: int
    description: str
    tool: str
    tool_params: dict[str, str] = field(default_factory=dict)
    expected_result: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> PlanStep:
        """从 dict 创建 PlanStep —— 容错处理缺失字段。"""
        return cls(
            id=data.get("id", data.get("step_id", 0)),
            description=data.get("description", ""),
            tool=data.get("tool", ""),
            tool_params=data.get("tool_params", data.get("params", {})),
            expected_result=data.get("expected_result", ""),
        )


@dataclass
class Plan:
    """完整的执行计划。

    字段说明：
      goal  -> 用户问题的简要描述
      steps -> 步骤列表（按顺序执行）
    """

    goal: str
    steps: list[PlanStep]

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        """从 dict 创建 Plan —— 容错处理缺失字段。"""
        goal = data.get("goal", "")
        steps_data = data.get("steps", [])
        steps = [PlanStep.from_dict(s) for s in steps_data]
        return cls(goal=goal, steps=steps)


@dataclass
class StepResult:
    """单个步骤的执行结果。"""

    step: PlanStep
    success: bool
    output: str
    validation_note: str = ""


# ========================================================================
# Plan Prompt 模板 —— 让 LLM 输出 JSON 计划
# ========================================================================

PLAN_SYSTEM_PROMPT = """你是一个任务规划专家。你的职责是分析用户的问题，然后制定一个结构化的执行计划。

你可以使用以下工具：

{tool_descriptions}

你需要输出一个 JSON 格式的执行计划。JSON 的结构必须是：

{{
  "goal": "用户问题的简要描述（一句话）",
  "steps": [
    {{
      "id": 1,
      "description": "这一步要做什么（给人看的）",
      "tool": "工具名（必须是上面列出的之一）",
      "tool_params": {{"参数名": "参数值"}},
      "expected_result": "期望从这一步得到什么信息（用于后续判断这一步是否成功）"
    }}
  ]
}}

规划原则：
1. 步骤数控制在 2-5 步，不要规划过多步骤
2. 每个步骤只能使用一个工具（Plan-Execute 不像 ReAct 那样灵活）
3. 步骤之间应该有逻辑关系：前一步的输出是后一步的输入依据
4. 最后一步通常不需要工具 —— 如果前面的信息已经足够，最后一步可以是 "final_summary"
5. tool_params 中的参数值要具体（不要写"待定"或"根据上一步结果"）
6. 只输出 JSON，不要输出任何其他文字（不要解释、不要 markdown 代码块标记）
"""

# 用于 replan 的 prompt —— 和初始规划不同，它看到了已执行的步骤和失败信息
REPLAN_PROMPT = """你是一个任务规划专家。以下任务在执行过程中遇到了问题，需要你重新规划剩余步骤。

原始目标：{goal}

已完成的步骤及其结果：
{completed_steps}

失败的步骤：
{failed_step}

失败原因：{failure_reason}

请为剩余的工作制定新的执行计划。输出 JSON 格式：

{{
  "goal": "调整后的目标",
  "steps": [
    {{
      "id": {next_id},
      "description": "...",
      "tool": "工具名",
      "tool_params": {{"参数名": "参数值"}},
      "expected_result": "..."
    }}
  ]
}}

注意：
1. 不要重复已完成且成功的步骤
2. 分析失败原因，调整策略（如换工具、换参数、换搜索关键词）
3. 只输出 JSON，不要输出任何其他文字
"""

# 最终汇总 prompt
FINAL_SUMMARY_PROMPT = """你是一个信息汇总专家。请根据以下执行结果，回答用户的问题。

用户问题：{question}

执行过程和结果：
{results_summary}

请用中文给出完整、有条理的最终答案。如果某些步骤失败了，诚实说明哪些信息无法获取。
"""


# ========================================================================
# Plan 解析器 —— 从 LLM 文本中提取 JSON
# ========================================================================


def extract_json_from_text(text: str) -> dict | None:
    """从 LLM 的文本输出中提取 JSON。

    LLM 输出 JSON 时常见的问题：
      1. 被 ```json ... ``` 包裹（markdown 代码块习惯）
      2. 被 ``` ... ``` 包裹（不带语言标记）
      3. JSON 前后有其他文字（如"好的，这是计划：{...}"）
      4. JSON 中包含单引号而非双引号（Python 习惯）

    这个函数逐一尝试处理这些情况。
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 策略 1: 提取 ```json ... ``` 代码块
    json_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 2: 提取 ``` ... ``` 代码块（不带语言标记）
    code_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 3: 查找第一个 { 和最后一个 }，提取中间内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 策略 4: 尝试替换单引号为双引号（有些模型会输出 Python dict 格式）
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        try:
            # ast.literal_eval 可以解析 Python dict
            import ast

            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

    return None


# ========================================================================
# 步骤校验器 —— 判断一个步骤的执行结果是否"成功"
# ========================================================================


def validate_step_result(step: PlanStep, output: str) -> tuple[bool, str]:
    """校验步骤执行结果。

    返回值：(是否成功, 校验说明)

    校验规则（从简单到严格）：
      1. 输出不能为空
      2. 输出不能包含错误标记（"错误"、"Error"等）
      3. 如果步骤有 expected_result，输出长度至少要有 10 个字符
         （一个非常宽松的"内容有效性"检查）
    """
    if not output or not output.strip():
        return False, "工具返回了空结果"

    output_stripped = output.strip()

    # 检查是否包含错误标记
    error_keywords = ["错误：", "Error:", "执行错误", "未知工具", "参数错误"]
    for keyword in error_keywords:
        if keyword in output_stripped:
            return False, f"工具返回了错误: {output_stripped[:100]}"

    # 如果指定了期望结果，做一个非常简单的检查：输出不能只有 1-2 个字符
    # 注意：计算器结果如 "320.0" 只有 5 个字符，完全正常
    # 只有小于 3 个字符才可疑（如 "."、"-"、"无"等）
    if step.expected_result and len(output_stripped) < 3:
        return False, f"结果太短（{len(output_stripped)}字符），可能不完整"

    note = "[OK]"
    if step.expected_result:
        note += f" 期望: {step.expected_result[:50]}..."
    return True, note


# ========================================================================
# 工具执行器 —— 和 ReAct Agent 公用同一套 TOOL_REGISTRY
# ========================================================================


def execute_tool(tool_name: str, params: dict[str, str]) -> str:
    """执行工具，返回结果字符串。"""
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        return f"错误：未知工具 '{tool_name}'。可用工具: {available}"

    tool_info = TOOL_REGISTRY[tool_name]
    func = tool_info["function"]

    try:
        if not params:
            result = func()
        else:
            result = func(**params)
        return str(result)
    except TypeError as e:
        return f"参数错误：{e}。正确用法：{tool_info['param_description']}"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"


def build_tool_descriptions() -> str:
    """构建工具描述文本（和 ReAct 用同一套工具）。"""
    lines: list[str] = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  用法: {info['param_description']}")
    return "\n".join(lines)


# ========================================================================
# PlanExecuteAgent —— 核心类
# ========================================================================


class PlanExecuteAgent:
    """Plan-Execute 范式的 Agent。

    和 ReactAgent (Day 2) 的对比：
      ReactAgent:      每轮需要一次 LLM 调用 (Thought + Action)
      PlanExecuteAgent: 规划 1 次 LLM 调用 + N 次纯代码执行 + 可选 replan

    这意味着：如果计划有 4 步，且全部成功，
    PlanExecuteAgent 只需要 3 次 LLM 调用（规划 1 + 汇总 1 + 可能有 1 次失败的 replan），
    而 ReAct 至少需要 4 次（每步一次 Thought）。

    构造参数：
      client       -> LLMClient 实例
      max_replans  -> 最多允许几次 replan（默认 2，防止无限循环）
      verbose      -> 是否打印详细日志到控制台（学习用）
    """

    def __init__(self, client: Any, max_replans: int = 2, verbose: bool = True):
        self.client = client
        self.max_replans = max_replans
        self.verbose = verbose

        # 把工具描述缓存起来（初始化时构建一次）
        self.tool_descriptions = build_tool_descriptions()
        self.system_prompt = PLAN_SYSTEM_PROMPT.format(tool_descriptions=self.tool_descriptions)

        # 记录最后一次运行的调试信息
        self.last_trace: dict[str, Any] = {}

        logger.info(
            "PlanExecuteAgent 初始化: provider=%s, model=%s, max_replans=%d",
            client.provider,
            client.model,
            max_replans,
        )

    def run(self, question: str) -> str:
        """执行 Plan-Execute 完整流程，返回最终答案。

        流程：
          1. Planning:    LLM 生成 JSON 计划
          2. Execution:   逐步执行 + 校验
          3. Replanning:  失败时调整剩余计划（最多 max_replans 次）
          4. Finalization: LLM 汇总所有结果 -> 最终答案
        """
        self.last_trace = {"question": question, "phases": []}

        # ---- Phase 1: Planning ----
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 1: Planning -- LLM 生成执行计划")
            print("=" * 60)

        plan = self._generate_plan(question)
        if plan is None or not plan.steps:
            return "[ERR] 无法生成有效的执行计划。请尝试简化问题。"

        if self.verbose:
            print(f"\n目标: {plan.goal}")
            print(f"计划步骤数: {len(plan.steps)}")
            for s in plan.steps:
                print(f"  Step {s.id}: [{s.tool}] {s.description}")

        self.last_trace["phases"].append({"phase": "planning", "plan": plan})

        # ---- Phase 2: Execute + Validate + Replan ----
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Execution -- 逐步执行 + 校验")
            print("=" * 60)

        all_results: list[StepResult] = []
        current_plan = plan
        replan_count = 0

        while current_plan.steps:
            step = current_plan.steps[0]  # 取第一个待执行步骤

            # 执行步骤
            result = self._execute_step(step)
            all_results.append(result)

            # 从计划中移除已执行的步骤
            current_plan.steps = current_plan.steps[1:]

            if result.success:
                # 步骤成功 -> 继续下一步
                continue

            # 步骤失败 -> 检查是否还有剩余步骤需要调整
            if not current_plan.steps:
                # 没有剩余步骤了，不需要 replan
                if self.verbose:
                    print("  [注意] 最后一步失败，但没有剩余步骤需要调整")
                break

            # 需要 replan
            if replan_count >= self.max_replans:
                if self.verbose:
                    print(f"  [注意] 已达到最大 replan 次数 ({self.max_replans})，" "跳过剩余步骤")
                break

            if self.verbose:
                print(f"\n  -> 触发 Replan ({replan_count + 1}/{self.max_replans})")

            new_plan = self._replan(
                question=question,
                original_plan=plan,
                completed_results=all_results,
                failed_step=step,
                failure_reason=result.validation_note,
                next_id=step.id + 1,
            )

            replan_count += 1

            if new_plan and new_plan.steps:
                current_plan = new_plan
                if self.verbose:
                    print(f"  新计划: {len(new_plan.steps)} 个步骤")
                    for s in new_plan.steps:
                        print(f"    Step {s.id}: [{s.tool}] {s.description}")
            else:
                if self.verbose:
                    print("  [注意] Replan 失败，使用原计划剩余步骤")
                # 继续用原计划的剩余步骤

        self.last_trace["phases"].append(
            {
                "phase": "execution",
                "results": all_results,
                "replan_count": replan_count,
            }
        )

        # ---- Phase 3: Finalization ----
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 3: Finalization -- LLM 汇总结果")
            print("=" * 60)

        final_answer = self._generate_final_answer(question, all_results)
        self.last_trace["phases"].append({"phase": "finalization", "answer": final_answer})

        return final_answer

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_plan(self, question: str) -> Plan | None:
        """调用 LLM 生成执行计划。"""
        user_message = f"请为以下问题制定执行计划：\n\n{question}"

        raw_output = self._call_llm(self.system_prompt, user_message)

        if self.verbose:
            print(f"\nLLM 原始输出:\n{raw_output[:500]}")

        plan_dict = extract_json_from_text(raw_output)
        if plan_dict is None:
            logger.warning("无法从 LLM 输出中提取 JSON 计划")
            # 降级策略：创建一个单步骤计划，让 LLM 直接回答
            return Plan(
                goal=question,
                steps=[
                    PlanStep(
                        id=1,
                        description="搜索相关信息",
                        tool="search",
                        tool_params={"query": question},
                        expected_result="获取与问题相关的信息",
                    )
                ],
            )

        try:
            plan = Plan.from_dict(plan_dict)
            return plan
        except Exception as e:
            logger.warning("Plan 解析失败: %s", e)
            return None

    def _execute_step(self, step: PlanStep) -> StepResult:
        """执行单个步骤 + 校验。"""
        if self.verbose:
            print(f"\n  Step {step.id}: [{step.tool}] {step.description}")
            if step.tool_params:
                print(f"    参数: {step.tool_params}")

        # "final_summary" 是一个特殊的伪工具 —— 它不需要执行
        # 表示 LLM 认为信息已经足够，可以进入汇总阶段
        if step.tool in ("final_summary", "none", ""):
            return StepResult(
                step=step,
                success=True,
                output="(跳过，无需工具)",
                validation_note="[OK] 汇总步骤，无需执行工具",
            )

        output = execute_tool(step.tool, step.tool_params)

        if self.verbose:
            print(f"    输出: {output[:150]}...")

        success, note = validate_step_result(step, output)

        if self.verbose:
            status = "[OK]" if success else "[FAIL]"
            print(f"    校验: {status} {note}")

        return StepResult(step=step, success=success, output=output, validation_note=note)

    def _replan(
        self,
        question: str,
        original_plan: Plan,
        completed_results: list[StepResult],
        failed_step: PlanStep,
        failure_reason: str,
        next_id: int,
    ) -> Plan | None:
        """步骤失败时，调用 LLM 重新规划剩余步骤。

        和初始规划的不同：
          - 初始规划：LLM 只知道问题，不知道任何执行结果
          - Replan：LLM 看到了已成功的步骤 + 失败的步骤 + 失败原因
                    它可以调整策略（换个工具、换个搜索关键词、换个角度）
        """
        # 构建已完成步骤的摘要
        completed_lines: list[str] = []
        for r in completed_results:
            if r.success:
                completed_lines.append(
                    f"  Step {r.step.id} [{r.step.tool}] {r.step.description}\n"
                    f"    结果: {r.output[:150]}..."
                )
        completed_text = "\n".join(completed_lines) if completed_lines else "(无)"

        # 从已完成结果中找到失败步骤的输出
        failed_output = "N/A"
        for r in completed_results:
            if r.step.id == failed_step.id:
                failed_output = r.output[:200]
                break

        failed_text = (
            f"  Step {failed_step.id} [{failed_step.tool}] {failed_step.description}\n"
            f"  参数: {failed_step.tool_params}\n"
            f"  输出: {failed_output}"
        )

        prompt = REPLAN_PROMPT.format(
            goal=original_plan.goal,
            completed_steps=completed_text,
            failed_step=failed_text,
            failure_reason=failure_reason,
            next_id=next_id,
        )

        raw_output = self._call_llm(
            PLAN_SYSTEM_PROMPT.format(tool_descriptions=self.tool_descriptions),
            prompt,
        )

        if self.verbose:
            print(f"\n  Replan LLM 输出:\n{raw_output[:300]}")

        plan_dict = extract_json_from_text(raw_output)
        if plan_dict is None:
            logger.warning("Replan: 无法从 LLM 输出中提取 JSON")
            return None

        try:
            return Plan.from_dict(plan_dict)
        except Exception as e:
            logger.warning("Replan 解析失败: %s", e)
            return None

    def _generate_final_answer(self, question: str, results: list[StepResult]) -> str:
        """汇总所有步骤的结果，生成最终答案。"""
        # 构建结果摘要
        lines: list[str] = []
        for r in results:
            status = "[OK]" if r.success else "[FAIL]"
            lines.append(
                f"Step {r.step.id} ({status}): {r.step.description}\n"
                f"  工具: {r.step.tool}\n"
                f"  输出: {r.output[:300]}"
            )
        results_text = "\n\n".join(lines)

        prompt = FINAL_SUMMARY_PROMPT.format(question=question, results_summary=results_text)

        answer = self._call_llm("你是一个信息汇总专家，用中文回答。", prompt)
        return answer.strip()

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """调用 LLM，返回文本回复。

        和 ReactAgent._call_llm 不同：
          - React 需要 stop 序列（防止 LLM 编造 Observation）
          - Plan-Execute 不需要 stop 序列（我们期望 LLM 输出完整的 JSON）
          - Plan-Execute 使用 system + user 消息结构（不是单个 prompt）
        """
        if self.client.sdk_type == "anthropic":
            response = self.client._client.messages.create(
                model=self.client.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return "".join(b.text for b in response.content if hasattr(b, "text") and b.text)
        else:
            response = self.client._client.chat.completions.create(
                model=self.client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content or ""


# ========================================================================
# 对比辅助：用同一个问题跑 ReAct 和 Plan-Execute，展示差异
# ========================================================================


def build_comparison_table(react_trace: dict, plan_execute_trace: dict) -> str:
    """构建 ReAct vs Plan-Execute 对比表。

    这是一个辅助函数，用于在学习时直观比较两种范式。
    参数是两个 Agent 的 trace 信息。
    """
    lines = [
        "ReAct vs Plan-Execute 对比",
        "=" * 50,
        "",
        f"问题: {react_trace.get('question', 'N/A')}",
        "",
        "维度           | ReAct              | Plan-Execute",
        "---------------+--------------------+--------------------",
    ]

    # LLM 调用次数（估算）
    react_calls = react_trace.get("rounds", "?")
    pe_trace_phases = plan_execute_trace.get("phases", [])
    pe_calls = 2  # 至少 2 次（规划 + 汇总）
    for phase in pe_trace_phases:
        if phase.get("phase") == "execution":
            pe_calls += phase.get("replan_count", 0)

    lines.append(f"LLM 调用次数    | ~{react_calls} 次              | {pe_calls} 次")
    lines.append("推理可见性      | 完全可见 (Thought)   | 部分可见 (Plan JSON)")
    lines.append("格式依赖        | 文本解析 (易出错)    | JSON 解析 (更可靠)")
    lines.append("灵活性          | 高 (每步动态决策)    | 低 (预设步骤)")
    lines.append("成本            | 较高 (多次 LLM 调用) | 较低 (少量 LLM 调用)")

    return "\n".join(lines)
