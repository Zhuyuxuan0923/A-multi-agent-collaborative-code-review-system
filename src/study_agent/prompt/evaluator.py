"""Prompt 评测框架 —— 用同一批测试用例对比不同 prompt 写法的效果。

这个模块解决什么问题？
  写 prompt 像"调音"——你凭感觉改了某个词，但不知道是真的更好了还是心理作用。
  评测框架把"感觉"变成"数字"：同一批题，换不同的 prompt 去跑，看准确率。

核心工作流：

  50 条测试用例（每一条都有"标准答案"）
       │
       ├── 用 Prompt 风格 A（最简）跑 50 次 → 算准确率
       ├── 用 Prompt 风格 B（结构化）跑 50 次 → 算准确率
       └── 用 Prompt 风格 C（Few-Shot）跑 50 次 → 算准确率
       │
       ▼
  对比报告：哪个风格最好？好在哪？哪些题型拉开差距？

评测指标（四个维度）：

  ① JSON 合法率     —— 输出是不是合法的 JSON？
  ② 字段完整率      —— 必填字段有没有都出现？
  ③ 逐字段准确率    —— category 对了几个？priority 对了几个？
  ④ 综合得分        —— 以上三项的加权平均

使用方法：
  from study_agent.llm import LLMClient, StructuredExtractor, ExtractionSchema
  from study_agent.prompt.evaluator import PromptEvaluator, EvalCase
  from study_agent.prompt.test_cases import CLASSIFICATION_CASES

  client = LLMClient.from_env()
  extractor = StructuredExtractor(client)
  evaluator = PromptEvaluator(client, extractor, CLASSIFY_SCHEMA)

  results = evaluator.run_batch(CLASSIFICATION_CASES, your_prompt_styles)
  print(evaluator.report(results))
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ① 数据类型定义
# ═══════════════════════════════════════════════════════════════


@dataclass
class EvalCase:
    """一条测试用例：输入文本 + 期望的标准答案。

    字段说明：
      id       → 用例编号，如 "TC001"（TC = Test Case）
      text     → 原始输入文本（模拟用户发来的消息）
      expected → 期望 LLM 输出的结果，key 是字段名，value 是期望值
    """

    id: str
    text: str
    expected: dict[str, str]


@dataclass
class EvalResult:
    """一次评测的结果——某一个 test case × 某一种 prompt 风格的运行记录。

    字段说明：
      case_id         → 哪个测试用例
      style           → 用了哪种 prompt 风格（如 "minimal" / "structured" / "fewshot"）
      raw_response    → LLM 返回的原始文本（调试时很重要！）
      parsed          → 解析后的 dict，解析失败则为 None
      json_valid      → JSON 是否合法
      fields_complete → 所有 required 字段是否都出现了
      field_match     → 每个字段是否和期望值一致（field_name → True/False）
      errors          → 出错信息列表
      elapsed_ms      → 这次调用花了多少毫秒
    """

    case_id: str
    style: str
    raw_response: str = ""
    parsed: dict[str, Any] | None = None
    json_valid: bool = False
    fields_complete: bool = False
    field_match: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def overall_pass(self) -> bool:
        """所有检查都通过才算过。"""
        return (
            self.json_valid
            and self.fields_complete
            and (len(self.field_match) > 0 and all(self.field_match.values()))
        )


# ═══════════════════════════════════════════════════════════════
# ② 默认分类 Schema —— 客服消息三分类
# ═══════════════════════════════════════════════════════════════

CLASSIFY_SCHEMA_DEF = {
    "name": "classify_inquiry",
    "description": "分类一条客服咨询消息——判断类别、优先级和情感倾向",
    "properties": {
        "category": {
            "type": "string",
            "description": (
                "消息类别，必须是以下之一："
                "billing（账单/付款）、technical（技术问题）、"
                "account（账户管理）、product（产品咨询）、complaint（投诉）"
            ),
        },
        "priority": {
            "type": "string",
            "description": "处理优先级：high（需立即处理）、medium（正常处理）、low（不紧急）",
        },
        "sentiment": {
            "type": "string",
            "description": (
                "用户情感：negative（负面/生气/沮丧）、"
                "neutral（中性/客观）、positive（正面/满意/开心）"
            ),
        },
    },
    "required": ["category", "priority", "sentiment"],
}


# ═══════════════════════════════════════════════════════════════
# ③ PromptEvaluator —— 评测引擎
# ═══════════════════════════════════════════════════════════════


class PromptEvaluator:
    """Prompt 评测引擎——用同一批测试用例对比多种 prompt 写法的效果。

    用法：
      evaluator = PromptEvaluator(client, extractor, CLASSIFY_SCHEMA_DEF)
      results = evaluator.run_batch(cases, {
          "minimal": build_minimal_prompt,
          "structured": build_structured_prompt,
          "fewshot": build_fewshot_prompt,
      })
      print(evaluator.report(results))
    """

    def __init__(self, client: Any, extractor: Any, schema_def: dict[str, Any]):
        """创建评测器。

        参数：
          client     → LLMClient 实例
          extractor  → StructuredExtractor 实例（确保输出合法 JSON）
          schema_def → 提取 Schema 的字典定义（有 name/description/properties/required 四个 key）
        """
        self.client = client
        self.extractor = extractor
        self.schema_def = schema_def
        self.required_fields: list[str] = schema_def.get("required", [])

        # 导入 ExtractionSchema 来构建标准 Schema 对象
        from study_agent.llm.structured import ExtractionSchema

        self.schema = ExtractionSchema(
            name=schema_def["name"],
            description=schema_def["description"],
            properties=schema_def["properties"],
            required=schema_def["required"],
        )

    # ── 单次评测 ──────────────────────────────────────────

    def evaluate_one(
        self,
        case: EvalCase,
        style_name: str,
        build_prompt: Callable[[str], tuple[str | None, str]],
        method: str = "function_call",
    ) -> EvalResult:
        """对一条测试用例、用一种 prompt 风格，执行一次评测。

        参数：
          case         → 测试用例
          style_name   → prompt 风格名（用于报告中标记）
          build_prompt → 一个函数，接受文本，返回 (system_prompt, user_prompt)
          method       → 提取方法（默认 function_call，最可靠）

        返回：
          EvalResult 包含所有检查结果

        build_prompt 签名：
          def my_style(text: str) -> tuple[str | None, str]:
              system = "你是客服分类助手..."
              user = f"请分类: {text}"
              return system, user
        """
        result = EvalResult(case_id=case.id, style=style_name)

        # 1. 构建 prompt
        system, user = build_prompt(case.text)

        # 2. 调用 LLM（用 StructuredExtractor 保证 JSON）
        t0 = time.perf_counter()
        try:
            # 为了把 system + user 一起传给 extractor，
            # 我们直接把 user message 作为 text，把 system 拼进去
            # 这里稍微 hack 一下：用 client.chat() 手工传 system
            raw = self.client.chat(user, system=system)
            result.raw_response = raw

            # 从原始回复中抢救 JSON
            from study_agent.llm.structured import _parse_json_response

            parsed = _parse_json_response(raw)
            result.parsed = parsed
        except Exception as e:
            result.errors.append(f"API 调用失败: {type(e).__name__}: {e}")
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result

        result.elapsed_ms = (time.perf_counter() - t0) * 1000

        # 3. 检查 JSON 合法性
        if result.parsed is None:
            result.errors.append("JSON 解析失败——LLM 输出无法解析为 JSON")
            return result
        result.json_valid = True

        # 4. 检查字段完整性（所有 required 字段都在吗？）
        missing = [f for f in self.required_fields if f not in result.parsed]
        if missing:
            result.fields_complete = False
            result.errors.append(f"缺少必填字段: {missing}")
        else:
            result.fields_complete = True

        # 5. 逐字段检查准确性（值对不对？）
        for field_name in self.required_fields:
            expected_val = case.expected.get(field_name, "").strip().lower()
            actual_val = str(result.parsed.get(field_name, "")).strip().lower()
            result.field_match[field_name] = actual_val == expected_val

        return result

    # ── 批量评测 ──────────────────────────────────────────

    def run_batch(
        self,
        cases: list[EvalCase],
        styles: dict[str, Callable[[str], tuple[str | None, str]]],
        method: str = "function_call",
    ) -> list[EvalResult]:
        """对所有测试用例 × 所有 prompt 风格运行评测。

        参数：
          cases  → 测试用例列表
          styles → {"风格名": build_prompt函数, ...}
          method → 提取方法

        返回：
          所有 EvalResult 的平铺列表（len = len(cases) × len(styles)）

        耗时说明：
          如果有 50 个用例、3 种风格，就是 150 次 LLM 调用。
          每次调用约 0.5-2 秒，总耗时约 2-5 分钟（取决于 API 速率）。
        """
        all_results: list[EvalResult] = []
        total = len(cases) * len(styles)
        done = 0

        logger.info("开始批量评测: %d 用例 × %d 风格 = %d 次调用", len(cases), len(styles), total)

        for case in cases:
            for style_name, build_fn in styles.items():
                result = self.evaluate_one(case, style_name, build_fn, method)
                all_results.append(result)
                done += 1
                if done % 15 == 0:
                    logger.info("  进度: %d/%d", done, total)

        logger.info("评测完成: %d 次调用", done)
        return all_results

    # ── 汇总统计 ──────────────────────────────────────────

    def summarize(self, results: list[EvalResult]) -> dict[str, dict[str, Any]]:
        """按 prompt 风格分组统计。

        返回格式：
          {
            "minimal": {
                "total": 50, "json_valid": 42, "fields_complete": 38,
                "overall_pass": 30, "overall_rate": 0.60,
                "field_accuracy": {"category": 0.80, "priority": 0.72, "sentiment": 0.76},
                "avg_time_ms": 850.0
            },
            ...
          }
        """
        # 按风格分组
        groups: dict[str, list[EvalResult]] = {}
        for r in results:
            groups.setdefault(r.style, []).append(r)

        summary: dict[str, dict[str, Any]] = {}
        for style, group in groups.items():
            total = len(group)
            json_ok = sum(1 for r in group if r.json_valid)
            fields_ok = sum(1 for r in group if r.fields_complete)
            passed = sum(1 for r in group if r.overall_pass)
            avg_time = sum(r.elapsed_ms for r in group) / total if total > 0 else 0

            # 逐字段准确率
            field_acc: dict[str, float] = {}
            for field_name in self.required_fields:
                # 只统计有结果的（parsed 不为 None）
                valid_results = [r for r in group if r.parsed is not None]
                if valid_results:
                    correct = sum(1 for r in valid_results if r.field_match.get(field_name, False))
                    field_acc[field_name] = correct / len(valid_results)
                else:
                    field_acc[field_name] = 0.0

            summary[style] = {
                "total": total,
                "json_valid": json_ok,
                "json_valid_rate": json_ok / total,
                "fields_complete": fields_ok,
                "fields_complete_rate": fields_ok / total,
                "overall_pass": passed,
                "overall_rate": passed / total,
                "field_accuracy": field_acc,
                "avg_time_ms": avg_time,
            }

        return summary

    # ── 生成 Markdown 报告 ────────────────────────────────

    def report(self, results: list[EvalResult]) -> str:
        """根据评测结果生成 Markdown 格式的对比报告。

        报告包含：
          1. 总览表——各风格的四个核心指标
          2. 逐字段准确率——哪个风格在哪个字段上表现最好？
          3. 结论——哪个风格赢了？赢了几个百分点？
        """
        summary = self.summarize(results)

        lines: list[str] = []
        lines.append("# Prompt 评测报告")
        lines.append("")
        lines.append(f"**评测时间**：{time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**模型**：{self.client.provider} / {self.client.model}")
        lines.append(f"**测试用例数**：{summary[next(iter(summary))]['total']}")
        lines.append(f"**Prompt 风格数**：{len(summary)}")
        lines.append("")

        # ── 总览表 ──
        lines.append("## 一、总览对比")
        lines.append("")
        headers = ["Prompt 风格", "JSON 合法率", "字段完整率", "综合通过率", "平均耗时"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["------"] * len(headers)) + "|")

        for style, stats in summary.items():
            row = [
                f"**{style}**",
                f"{stats['json_valid_rate']:.0%}",
                f"{stats['fields_complete_rate']:.0%}",
                f"{stats['overall_rate']:.0%}",
                f"{stats['avg_time_ms']:.0f}ms",
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # ── 逐字段准确率 ──
        lines.append("## 二、逐字段准确率")
        lines.append("")
        # 表头：Prompt 风格 | field1 | field2 | ...
        f_headers = ["Prompt 风格"] + self.required_fields
        lines.append("| " + " | ".join(f_headers) + " |")
        lines.append("|" + "|".join(["------"] * len(f_headers)) + "|")

        for style, stats in summary.items():
            cells = [f"**{style}**"]
            for field_name in self.required_fields:
                acc = stats["field_accuracy"].get(field_name, 0.0)
                cells.append(f"{acc:.0%}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # ── 结论 ──
        lines.append("## 三、结论")
        lines.append("")

        # 找出最佳风格
        if len(summary) >= 2:
            styles_sorted = sorted(
                summary.items(), key=lambda x: x[1]["overall_rate"], reverse=True
            )
            best_name, best_stats = styles_sorted[0]
            worst_name, worst_stats = styles_sorted[-1]
            diff = best_stats["overall_rate"] - worst_stats["overall_rate"]

            lines.append(f"- **最佳风格**：{best_name}（通过率 {best_stats['overall_rate']:.0%}）")
            lines.append(
                f"- **最差风格**：{worst_name}（通过率 {worst_stats['overall_rate']:.0%}）"
            )
            lines.append(f"- **差距**：{diff:.0%}（{diff * 100:.0f} 个百分点）")
            lines.append("")

            # 分析最大差异的字段
            max_field_diff = 0.0
            max_field_name = ""
            for field_name in self.required_fields:
                fd = best_stats["field_accuracy"].get(field_name, 0) - worst_stats[
                    "field_accuracy"
                ].get(field_name, 0)
                if fd > max_field_diff:
                    max_field_diff = fd
                    max_field_name = field_name
            if max_field_name:
                best_acc = best_stats["field_accuracy"][max_field_name]
                worst_acc = worst_stats["field_accuracy"][max_field_name]
                lines.append(
                    f"  **{max_field_name}** 字段的差距最大"
                    f"（{best_acc:.0%} vs {worst_acc:.0%}），"
                    f"说明 prompt 质量的差异在这个维度上影响最明显。"
                )
                lines.append("")

        # 排查失败的 case
        lines.append("## 四、失败最多的测试用例")
        lines.append("")
        # 按 case_id 统计失败次数
        case_fail_count: dict[str, int] = {}
        for r in results:
            if not r.overall_pass:
                case_fail_count[r.case_id] = case_fail_count.get(r.case_id, 0) + 1

        # 列出失败 >= 2 种风格的 case（即"所有风格都搞不定"的难题）
        hard_cases = [
            (cid, cnt) for cid, cnt in case_fail_count.items() if cnt >= len(summary) * 0.5
        ]
        hard_cases.sort(key=lambda x: x[1], reverse=True)

        if hard_cases:
            lines.append(
                "这些用例在多种 prompt 风格下都失败了——可能是题目本身有歧义，或是 LLM 的认知盲区："
            )
            lines.append("")
            for cid, cnt in hard_cases[:5]:  # 最多列5个
                lines.append(f"- **{cid}**（{cnt}/{len(summary)} 种风格失败）")
            lines.append("")
        else:
            lines.append("没有发现所有风格都搞不定的题目。")
            lines.append("")

        return "\n".join(lines)

    # ── 导出 CSV 格式的详细结果 ────────────────────────────

    def export_csv(self, results: list[EvalResult]) -> str:
        """把评测结果导出为 CSV 格式（可用 Excel / Google Sheets 打开）。

        CSV = Comma-Separated Values，纯文本表格格式。
        """
        lines = ["case_id,style,json_valid,fields_complete,overall_pass,errors"]
        for r in results:
            errors_escaped = "; ".join(r.errors).replace('"', '""')
            lines.append(
                f'{r.case_id},{r.style},{r.json_valid},{r.fields_complete},{r.overall_pass},"{errors_escaped}"'
            )
        return "\n".join(lines)
