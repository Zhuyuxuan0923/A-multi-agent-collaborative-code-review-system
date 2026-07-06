"""Trace 导出器 -- 将 AgentTrace 导出为多种格式。

三种导出格式的适用场景：
  JSON     -> 程序间传递、存入数据库、后续分析
  Markdown -> 人类阅读、放入文档、发给同事审查
  Console  -> 实时调试、开发时快速查看
"""

from __future__ import annotations

from datetime import UTC, datetime

from study_agent.agent.trace import AgentTrace, TraceSpan

# ═══════════════════════════════════════════════════════════
# JSON 导出
# ═══════════════════════════════════════════════════════════


def export_json(trace: AgentTrace, indent: int = 2) -> str:
    """导出为 JSON 字符串。

    JSON 格式保留了完整的 Trace 数据，适合：
      - 存储到文件/数据库
      - 程序间传递
      - 后续用脚本分析（比如统计一周内工具调用失败率）
    """
    return trace.model_dump_json(indent=indent)


def export_json_file(trace: AgentTrace, filepath: str) -> None:
    """导出为 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(export_json(trace))


# ═══════════════════════════════════════════════════════════
# Markdown 导出
# ═══════════════════════════════════════════════════════════


def export_markdown(trace: AgentTrace) -> str:
    """导出为 Markdown 格式的人类可读报告。

    报告结构：
      1. 执行概览 -- 摘要信息
      2. 时间线 -- 关键事件按时间排列
      3. 逐轮详情 -- 每轮 LLM 调用 + 工具调用详情
      4. 性能分析 -- 耗时分布
      5. 错误清单 -- 如有错误单独列出
    """
    lines: list[str] = []
    _append_header(lines, trace)
    _append_overview(lines, trace)
    _append_timeline(lines, trace)
    _append_round_details(lines, trace)
    _append_performance(lines, trace)
    if trace.error_count > 0:
        _append_errors(lines, trace)
    _append_footer(lines, trace)
    return "\n".join(lines)


def export_markdown_file(trace: AgentTrace, filepath: str) -> None:
    """导出为 Markdown 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(export_markdown(trace))


def _append_header(lines: list[str], trace: AgentTrace) -> None:
    lines.append("# Agent Trace 报告")
    lines.append("")
    lines.append(f"**Trace ID**: `{trace.trace_id}`")
    lines.append(f"**Agent 类型**: {trace.agent_type}")
    lines.append(f"**生成时间**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")


def _append_overview(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("## 1. 执行概览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 用户问题 | {trace.question} |")
    lines.append(f"| 总耗时 | {trace.total_duration_ms:.0f} ms |")
    lines.append(f"| ReAct 轮次 | {trace.total_rounds} |")
    lines.append(f"| LLM 调用次数 | {trace.total_llm_calls} |")
    lines.append(f"| 工具调用次数 | {trace.total_tool_calls} |")
    lines.append(f"| 事件总数 | {trace.total_events} |")
    lines.append(f"| 错误数 | {trace.error_count} |")
    lines.append(f"| 平均每轮耗时 | {trace.avg_round_duration_ms:.0f} ms |")
    lines.append(f"| 最终答案长度 | {len(trace.final_answer)} 字符 |")
    lines.append("")
    lines.append("**最终答案:**")
    lines.append("")
    lines.append(f"> {trace.final_answer[:300]}")
    lines.append("")


def _append_timeline(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("## 2. 事件时间线")
    lines.append("")
    lines.append("| 时间偏移 | Span 类型 | 事件类型 | 描述 |")
    lines.append("|----------|-----------|----------|------|")

    timeline = trace.timeline()
    for entry in timeline[:50]:  # 最多显示 50 条
        span_type = entry["span_type"]
        event_type = entry["event_type"]
        message = entry["message"][:60]
        lines.append(
            f"| {entry['time_offset_ms']:.0f}ms | {span_type} | {event_type} | {message} |"
        )

    if len(timeline) > 50:
        lines.append(f"| ... | ... | ... | (还有 {len(timeline) - 50} 条事件) |")
    lines.append("")


def _append_round_details(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("## 3. 逐轮详情")
    lines.append("")

    round_spans = [s for s in trace.spans if s.span_type == "react_round"]
    for span in round_spans:
        round_num = span.metadata.get("round_num", "?")
        lines.append(f"### Round {round_num} ({span.duration_ms:.0f}ms)")
        lines.append("")

        # 找这一轮的 LLM call 和 tool execution
        llm_spans = [
            s for s in trace.spans if s.span_type == "llm_call" and s.parent_span_id == span.span_id
        ]
        tool_spans = [
            s
            for s in trace.spans
            if s.span_type == "tool_execution" and s.parent_span_id == span.span_id
        ]

        for llm_span in llm_spans:
            lines.append(f"- **LLM 调用** ({llm_span.duration_ms:.0f}ms):")
            for event in llm_span.events:
                if event.event_type == "llm_response":
                    lines.append(f"  - 响应长度: {event.data.get('response_length', '?')} 字符")
                    lines.append(f"  - Token: {event.data.get('token_count', '?')}")
                if event.event_type == "llm_request":
                    lines.append(f"  - Prompt 长度: {event.data.get('prompt_length', '?')} 字符")

        for tool_span in tool_spans:
            tool_name = tool_span.metadata.get("tool_name", "?")
            success = tool_span.metadata.get("success", True)
            status = "[OK]" if success else "[ERR]"
            lines.append(f"- **工具调用** `{tool_name}` {status} ({tool_span.duration_ms:.0f}ms):")
            for event in tool_span.events:
                if event.event_type == "tool_call":
                    params = event.data.get("params", {})
                    lines.append(f"  - 参数: {params}")
                if event.event_type == "tool_result":
                    lines.append(f"  - 结果: {event.data.get('result_preview', '')[:100]}")

        # 检查是否有解析错误
        has_parse_error = span.metadata.get("parse_error", False)
        if has_parse_error:
            lines.append("- **[WARN] 本轮解析失败**")
        if span.metadata.get("final"):
            lines.append("- **[OK] 本轮找到最终答案**")

        lines.append("")


def _append_performance(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("## 4. 性能分析")
    lines.append("")

    # 耗时分布
    llm_spans = [s for s in trace.spans if s.span_type == "llm_call"]
    tool_spans = [s for s in trace.spans if s.span_type == "tool_execution"]

    total_llm_ms = sum(s.duration_ms for s in llm_spans)
    total_tool_ms = sum(s.duration_ms for s in tool_spans)
    total_ms = trace.total_duration_ms

    lines.append("### 耗时分布")
    lines.append("")
    lines.append("| 阶段 | 耗时 | 占比 |")
    lines.append("|------|------|------|")
    if total_ms > 0:
        lines.append(f"| LLM 调用 | {total_llm_ms:.0f}ms | {total_llm_ms/total_ms*100:.1f}% |")
        lines.append(f"| 工具执行 | {total_tool_ms:.0f}ms | {total_tool_ms/total_ms*100:.1f}% |")
        other_ms = total_ms - total_llm_ms - total_tool_ms
        lines.append(f"| 其他 | {other_ms:.0f}ms | {other_ms/total_ms*100:.1f}% |")
    lines.append("")

    # 每轮耗时
    lines.append("### 每轮耗时")
    lines.append("")
    lines.append("| 轮次 | 耗时 | LLM | 工具 |")
    lines.append("|------|------|-----|------|")
    round_spans = [s for s in trace.spans if s.span_type == "react_round"]
    for span in round_spans:
        rn = span.metadata.get("round_num", "?")
        r_llm = sum(s.duration_ms for s in llm_spans if s.parent_span_id == span.span_id)
        r_tool = sum(s.duration_ms for s in tool_spans if s.parent_span_id == span.span_id)
        lines.append(f"| Round {rn} | {span.duration_ms:.0f}ms | {r_llm:.0f}ms | {r_tool:.0f}ms |")
    lines.append("")


def _append_errors(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("## 5. 错误清单")
    lines.append("")
    for span in trace.error_spans:
        lines.append(f"- **{span.span_type}** ({span.span_id}):")
        for event in span.events:
            if event.event_type == "error":
                lines.append(f"  - {event.message}")
                if event.data:
                    lines.append(f"  - 详情: {event.data}")
    lines.append("")


def _append_footer(lines: list[str], trace: AgentTrace) -> None:
    lines.append("---")
    lines.append("")
    lines.append(f"*报告由 TraceExporter 自动生成 | Trace ID: `{trace.trace_id}`*")


# ═══════════════════════════════════════════════════════════
# Console 导出（实时调试用）
# ═══════════════════════════════════════════════════════════


def export_console_summary(trace: AgentTrace) -> str:
    """生成适合在终端打印的简洁摘要。"""
    lines = [
        f"{'='*60}",
        "  Agent Trace 摘要",
        f"{'='*60}",
        f"  Trace ID:    {trace.trace_id}",
        f"  Agent:       {trace.agent_type}",
        f"  问题:        {trace.question[:60]}",
        f"  总耗时:      {trace.total_duration_ms:.0f} ms",
        f"  轮次:        {trace.total_rounds}",
        f"  LLM 调用:    {trace.total_llm_calls}",
        f"  工具调用:    {trace.total_tool_calls}",
        f"  事件总数:    {trace.total_events}",
        f"  错误:        {trace.error_count}",
        f"  答案长度:    {len(trace.final_answer)} 字符",
        f"{'='*60}",
        "  耗时分布:",
    ]

    llm_spans = [s for s in trace.spans if s.span_type == "llm_call"]
    tool_spans = [s for s in trace.spans if s.span_type == "tool_execution"]
    total_llm = sum(s.duration_ms for s in llm_spans)
    total_tool = sum(s.duration_ms for s in tool_spans)

    lines.append(f"    LLM 调用:   {total_llm:.0f} ms ({len(llm_spans)} 次)")
    lines.append(f"    工具执行:   {total_tool:.0f} ms ({len(tool_spans)} 次)")

    round_spans = [s for s in trace.spans if s.span_type == "react_round"]
    lines.append(f"  {'='*60}")
    lines.append("  每轮详情:")
    for span in round_spans:
        rn = span.metadata.get("round_num", "?")
        r_llm = sum(s.duration_ms for s in llm_spans if s.parent_span_id == span.span_id)
        r_tool = sum(s.duration_ms for s in tool_spans if s.parent_span_id == span.span_id)
        flags = ""
        if span.metadata.get("parse_error"):
            flags += " [PARSE_ERR]"
        if span.metadata.get("final"):
            flags += " [FINAL]"
        lines.append(
            f"    Round {rn}: {span.duration_ms:.0f}ms (LLM:{r_llm:.0f}ms Tool:{r_tool:.0f}ms){flags}"
        )

    if trace.error_count > 0:
        lines.append(f"  {'='*60}")
        lines.append("  错误:")
        for span in trace.error_spans:
            for event in span.events:
                if event.event_type == "error":
                    lines.append(f"    - {event.message}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def export_console_tree(trace: AgentTrace) -> str:
    """生成适合在终端打印的 Span 树形结构。"""
    lines = [f"Trace: {trace.trace_id} ({trace.agent_type})"]

    # 先找 root span
    root_spans = [s for s in trace.spans if s.parent_span_id is None]
    other_spans = [s for s in trace.spans if s.parent_span_id is not None]

    for root in root_spans:
        lines.append(f"+-- [{root.span_type}] {root.span_id} ({root.duration_ms:.0f}ms)")
        _render_children(lines, root.span_id, other_spans, indent="    ")

    return "\n".join(lines)


def _render_children(
    lines: list[str],
    parent_id: str,
    all_spans: list[TraceSpan],
    indent: str,
) -> None:
    children = [s for s in all_spans if s.parent_span_id == parent_id]
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "\\--" if is_last else "+--"
        error_flag = " [ERR]" if child.has_errors else ""
        lines.append(
            f"{indent}{connector} [{child.span_type}] {child.span_id} "
            f"({child.duration_ms:.0f}ms, {child.event_count} events){error_flag}"
        )
        next_indent = indent + ("    " if is_last else "|   ")
        _render_children(lines, child.span_id, all_spans, next_indent)
