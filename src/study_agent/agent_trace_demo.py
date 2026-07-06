"""Agent Trace 可观测性演示。

演示内容：
  1. 真实运行 TracedReactAgent（需要 API Key）
  2. 展示 Console 摘要输出
  3. 导出 JSON 格式 Trace
  4. 导出 Markdown 格式报告
  5. 展示事件时间线
  6. 展示 Span 树形结构
  7. 模拟 Trace 数据（无需 API Key 也能跑）

运行方式：
  # 完整模式（需要 API Key）
  python -m study_agent.agent_trace_demo

  # 仅模拟模式（无需 API Key）
  python -m study_agent.agent_trace_demo --mock-only
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════
# 模拟 Trace 数据（不需要 API Key 也能看到 Trace 的样子）
# ═══════════════════════════════════════════════════════════


def create_mock_trace() -> Any:
    """手动构建一个模拟的 AgentTrace，展示 Trace 的数据结构。

    这个函数不调用任何 LLM，纯手工构造 Trace 数据。
    用途：当没有 API Key 或网络不通时，也能看到 Trace 长什么样。
    """
    from study_agent.agent.trace import TraceCollector

    collector = TraceCollector()
    collector.start_trace(
        question="React 19 有哪些新特性？",
        agent_type="TracedReactAgent (模拟)",
    )

    # Round 1
    r1_id = collector.start_span("react_round", metadata={"round_num": 1})

    # LLM 调用
    llm1_id = collector.record_llm_call(
        parent_span_id=r1_id,
        prompt="[System Prompt + Question: React 19 有哪些新特性？]",
        response='Thought: 我需要搜索 React 19 的新特性\nAction: search(query="React 19")',
        duration_ms=1250.5,
        token_count=450,
        model="deepseek-chat",
    )
    collector.add_event(r1_id, "thought", "我需要搜索 React 19 的新特性")

    # 解析
    collector.record_parse_attempt(
        r1_id,
        '...Action: search(query="React 19")',
        True,
        {"tool": "search", "params": {"query": "React 19"}},
    )

    # 工具
    collector.record_tool_call(
        parent_span_id=r1_id,
        tool_name="search",
        params={"query": "React 19"},
        result="React 19 于 2024 年 12 月发布，主要新特性：Server Components、Actions、Document Metadata...",
        duration_ms=15.2,
        success=True,
    )

    collector.end_span(r1_id)

    # Round 2
    r2_id = collector.start_span("react_round", metadata={"round_num": 2})

    collector.record_llm_call(
        parent_span_id=r2_id,
        prompt="[System Prompt + Round 1 上下文]",
        response="Thought: 已经获得 React 19 的详细信息，可以总结了\nFinal Answer: React 19 的主要新特性包括...",
        duration_ms=980.3,
        token_count=380,
        model="deepseek-chat",
    )
    collector.add_event(r2_id, "thought", "已经获得足够信息，可以总结答案")
    collector.add_event(
        r2_id,
        "final_answer",
        "React 19 的主要新特性包括...",
        {"round_num": 2, "answer_length": 250},
    )
    collector.end_span(r2_id, {"final": True})

    return collector.finish_trace(
        final_answer="React 19 于 2024 年 12 月发布，主要新特性包括："
        "1) Server Components 稳定版；2) Actions 机制；"
        "3) Document Metadata 原生支持；4) 改进的 ref 处理；"
        "5) use() hook。性能方面，客户端 bundle 减小约 20%。"
    )


# ═══════════════════════════════════════════════════════════
# 真实运行（需要 API Key）
# ═══════════════════════════════════════════════════════════


def run_real_trace(question: str | None = None) -> Any:
    """使用真实的 LLM API 运行 TracedReactAgent 并返回 Trace。"""
    from study_agent.agent.traced_agent import TracedReactAgent
    from study_agent.llm.client import LLMClient

    # 尝试多个 provider，找到第一个可用的
    provider = None
    for p in ["deepseek", "openai", "anthropic", "zhipu"]:
        env_key = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
        }.get(p, "")
        if os.getenv(env_key):
            provider = p
            break

    if provider is None:
        print("[WARN] 没有找到可用的 API Key，切换到模拟模式")
        return None

    print(f"[OK] 使用 provider: {provider}")

    try:
        client = LLMClient(provider=provider)
        agent = TracedReactAgent(client, max_rounds=5, parse_method="regex")

        question = question or "React 19 有哪些新特性？"
        print(f"[OK] 问题: {question}")
        print(f"[OK] 开始执行（最多 {agent.max_rounds} 轮）...")
        print(f"{'='*60}")

        answer = agent.run(question)

        print(f"{'='*60}")
        print("[OK] 执行完成")
        print("")

        return agent.trace

    except Exception as e:
        print(f"[ERR] 真实运行失败: {e}")
        print("[提示] 切换到模拟模式查看 Trace 结构")
        return None


# ═══════════════════════════════════════════════════════════
# 展示 Trace
# ═══════════════════════════════════════════════════════════


def display_trace(trace: Any) -> None:
    """展示 Trace 的多种视图。"""
    from study_agent.agent.trace_exporter import (
        export_console_summary,
        export_console_tree,
    )

    # 1. Console 摘要
    print(export_console_summary(trace))
    print()

    # 2. Span 树
    print("Span 树形结构:")
    print(export_console_tree(trace))
    print()

    # 3. 事件时间线（前 15 条）
    print("事件时间线 (前 15 条):")
    print(f"{'偏移':>8}  {'Span类型':<16} {'事件类型':<16} 描述")
    print(f"{'-'*8}  {'-'*16} {'-'*16}  {'-'*30}")
    timeline = trace.timeline()
    for entry in timeline[:15]:
        print(
            f"{entry['time_offset_ms']:7.0f}ms "
            f"{entry['span_type']:<16} "
            f"{entry['event_type']:<16} "
            f"{entry['message'][:50]}"
        )
    if len(timeline) > 15:
        print(f"  ... 还有 {len(timeline) - 15} 条事件")
    print()

    # 4. 统计信息
    print("统计:")
    print(f"  Spans: {len(trace.spans)}")
    for span_type in ["root", "react_round", "llm_call", "tool_execution"]:
        count = sum(1 for s in trace.spans if s.span_type == span_type)
        if count > 0:
            total_ms = sum(s.duration_ms for s in trace.spans if s.span_type == span_type)
            print(f"    {span_type}: {count} 个, 总耗时 {total_ms:.0f}ms")


def save_trace_outputs(trace: Any) -> None:
    """保存 Trace 的 JSON 和 Markdown 导出文件。"""
    from study_agent.agent.trace_exporter import (
        export_json_file,
        export_markdown_file,
    )

    output_dir = Path(__file__).parent.parent.parent / "docs" / "week6"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "trace_output.json"
    export_json_file(trace, str(json_path))
    print(f"[OK] JSON 导出 -> {json_path}")

    md_path = output_dir / "trace_report.md"
    export_markdown_file(trace, str(md_path))
    print(f"[OK] Markdown 报告 -> {md_path}")


# ═══════════════════════════════════════════════════════════
# 对比：有无 Trace 的调试体验
# ═══════════════════════════════════════════════════════════


def show_comparison() -> None:
    """展示有 Trace 和没有 Trace 时，调试 Agent 的体验差异。"""
    print(
        """
+==============================================================================+
|  对比：有 Trace vs 无 Trace 的调试体验                                          |
+==============================================================================+

场景：用户问 "React 19 有哪些新特性？"，Agent 调用 search 工具，返回了答案。
     但你发现答案里漏掉了 "use() hook" 这个特性。你需要排查为什么漏了。

--- 无 Trace 时（只有 logger 输出）---

  [仅有的线索]
  INFO:__main__:ReAct Round 1/8
  INFO:__main__:  工具 search({'query': 'React 19'}) -> React 19 于 2024年...
  INFO:__main__:ReAct Round 2/8
  INFO:__main__:ReAct 循环完成, 共 2 轮

  你能回答这些问题吗？
  Q1: 第 1 轮 LLM 的 prompt 里有没有包含足够的关键词引导？
  Q2: search 工具返回的结果里有没有 "use() hook"？
  Q3: 第 2 轮 LLM 的输出中，Thought 有没有提到 use() hook？
  Q4: 总共 2 轮是否合理？每轮耗时多少？

  答案：全都回答不了。你只能看最终输出 + 零星的日志行。
       排查 = 加 print() -> 重新运行 -> 再加 print() -> 再运行...（每次等几秒到几十秒）

--- 有 Trace 时 ---

  打开 trace_report.json 或 trace_report.md，你可以：

  [查 prompt]
  -> llm_call span -> llm_request event -> data.prompt_preview
  A1: 看到第 1 轮 prompt 的完整内容，确认 system prompt 有没有引导"列出所有特性"
     如果 prompt 里没提到，就找到了根因 -- system prompt 需要改进

  [查工具返回]
  -> tool_execution span -> tool_result event -> data.result_preview
  A2: 看到 search 工具返回的完整内容，确认返回里有没有 "use() hook"
     如果返回里有但 LLM 没提取到，说明是 LLM 的摘要能力问题

  [查 LLM 推理]
  -> react_round span -> thought event
  A3: 看到第 2 轮 LLM 的 Thought 内容，确认推理过程有没有遗漏 use() hook

  [查性能]
  -> span 的 duration_ms 属性
  A4: 看到 Round 1 耗时 1250ms（LLM:1200ms + Tool:15ms），Round 2 耗时 980ms
     总共 2.2 秒完成 -- 性能合理

  [查时间线]
  -> trace.timeline()
  看到完整的事件序列，从 LLM 请求到工具结果到最终答案，精确到毫秒

总结：
  无 Trace = 黑盒调试，靠猜 + print()
  有 Trace = 白盒调试，每个决策点都有结构化数据可查

  这就是为什么"可观测性"是 Agent 工程化的基础设施。
"""
    )

    input("按 Enter 继续...")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Agent Trace 可观测性演示")
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="仅使用模拟数据，不调用真实 LLM API",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="自定义问题（默认: React 19 有哪些新特性？）",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="跳过对比说明",
    )
    args = parser.parse_args()

    print("Agent Trace 可观测性演示")
    print(f"时间: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    trace = None

    if not args.mock_only:
        print("--- 模式: 真实 LLM 调用 ---")
        print()
        trace = run_real_trace(args.question)

    if trace is None:
        if not args.mock_only:
            print()
            print("--- 切换到模拟模式 ---")
            print()
        else:
            print("--- 模式: 模拟数据 ---")
            print()
        trace = create_mock_trace()
        print("[OK] 模拟 Trace 已生成")
        print()

    # 展示 Trace
    print("=" * 60)
    print("  Trace 内容展示")
    print("=" * 60)
    print()
    display_trace(trace)

    # 保存导出文件
    print()
    save_trace_outputs(trace)

    # 对比说明
    if not args.skip_comparison:
        print()
        show_comparison()

    print()
    print("[OK] 演示完成!")
    print()
    print("生成的文件:")
    print("  docs/week6/trace_output.json  -- 完整 Trace JSON")
    print("  docs/week6/trace_report.md    -- 人类可读的 Markdown 报告")
    print()
    print("动手试试:")
    print("  1. 打开 trace_output.json，找到 llm_request 事件，看 prompt 内容")
    print("  2. 打开 trace_report.md，看时间线和逐轮详情")
    print("  3. 改一下 question，看 Trace 有什么不同")
    print("  4. 对比不同 provider 的耗时（如果有多个 API Key）")


if __name__ == "__main__":
    main()
