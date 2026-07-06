"""TracedReactAgent -- 带 Trace 追踪的 ReAct Agent。

这个模块展示了"可观测性"如何嵌入到 Agent 中。
TracedReactAgent 继承自 ReactAgent，在每个关键节点自动记录 Trace：

  1. Agent 启动      -> 创建 AgentTrace
  2. 每轮循环开始    -> 创建 react_round Span
  3. 调用 LLM        -> 创建 llm_call Span，记录 prompt/response/耗时
  4. 解析 Action     -> 记录 parse_success 或 parse_failure 事件
  5. 执行工具        -> 创建 tool_execution Span，记录工具名/参数/结果/耗时
  6. 找到最终答案    -> 记录 final_answer 事件
  7. Agent 结束      -> 关闭所有 Span，finish Trace

对比普通 ReactAgent：
  - 普通版：只能看到 logger 输出，无法结构化分析
  - Traced 版：每次执行产生完整的 AgentTrace，可导出 JSON/Markdown

设计原则：
  - 不修改 ReactAgent 的任何代码（开闭原则）
  - Trace 失败不影响 Agent 执行（可观测性不能成为故障点）
  - TraceCollector 的 API 足够简洁，5 个方法覆盖全部场景
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from study_agent.agent.react_agent import ReactAgent, execute_tool
from study_agent.agent.trace import AgentTrace, TraceCollector

logger = logging.getLogger(__name__)


class TracedReactAgent(ReactAgent):
    """带 Trace 追踪的 ReAct Agent。

    用法和 ReactAgent 完全一样，唯一的区别是：
      agent.trace 属性在 run() 之后包含完整的执行记录。
    """

    def __init__(self, client: Any, max_rounds: int = 8, parse_method: str = "regex"):
        super().__init__(client, max_rounds=max_rounds, parse_method=parse_method)
        self._collector = TraceCollector()
        self._last_trace: AgentTrace | None = None

    @property
    def trace(self) -> AgentTrace | None:
        """最近一次执行的完整 Trace。run() 之前为 None。"""
        return self._last_trace

    @property
    def collector(self) -> TraceCollector:
        """获取 TraceCollector 实例（用于自定义事件记录）。"""
        return self._collector

    def run(self, question: str) -> str:
        """执行 ReAct 循环（带 Trace 追踪）。

        和父类 ReactAgent.run() 的逻辑完全一致，
        仅在每个关键节点插入了 trace 记录。
        """
        # --- 开始 Trace ---
        self._collector.start_trace(
            question=question,
            agent_type="TracedReactAgent",
            metadata={
                "max_rounds": self.max_rounds,
                "parse_method": self.parse_action.__name__,
                "tool_count": len(self.system_prompt),
            },
        )

        current_prompt = f"Question: {question}\nThought: "
        final_answer = ""

        for round_num in range(1, self.max_rounds + 1):
            # --- 开始一轮 ReAct ---
            round_span_id = self._collector.start_span(
                "react_round",
                metadata={"round_num": round_num},
            )
            logger.info("ReAct Round %d/%d", round_num, self.max_rounds)

            full_prompt = self.system_prompt + "\n\n" + current_prompt

            # --- LLM 调用（带计时）---
            llm_start = time.perf_counter()
            response = self._call_llm(full_prompt)
            llm_duration_ms = (time.perf_counter() - llm_start) * 1000

            if not response.strip().startswith("Thought:"):
                response = "Thought: " + response

            # 记录 LLM 调用
            self._collector.record_llm_call(
                parent_span_id=round_span_id,
                prompt=full_prompt,
                response=response,
                duration_ms=llm_duration_ms,
                model=getattr(getattr(self.client, "model", None), "__str__", lambda: "unknown")(),
            )

            # 记录 Thought（提取 Thought: 到 Action: 或 Final Answer: 之间的内容）
            thought_match = re.search(
                r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)",
                response,
                re.DOTALL,
            )
            if thought_match:
                self._collector.add_event(
                    round_span_id,
                    "thought",
                    message=thought_match.group(1).strip()[:200],
                )

            logger.info("LLM 原始输出:\n%s", response[:300])

            # --- 检查 Final Answer ---
            final_answer = self._extract_final_answer(response)
            if final_answer:
                self.last_round_count = round_num
                self._collector.add_event(
                    round_span_id,
                    "final_answer",
                    message=final_answer[:200],
                    data={"round_num": round_num, "answer_length": len(final_answer)},
                )
                self._collector.end_span(round_span_id, {"final": True})
                logger.info("ReAct 循环完成, 共 %d 轮", round_num)
                break

            # --- 解析 Action ---
            parsed = self.parse_action(response)
            self._collector.record_parse_attempt(
                span_id=round_span_id,
                raw_output=response,
                parse_success=parsed is not None,
                parsed_data=parsed if parsed else None,
            )

            if parsed is None:
                logger.warning("无法解析 Action，LLM 输出:\n%s", response[:200])
                self._collector.add_event(
                    round_span_id,
                    "error",
                    message="无法解析 Action",
                    data={"response_preview": response[:200]},
                )
                current_prompt += response + "\n"
                current_prompt += (
                    'Observation: 格式错误。请使用 Action: tool_name(key="value") 格式。\n'
                )
                current_prompt += "Thought: "
                self._collector.end_span(round_span_id, {"parse_error": True})
                continue

            # --- 执行工具 ---
            tool_name = parsed.get("tool", "")
            params = parsed.get("params") or parsed.get("args_str") or {}
            if isinstance(params, str):
                from study_agent.agent.react_agent import _parse_naive_args

                params = _parse_naive_args(params)

            tool_start = time.perf_counter()
            observation = execute_tool(tool_name, params)
            tool_duration_ms = (time.perf_counter() - tool_start) * 1000

            # 记录工具调用
            self._collector.record_tool_call(
                parent_span_id=round_span_id,
                tool_name=tool_name,
                params=params,
                result=observation,
                duration_ms=tool_duration_ms,
                success=not observation.startswith("错误"),
            )

            logger.info("  工具 %s(%s) -> %s", tool_name, params, observation[:80])

            # --- 截断 Action 之后的内容，追加 Observation ---
            action_until = response
            action_match = re.search(r"(Action:\s*[^\n]+)", response)
            if action_match:
                action_until = response[: action_match.end()]
            current_prompt += action_until + "\n"
            current_prompt += f"Observation: {observation}\n"
            current_prompt += "Thought: "

            # 记录状态快照（当前 prompt 长度）
            self._collector.add_event(
                round_span_id,
                "state_snapshot",
                message=f"Round {round_num} 完成, prompt 长度: {len(current_prompt)}",
                data={"prompt_length": len(current_prompt)},
            )

            # --- 结束本轮 ---
            self._collector.end_span(round_span_id)

        # --- 处理超时 ---
        if not final_answer:
            final_answer = f"经过 {self.max_rounds} 轮思考仍未得出最终答案。请简化问题。"
            self._collector.add_event(
                round_span_id if "round_span_id" in dir() else "",
                "error",
                message="达到最大轮次限制",
                data={"max_rounds": self.max_rounds},
            )

        # --- 结束 Trace ---
        self._last_trace = self._collector.finish_trace(
            final_answer=final_answer,
            metadata={
                "total_rounds": self.last_round_count,
                "max_rounds": self.max_rounds,
            },
        )

        return final_answer

    def get_trace(self) -> dict[str, Any]:
        """返回最后一次运行的 Trace 摘要（兼容旧接口）。"""
        if self._last_trace is None:
            return {"error": "尚未执行 run()，没有 Trace 数据"}
        return self._last_trace.summary()

    def get_trace_json(self, indent: int = 2) -> str:
        """返回最后一次运行的 Trace 的 JSON 字符串。"""
        if self._last_trace is None:
            return '{"error": "no trace data"}'
        return self._last_trace.model_dump_json(indent=indent)

    def get_timeline(self) -> list[dict[str, Any]]:
        """返回按时间排序的事件时间线。"""
        if self._last_trace is None:
            return []
        return self._last_trace.timeline()
