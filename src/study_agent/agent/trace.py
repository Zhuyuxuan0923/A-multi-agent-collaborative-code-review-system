"""Agent Trace -- 可观测性数据模型与采集器。

Trace 是什么？
  把 Agent 的一次完整执行想象成一段"行车记录"。
  没有 trace 时：你只能看到最终输出（终点）
  有 trace 时：你可以回放每一步 -- LLM 看到了什么、输出了什么、调用了哪个工具

Trace 的三层数据结构：

  AgentTrace（一次完整执行）
  +-- TraceSpan（一个执行阶段，比如一轮 ReAct 循环）
      +-- TraceEvent（一个原子事件，比如一次 LLM 调用）

生活中的类比：
  AgentTrace = 一趟航班（北京 -> 上海）
  TraceSpan  = 一个航段（起飞、巡航、降落）
  TraceEvent = 一个瞬间（收起落架、高度 10000 英尺、放襟翼）
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════
# TraceEvent -- 原子事件
# ═══════════════════════════════════════════════════════════

EventType = Literal[
    "llm_request",  # 发送给 LLM 的 prompt
    "llm_response",  # LLM 返回的文本
    "tool_call",  # 准备调用工具
    "tool_result",  # 工具返回结果
    "parse_attempt",  # 尝试解析 LLM 输出中的 Action
    "parse_success",  # 解析成功
    "parse_failure",  # 解析失败
    "thought",  # LLM 输出的 Thought 内容
    "final_answer",  # Agent 给出最终答案
    "error",  # 任何错误
    "state_snapshot",  # 当前状态快照（prompt 的累积内容）
]


class TraceEvent(BaseModel):
    """一次原子事件 -- Trace 的最小记录单位。

    每个事件记录了"在什么时间、发生了什么、附带什么数据"。
    """

    event_type: EventType = Field(description="事件类型")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="事件发生时间 (UTC)",
    )
    message: str = Field(default="", description="人类可读的事件描述")
    data: dict[str, Any] = Field(default_factory=dict, description="事件附带的任意数据")

    def elapsed_since(self, other: TraceEvent) -> float:
        """计算本事件与另一个事件之间的毫秒差。"""
        return (self.timestamp - other.timestamp).total_seconds() * 1000


# ═══════════════════════════════════════════════════════════
# TraceSpan -- 执行阶段
# ═══════════════════════════════════════════════════════════

SpanType = Literal[
    "react_round",  # 一轮 ReAct 循环 (Thought + Action + Observation)
    "llm_call",  # 单次 LLM API 调用
    "tool_execution",  # 单次工具执行
    "plan_step",  # Plan-Execute 的计划步骤
    "execute_step",  # Plan-Execute 的执行步骤
    "root",  # 根 span（整个 Agent 执行）
]


class TraceSpan(BaseModel):
    """一个执行阶段 -- 包含一组相关事件的容器。

    每个 span 有开始和结束时间，可以嵌套（parent_span_id）。
    Span 内部的 events 按时间顺序记录了阶段内发生的所有事情。
    """

    span_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Span 唯一标识",
    )
    span_type: SpanType = Field(description="Span 类型")
    parent_span_id: str | None = Field(default=None, description="父 Span ID（用于嵌套关系）")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Span 开始时间",
    )
    end_time: datetime | None = Field(default=None, description="Span 结束时间（未结束时为 None）")
    events: list[TraceEvent] = Field(default_factory=list, description="Span 内的事件列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Span 级别的元数据")

    @property
    def duration_ms(self) -> float:
        """Span 持续时长（毫秒）。"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def event_count(self) -> int:
        """事件总数。"""
        return len(self.events)

    @property
    def has_errors(self) -> bool:
        """是否包含错误事件。"""
        return any(e.event_type == "error" for e in self.events)

    def add_event(
        self, event_type: EventType, message: str = "", data: dict[str, Any] | None = None
    ) -> TraceEvent:
        """向 Span 添加一个事件。"""
        event = TraceEvent(
            event_type=event_type,
            message=message,
            data=data or {},
        )
        self.events.append(event)
        return event

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        """标记 Span 结束。"""
        self.end_time = datetime.now(UTC)
        if metadata:
            self.metadata.update(metadata)


# ═══════════════════════════════════════════════════════════
# AgentTrace -- 完整执行记录
# ═══════════════════════════════════════════════════════════


class AgentTrace(BaseModel):
    """一次完整的 Agent 执行记录。

    这是 Trace 系统的顶层数据结构。一次 agent.run(question) 调用
    产生一个 AgentTrace 实例，包含所有的 span 和 event。
    """

    trace_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description="Trace 唯一标识",
    )
    agent_type: str = Field(default="unknown", description="Agent 类型名称")
    question: str = Field(default="", description="用户问题")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="执行开始时间",
    )
    end_time: datetime | None = Field(default=None, description="执行结束时间")
    spans: list[TraceSpan] = Field(default_factory=list, description="所有 Span")
    final_answer: str = Field(default="", description="最终答案")
    metadata: dict[str, Any] = Field(default_factory=dict, description="执行级别的元数据")

    # -- 聚合统计 --

    @property
    def total_llm_calls(self) -> int:
        """LLM 调用总次数。"""
        return sum(1 for s in self.spans if s.span_type == "llm_call")

    @property
    def total_tool_calls(self) -> int:
        """工具调用总次数。"""
        return sum(1 for s in self.spans if s.span_type == "tool_execution")

    @property
    def total_rounds(self) -> int:
        """ReAct 循环总轮次。"""
        return sum(1 for s in self.spans if s.span_type == "react_round")

    @property
    def total_events(self) -> int:
        """所有事件总数。"""
        return sum(s.event_count for s in self.spans)

    @property
    def total_duration_ms(self) -> float:
        """总耗时（毫秒）。"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def error_spans(self) -> list[TraceSpan]:
        """包含错误的 Span 列表。"""
        return [s for s in self.spans if s.has_errors]

    @property
    def error_count(self) -> int:
        """错误总数。"""
        return len(self.error_spans)

    @property
    def avg_round_duration_ms(self) -> float:
        """平均每轮耗时（毫秒）。"""
        rounds = [s for s in self.spans if s.span_type == "react_round"]
        if not rounds:
            return 0.0
        return sum(r.duration_ms for r in rounds) / len(rounds)

    # -- 操作方法 --

    def create_span(
        self,
        span_type: SpanType,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """创建并添加一个 Span。返回 Span 对象供后续操作。"""
        span = TraceSpan(
            span_type=span_type,
            parent_span_id=parent_span_id,
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def finish(self, final_answer: str = "", metadata: dict[str, Any] | None = None) -> None:
        """标记 Trace 结束。"""
        self.end_time = datetime.now(UTC)
        self.final_answer = final_answer
        if metadata:
            self.metadata.update(metadata)

    def summary(self) -> dict[str, Any]:
        """生成 Trace 摘要，用于快速了解执行概况。"""
        return {
            "trace_id": self.trace_id,
            "agent_type": self.agent_type,
            "question": self.question[:80],
            "duration_ms": round(self.total_duration_ms, 1),
            "rounds": self.total_rounds,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "events": self.total_events,
            "errors": self.error_count,
            "answer_length": len(self.final_answer),
        }

    def timeline(self) -> list[dict[str, Any]]:
        """生成按时间排序的事件时间线（扁平化）。"""
        result: list[dict[str, Any]] = []
        for span in self.spans:
            for event in span.events:
                result.append(
                    {
                        "time_offset_ms": round(
                            (event.timestamp - self.start_time).total_seconds() * 1000, 1
                        ),
                        "span_id": span.span_id,
                        "span_type": span.span_type,
                        "event_type": event.event_type,
                        "message": event.message,
                    }
                )
        result.sort(key=lambda x: x["time_offset_ms"])
        return result


# ═══════════════════════════════════════════════════════════
# TraceCollector -- 采集器
# ═══════════════════════════════════════════════════════════


class TraceCollector:
    """Trace 采集器 -- Agent 执行过程中的"行车记录仪"。

    TraceCollector 提供了简洁的 API，Agent 在执行的各个关键节点
    调用对应方法，采集器自动记录时间、创建 Span、关联事件。

    使用方式：
        collector = TraceCollector()
        collector.start_trace("What is React 19?", "ReactAgent")

        span_id = collector.start_span("llm_call")
        collector.add_event(span_id, "llm_request", "Sending prompt...",
                            {"prompt_length": 500})
        # ... LLM 调用 ...
        collector.add_event(span_id, "llm_response", "Got response",
                            {"response_length": 200, "tokens": 150})
        collector.end_span(span_id)

        trace = collector.finish_trace("React 19 is...")
    """

    def __init__(self):
        self._trace: AgentTrace | None = None
        self._active_spans: dict[str, TraceSpan] = {}

    @property
    def trace(self) -> AgentTrace | None:
        return self._trace

    # -- Trace 生命周期 --

    def start_trace(
        self, question: str, agent_type: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """开始一次新的 Trace。返回 trace_id。"""
        self._trace = AgentTrace(
            question=question,
            agent_type=agent_type,
            metadata=metadata or {},
        )
        # 创建一个 root span 代表整个执行
        root_span = self._trace.create_span(span_type="root")
        self._active_spans[root_span.span_id] = root_span
        return self._trace.trace_id

    def finish_trace(
        self, final_answer: str = "", metadata: dict[str, Any] | None = None
    ) -> AgentTrace:
        """结束 Trace，关闭所有未关闭的 Span。返回完整的 AgentTrace。"""
        if self._trace is None:
            raise RuntimeError("Trace 尚未开始，请先调用 start_trace()")

        # 关闭所有活跃的 span
        for span in list(self._active_spans.values()):
            if span.end_time is None:
                span.finish()

        self._trace.finish(final_answer=final_answer, metadata=metadata)
        return self._trace

    # -- Span 生命周期 --

    def start_span(
        self,
        span_type: SpanType,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """开始一个新的 Span。返回 span_id。"""
        if self._trace is None:
            raise RuntimeError("Trace 尚未开始，请先调用 start_trace()")

        span = self._trace.create_span(
            span_type=span_type,
            parent_span_id=parent_span_id,
            metadata=metadata,
        )
        self._active_spans[span.span_id] = span
        return span.span_id

    def end_span(self, span_id: str, metadata: dict[str, Any] | None = None) -> None:
        """结束一个 Span。"""
        span = self._active_spans.pop(span_id, None)
        if span is None:
            return
        span.finish(metadata=metadata)

    # -- 事件记录 --

    def add_event(
        self,
        span_id: str,
        event_type: EventType,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """向指定 Span 添加一个事件。"""
        span = self._active_spans.get(span_id)
        if span is None:
            raise ValueError(f"Span {span_id} 不存在或已关闭")
        return span.add_event(event_type, message=message, data=data)

    # -- 便捷方法 --

    def record_llm_call(
        self,
        parent_span_id: str,
        prompt: str,
        response: str,
        duration_ms: float,
        token_count: int = 0,
        model: str = "",
    ) -> str:
        """记录一次完整的 LLM 调用（创建 span + 两个事件）。"""
        span_id = self.start_span("llm_call", parent_span_id=parent_span_id)
        self.add_event(
            span_id,
            "llm_request",
            "发送 LLM 请求",
            {
                "prompt_preview": prompt[:200],
                "prompt_length": len(prompt),
            },
        )
        self.add_event(
            span_id,
            "llm_response",
            "收到 LLM 响应",
            {
                "response_preview": response[:200],
                "response_length": len(response),
                "token_count": token_count,
                "model": model,
                "duration_ms": duration_ms,
            },
        )
        self.end_span(span_id, {"duration_ms": duration_ms, "token_count": token_count})
        return span_id

    def record_tool_call(
        self,
        parent_span_id: str,
        tool_name: str,
        params: dict[str, Any],
        result: str,
        duration_ms: float,
        success: bool = True,
    ) -> str:
        """记录一次工具调用（创建 span + 两个事件）。"""
        span_id = self.start_span("tool_execution", parent_span_id=parent_span_id)
        self.add_event(
            span_id,
            "tool_call",
            f"调用工具: {tool_name}",
            {
                "tool_name": tool_name,
                "params": params,
            },
        )
        event_type: EventType = "tool_result" if success else "error"
        self.add_event(
            span_id,
            event_type,
            f"工具返回: {result[:100]}",
            {
                "result_preview": result[:200],
                "result_length": len(result),
                "success": success,
            },
        )
        self.end_span(
            span_id,
            {
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "success": success,
            },
        )
        return span_id

    def record_parse_attempt(
        self,
        span_id: str,
        raw_output: str,
        parse_success: bool,
        parsed_data: dict[str, Any] | None = None,
    ) -> None:
        """记录一次 Action 解析尝试。"""
        if parse_success:
            self.add_event(
                span_id,
                "parse_success",
                "解析成功",
                {
                    "raw_preview": raw_output[:100],
                    "parsed": parsed_data or {},
                },
            )
        else:
            self.add_event(
                span_id,
                "parse_failure",
                "解析失败",
                {
                    "raw_preview": raw_output[:100],
                },
            )
