"""Agent 状态管理 —— 会话状态建模 + 并发会话管理。

本模块解决三个问题：
  1. Agent 在做什么、做到了哪一步？（AgentState）
  2. 多轮对话中如何保持上下文一致？（状态持久化）
  3. 多个用户同时使用时如何不串号？（SessionManager 并发隔离）

概念区分（重要）：
  Memory  = "之前聊了什么"  → 存历史对话（Buffer/Summary/Vector）
  State   = "现在做到哪了"  → 存当前任务进度、中间结果
  Session = "谁在跟谁聊"    → 一个用户的一次连续对话

类比：你在写作业
  Memory  = 之前所有的作业本
  State   = 当前这道题做到了第几步
  Session = 今天下午坐在书桌前这一段连续的写作业时间
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════


class AgentStatus(str, Enum):
    """Agent 当前在干什么的状态枚举。

    str + Enum 混用 = 既能做枚举比较，又能直接序列化成 JSON 字符串。
    这是 Pydantic 推荐的做法。
    """

    IDLE = "idle"  # 空闲，等待用户输入
    THINKING = "thinking"  # 正在推理/思考
    ACTING = "acting"  # 正在执行工具调用
    WAITING_USER = "waiting_user"  # 等待用户确认（如危险操作）
    COMPLETED = "completed"  # 任务完成
    ERROR = "error"  # 出错


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════


class ToolCallRecord(BaseModel):
    """一次工具调用的完整记录。

    为什么需要这个？
      当 Agent 调了 5 次工具后，你需要知道：
      - 每次调用是什么时候？
      - 输入了什么参数？
      - 返回了什么结果？
      - 有没有出错？
      这个模型把每次调用都结构化记录下来。
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    called_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    duration_ms: float = 0.0


class StepRecord(BaseModel):
    """任务中一个步骤的执行记录。"""

    step_index: int  # 第几步（从 1 开始）
    description: str  # 这一步做什么
    status: str = "pending"  # pending | running | done | failed
    result: str | None = None  # 这一步的输出
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class AgentState(BaseModel):
    """Agent 会话状态 —— 一次完整对话中 Agent 需要的所有"工作记忆"。

    设计原则：
      1. 所有字段都有默认值 → 创建新状态时只传 session_id 和 task 即可
      2. 使用 Pydantic → 自动校验类型，序列化/反序列化免费
      3. 状态可变 → 通过 SessionManager 的方法来更新，不直接改字段

    字段分组说明：
      [身份]   session_id, created_at, updated_at
      [任务]   task_description, status, steps
      [对话]   messages（用户和 AI 的完整对话）
      [工具]   tool_calls（所有工具调用记录）
      [记忆]   memory_snapshot（Memory 层的上下文快照）
      [元数据] metadata（扩展字段，存什么都行）
    """

    # ── 身份信息 ──
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="会话唯一 ID，12 位十六进制字符串",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="会话创建时间（ISO 8601 格式）",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="最后一次状态更新时间",
    )

    # ── 任务信息 ──
    task_description: str = Field(
        default="",
        description="当前任务描述，如 '帮用户调试 Python 代码'",
    )
    status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="Agent 当前状态",
    )
    steps: list[StepRecord] = Field(
        default_factory=list,
        description="任务步骤列表，记录每一步的执行状态",
    )

    # ── 对话信息 ──
    messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="对话历史，每项为 {'role': 'user'|'assistant', 'content': '...'}",
    )

    # ── 工具调用 ──
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="本次会话中所有工具调用的完整记录",
    )

    # ── Memory 快照 ──
    memory_snapshot: str = Field(
        default="",
        description="Memory 层提供的上下文快照，每次对话前更新",
    )

    # ── 扩展字段 ──
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据，存什么都行（用户偏好、临时变量等）",
    )

    # ── 便捷方法 ──

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到对话历史，自动更新时间戳。"""
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now(UTC).isoformat()

    def add_tool_call(self, record: ToolCallRecord) -> None:
        """添加一条工具调用记录。"""
        self.tool_calls.append(record)
        self.updated_at = datetime.now(UTC).isoformat()

    def add_step(self, description: str) -> int:
        """添加一个任务步骤，返回步骤编号（从 1 开始）。"""
        idx = len(self.steps) + 1
        self.steps.append(StepRecord(step_index=idx, description=description))
        self.updated_at = datetime.now(UTC).isoformat()
        return idx

    def mark_step_done(self, step_index: int, result: str = "") -> None:
        """标记某个步骤为完成。"""
        for s in self.steps:
            if s.step_index == step_index:
                s.status = "done"
                s.result = result
                break
        self.updated_at = datetime.now(UTC).isoformat()

    def set_status(self, status: AgentStatus) -> None:
        """更新 Agent 状态。"""
        self.status = status
        self.updated_at = datetime.now(UTC).isoformat()

    def touch(self) -> None:
        """更新 updated_at 时间戳（表示会话仍然活跃）。"""
        self.updated_at = datetime.now(UTC).isoformat()

    @property
    def round_count(self) -> int:
        """对话轮数（一轮 = user + assistant 各一次）。"""
        return len(self.messages) // 2

    @property
    def total_tool_calls(self) -> int:
        """总工具调用次数。"""
        return len(self.tool_calls)

    def summary(self) -> str:
        """生成一行状态摘要，方便调试和日志。"""
        return (
            f"[{self.session_id}] status={self.status.value} "
            f"rounds={self.round_count} tools={self.total_tool_calls} "
            f"task={self.task_description[:30]}"
        )


# ═══════════════════════════════════════════════════════════════
# 会话管理器
# ═══════════════════════════════════════════════════════════════


class SessionManager:
    """并发会话管理器 —— 管理多个同时进行的 Agent 会话。

    核心职责：
      1. 创建/获取/更新/删除会话
      2. 多个会话之间完全隔离（用户 A 的状态不影响用户 B）
      3. 线程安全（Lock 保护，防止并发修改导致数据错乱）

    什么是"线程安全"？
      假设两个用户同时发消息：
        用户 A → 更新 session_1
        用户 B → 更新 session_2
      如果不用 Lock，这两个操作可能同时修改内部数据结构，
      导致 session_1 的数据被 session_2 覆盖（数据竞争）。

      Lock = 给操作加一把"门锁"，同一时刻只让一个人进去操作。
      类比：公共厕所的门锁——进去之后锁门，外面的人得等。

    使用方式：
        mgr = SessionManager()
        sid = mgr.create_session(task="调试 Python 代码")
        state = mgr.get_session(sid)
        state.add_message("user", "我的代码报错了")
        mgr.update_session(sid, state)
    """

    def __init__(self, max_sessions: int = 100, session_ttl_seconds: int = 3600) -> None:
        """创建会话管理器。

        max_sessions       — 最多允许多少个并发会话（默认 100）
        session_ttl_seconds — 会话多久不活动就自动清理（默认 3600 秒 = 1 小时）
        """
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
        self._sessions: dict[str, AgentState] = {}
        self._lock = threading.Lock()

    # ── CRUD 操作 ──────────────────────────────────────────

    def create_session(self, task: str = "", metadata: dict[str, Any] | None = None) -> str:
        """创建一个新会话，返回 session_id。

        如果会话数已达上限，自动清理过期会话腾出空间。
        """
        with self._lock:
            # 到达上限时，先清理一波过期会话
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired()

            # 如果清理后还是满的，拒绝创建
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"会话数已达上限 ({self._max_sessions})，请等待旧会话过期或手动删除"
                )

            state = AgentState(
                task_description=task,
                metadata=metadata or {},
            )
            self._sessions[state.session_id] = state
            return state.session_id

    def get_session(self, session_id: str) -> AgentState | None:
        """获取指定会话的状态。返回 None 表示不存在或已过期。"""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None

            # 检查是否过期
            if self._is_expired(state):
                del self._sessions[session_id]
                return None

            return state

    def update_session(self, session_id: str, state: AgentState) -> bool:
        """更新会话状态。返回 True 表示成功，False 表示会话不存在。"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            state.touch()
            self._sessions[session_id] = state
            return True

    def delete_session(self, session_id: str) -> bool:
        """删除会话。返回 True 表示成功删除，False 表示不存在。"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有活跃会话的简要信息（不返回完整状态，节省内存）。"""
        with self._lock:
            self._cleanup_expired()
            result = []
            for sid, state in self._sessions.items():
                result.append(
                    {
                        "session_id": sid,
                        "task": state.task_description,
                        "status": state.status.value,
                        "rounds": state.round_count,
                        "tools": state.total_tool_calls,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                    }
                )
            return result

    # ── 批量操作 ──────────────────────────────────────────

    def get_or_create(
        self, session_id: str | None = None, task: str = ""
    ) -> tuple[str, AgentState]:
        """获取已有会话，如果不存在或没传 session_id 就创建新的。

        这是最常用的方法——调用方不需要关心是新建还是复用。
        返回 (session_id, AgentState)。
        """
        if session_id:
            state = self.get_session(session_id)
            if state is not None:
                return session_id, state

        new_id = self.create_session(task=task)
        state = self.get_session(new_id)
        assert state is not None
        return new_id, state

    def cleanup(self) -> int:
        """强制清理所有过期会话，返回清理数量。"""
        with self._lock:
            return self._cleanup_expired()

    def count(self) -> int:
        """当前活跃会话数。"""
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)

    # ── 内部方法 ──────────────────────────────────────────

    def _is_expired(self, state: AgentState) -> bool:
        """判断会话是否过期。"""
        try:
            updated = datetime.fromisoformat(state.updated_at)
            elapsed = (datetime.now(UTC) - updated).total_seconds()
            return elapsed > self._session_ttl
        except (ValueError, TypeError):
            return False

    def _cleanup_expired(self) -> int:
        """删除所有过期会话，返回清理数量。调用前需已持有 _lock。"""
        expired_ids = [sid for sid, state in self._sessions.items() if self._is_expired(state)]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)
