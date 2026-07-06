"""SummaryMemory —— 摘要记忆。

工作原理：
  保留最近 N 轮对话的原文，对更早的对话调用 LLM 生成一段摘要。
  上下文 = "[摘要] ..." + "[最近对话] ..."

  类比：你做课堂笔记。
  - 最近 5 分钟的内容你记得每一句话（Buffer 部分）
  - 更早的内容你只记了要点（Summary 部分）

核心参数：
  recent_rounds: 保留多少轮原文（默认 5），超过这个数的对话会被合并到摘要里

优点：token 消耗稳定，长对话友好
缺点：摘要会丢失细节，依赖 LLM 摘要质量
适用：长对话（> 20 轮）、客服机器人、需要持久化记忆的场景
"""

from __future__ import annotations

from typing import Any

import tiktoken

from study_agent.llm.client import LLMClient
from study_agent.memory.base import BaseMemory


class SummaryMemory(BaseMemory):
    """保留最近原文 + LLM 压缩早期对话为摘要。"""

    def __init__(self, llm: LLMClient, recent_rounds: int = 5) -> None:
        """创建一个 SummaryMemory。

        llm           — 用来生成摘要的 LLM 客户端
        recent_rounds — 保留多少轮原文不压缩（默认 5 轮）
        """
        self._llm = llm
        self._recent_rounds = recent_rounds
        self._messages: list[tuple[str, str]] = []
        self._summary: str = ""  # 早期对话的摘要
        self._encoder = tiktoken.get_encoding("o200k_base")

    # ── 公开接口 ─────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """追加一轮对话，超过 recent_rounds 时触发摘要更新。"""
        self._messages.append((role, content))

        # 当消息数超过 recent_rounds 的 2 倍时，触发一次摘要合并
        # 不是每轮都摘要——那样 API 调用太频繁
        overflow = len(self._messages) - self._recent_rounds
        if overflow >= self._recent_rounds:
            self._update_summary()

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """拼出 "摘要 + 最近对话" 的上下文。

        query 参数这里不用，保留为了接口统一。
        """
        parts: list[str] = []

        # 1. 摘要部分
        if self._summary:
            parts.append(f"[对话历史摘要]\n{self._summary}")

        # 2. 最近原文部分
        recent = self._messages[-self._recent_rounds :]
        recent_text = "\n".join(f"{r}: {c}" for r, c in recent)
        parts.append(f"[最近对话]\n{recent_text}")

        full = "\n\n".join(parts)

        # 如果总长度超限，裁剪（优先保留最近的原文）
        return self._truncate(full, max_tokens)

    def get_stats(self) -> dict[str, Any]:
        """统计当前状态。"""
        full_text = "\n".join(f"{r}: {c}" for r, c in self._messages)
        return {
            "type": "SummaryMemory",
            "rounds": len(self._messages),
            "total_chars": len(full_text),
            "total_tokens": len(self._encoder.encode(full_text)),
            "summary_chars": len(self._summary),
            "recent_rounds": self._recent_rounds,
        }

    # ── 内部方法 ─────────────────────────────────────────

    def _update_summary(self) -> None:
        """把超过 recent_rounds 的消息合并到摘要中。"""
        # 需要被摘要的消息 = 除了最近 recent_rounds 条之外的全部
        to_summarize = self._messages[: -self._recent_rounds]
        if not to_summarize:
            return

        conversation = "\n".join(f"{r}: {c}" for r, c in to_summarize)

        # 如果已有摘要，做增量更新；否则从零生成
        if self._summary:
            prompt = (
                "下面是之前对话的摘要，以及新增的几轮对话。"
                "请把摘要更新一下，合并新对话的信息。只输出更新后的摘要，不要加额外说明。\n\n"
                f"[现有摘要]\n{self._summary}\n\n"
                f"[新增对话]\n{conversation}"
            )
        else:
            prompt = (
                "请为以下对话生成一段简洁的摘要（200 字以内），"
                "保留关键事实、用户偏好和重要决策。只输出摘要，不要加额外说明。\n\n"
                f"{conversation}"
            )

        try:
            self._summary = self._llm.chat(prompt)
        except Exception:
            # LLM 调用失败时不更新摘要，静默处理
            # 下一次 add 时会重试
            pass

        # 把被摘要掉的消息删掉，只保留 recent_rounds 条
        self._messages = self._messages[-self._recent_rounds :]

    def _truncate(self, text: str, max_tokens: int) -> str:
        """按 token 数裁剪文本，从头部裁（保留末尾——即最近的内容）。"""
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # 保留最后 max_tokens 个 token
        return self._encoder.decode(tokens[-max_tokens:])
