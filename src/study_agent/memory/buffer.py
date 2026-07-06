"""BufferMemory —— 全量记忆。

工作原理：
  像一个不断变长的纸条，每次对话都往后面追加。
  取上下文时从最近的消息开始往前拿，超出 token 上限就丢弃最早的内容。

优点：不会丢失任何信息（只要放得下），实现简单
缺点：对话长了之后 token 消耗线性增长，100 轮对话可能吃掉上万 token
适用：短对话（< 20 轮）、原型验证
"""

from __future__ import annotations

from typing import Any

import tiktoken

from study_agent.memory.base import BaseMemory


class BufferMemory(BaseMemory):
    """全量对话记忆 —— 记住一切，简单粗暴。"""

    def __init__(self) -> None:
        # 每条消息存为 (role, content) 元组
        self._messages: list[tuple[str, str]] = []
        # tiktoken 编码器 —— o200k_base 是 gpt-4o / gpt-4o-mini 使用的分词器
        self._encoder = tiktoken.get_encoding("o200k_base")

    # ── 公开接口 ─────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """追加一轮对话到末尾。"""
        self._messages.append((role, content))

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """从最新一轮往前取，取出不超过 max_tokens 的对话历史。

        query 参数在这里不用——Buffer 不关心查询内容，只管全量返回。
        保留这个参数是为了接口统一。
        """
        lines: list[str] = []
        tokens_used = 0

        # 从最新到最旧遍历（reversed），保证最新对话优先保留
        for role, content in reversed(self._messages):
            line = f"{role}: {content}"
            line_tokens = len(self._encoder.encode(line))
            if tokens_used + line_tokens > max_tokens:
                break  # 再加就超了，停
            lines.append(line)
            tokens_used += line_tokens

        # lines 现在是 [最新, ..., 较旧]，翻转回来变成时间顺序
        lines.reverse()
        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """统计当前状态。"""
        full_text = "\n".join(f"{r}: {c}" for r, c in self._messages)
        return {
            "type": "BufferMemory",
            "rounds": len(self._messages),
            "total_chars": len(full_text),
            "total_tokens": len(self._encoder.encode(full_text)),
        }
