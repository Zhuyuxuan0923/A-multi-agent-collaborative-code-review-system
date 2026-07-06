"""Memory 抽象基类。

所有 Memory 实现都继承这个类，保证外部调用者不需要关心内部是哪种 Memory。
这叫做"面向接口编程"——调用方只依赖 add() / get_context() / get_stats()，
具体实现可以随时替换。

类比：遥控器上的"播放"按钮。
  不管你是 DVD、蓝光、还是流媒体，按播放就是播放。
  BaseMemory 就是这个按钮的规格说明书。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """Memory 抽象基类 —— 定义三种方法，子类必须实现。"""

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """存入一轮对话。

        role 取值:
          - "user"      : 用户说的话
          - "assistant" : AI 的回复
        """
        ...

    @abstractmethod
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """取出适合喂给 LLM 的上下文。

        query      — 用户当前的提问（Buffer 不需要，Vector 用于检索）
        max_tokens — 返回文本的 token 上限，超出部分会被裁剪

        返回一段格式化后的字符串，可以直接拼到 LLM 的 messages 里。
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """返回当前状态的统计信息。

        返回的字典至少包含:
          - type         : Memory 类型名称
          - rounds       : 已存储的对话轮数
          - total_chars  : 已存储的总字符数
          - total_tokens : 已存储的总 token 数（tiktoken 精确计数）
        """
        ...
