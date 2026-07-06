"""Memory 包 —— Agent 记忆机制实现。

Buffer Memory  : 全量记忆，简单但费 token
Summary Memory : 摘要记忆，省 token 但丢细节
Vector Memory  : 向量检索记忆，按相关性取历史
Hybrid Memory  : 三层混合 —— Buffer（近期）+ Summary（远期）+ Vector（检索）

用法示例：
  from study_agent.memory import BufferMemory, HybridMemory

  mem = BufferMemory()
  mem.add("user", "Python 的 GIL 是什么？")
  mem.add("assistant", "GIL 是全局解释器锁...")
  context = mem.get_context("GIL 影响多线程吗？", max_tokens=1000)
"""

from study_agent.memory.base import BaseMemory
from study_agent.memory.buffer import BufferMemory
from study_agent.memory.comparator import MemoryComparator
from study_agent.memory.hybrid import HybridMemory
from study_agent.memory.summary import SummaryMemory
from study_agent.memory.vector import VectorMemory

__all__ = [
    "BaseMemory",
    "BufferMemory",
    "SummaryMemory",
    "VectorMemory",
    "HybridMemory",
    "MemoryComparator",
]
