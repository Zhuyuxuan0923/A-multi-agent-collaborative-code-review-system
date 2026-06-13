"""llm 模块 —— agent-core 的 LLM 调用层。

从这里可以拿到：
- LLMClient              → 统一 LLM 客户端（核心类）
- create_retry_decorator → 创建可配置的重试装饰器
- should_retry           → 判断异常是否值得重试
- StructuredExtractor    → 结构化输出提取器（三种策略）
- ExtractionSchema       → 提取任务的 Schema 定义
"""

from study_agent.llm.client import LLMClient
from study_agent.llm.retry import create_retry_decorator, should_retry
from study_agent.llm.structured import ExtractionSchema, StructuredExtractor

__all__ = [
    "LLMClient",
    "create_retry_decorator",
    "should_retry",
    "StructuredExtractor",
    "ExtractionSchema",
]
