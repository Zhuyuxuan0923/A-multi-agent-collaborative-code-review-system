"""LLM API 调用的重试机制 —— 基于 tenacity 库。

这个模块只做一件事：给 API 调用加上"自动重试"能力。

核心概念回顾（Day 5 学的）：
  1. 只有"临时性"错误才重试（网络断了、被限流）——永久性错误重试没用（Key 错了）
  2. 指数退避 = 每次重试等更久（1s → 2s → 4s → 8s...），给服务器喘息时间
  3. Rate Limit (HTTP 429) 要尊重服务器给的 Retry-After 时间

用法：
  from study_agent.llm.retry import create_retry_decorator

  retry_decorator = create_retry_decorator(max_retries=3)

  @retry_decorator
  def my_api_call():
      return client.chat.completions.create(...)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    BadRequestError as AnthropicBadRequestError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError as OpenaiAPIConnectionError,
)
from openai import (
    APIStatusError as OpenaiAPIStatusError,
)
from openai import (
    AuthenticationError as OpenaiAuthenticationError,
)
from openai import (
    BadRequestError as OpenaiBadRequestError,
)
from openai import (
    RateLimitError as OpenaiRateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def should_retry(exc: BaseException) -> bool:
    """判断一个异常是否值得重试。

    核心逻辑：
      ❌ 认证错误 (401) → Key 错了，重试 100 次也没用
      ❌ 请求错误 (400) → 参数格式不对，重试还是不对
      ✅ 网络错误       → 网络可能恢复
      ✅ 限流 (429)     → 等一会儿配额恢复
      ✅ 服务器错误     → 服务器可能恢复

    未知异常保守处理：不重试（避免把未知问题搞得更糟）。
    """
    # 永久性错误 —— 重试治不好
    if isinstance(exc, (OpenaiAuthenticationError, AnthropicAuthenticationError)):
        return False
    if isinstance(exc, (OpenaiBadRequestError, AnthropicBadRequestError)):
        return False

    # 临时性错误 —— 重试可能治好
    if isinstance(
        exc,
        (
            OpenaiAPIConnectionError,
            AnthropicAPIConnectionError,
            OpenaiRateLimitError,
            AnthropicRateLimitError,
            OpenaiAPIStatusError,
            AnthropicAPIStatusError,
        ),
    ):
        return True

    # 不认识的异常 —— 保守不重试
    return False


def create_retry_decorator(max_retries: int = 3) -> Callable[..., Callable[..., Any]]:
    """创建一个可配置的重试装饰器。

    参数：
      max_retries → 最多重试几次（默认 3）

    返回值是一个 @retry 装饰器，可以直接用在任何函数上。

    为什么把装饰器"包在函数里"？
      不同场景需要不同的重试次数：
      - 聊天场景：3 次够了，用户不想等太久
      - 文档处理：可以多试几次（5-10 次），因为耗时操作失败代价高
      - 后台任务：甚至可以设 10 次，因为没人在盯着看

    wait_exponential(multiplier=1, min=1, max=30) 的工作方式：
      第1次重试等 1*2^0 = 1 秒
      第2次重试等 1*2^1 = 2 秒
      第3次重试等 1*2^2 = 4 秒
      ...最长不超过 30 秒
    """
    return retry(  # type: ignore[no-any-return]
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception(should_retry),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
