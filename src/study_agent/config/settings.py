"""配置中心 —— 所有 provider 信息、默认模型、API Key 管理都在这里。

这个模块是 agent-core 包的"电话本"：
- 想知道 deepseek 的 API 地址？查这里
- 想加一个新的模型厂商？在这里加一行
- 想知道某个 provider 的环境变量叫什么？查这里

设计原则：
  换模型 = 改配置，不动代码。
  新增 provider = 在 PROVIDER_CONFIGS 加一行字典。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
# find_dotenv 会从当前目录向上搜索 .env 文件，找到项目根目录那个
_load_result = load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
if _load_result:
    pass  # .env 加载成功，环境变量已注入 os.environ


# ── Provider 配置的类型定义 ──
# TypedDict 是什么？
#   一种"给字典贴标签"的方式。普通的 dict 里 MyPy 不知道每个 key 对应什么类型，
#   TypedDict 告诉它：这个字典里一定有 base_url、sdk_type、env_key 三个key，
#   它们的类型分别是 str|None、str、str。
class ProviderConfig(TypedDict):
    base_url: str | None
    sdk_type: str
    env_key: str


# ═══════════════════════════════════════════════════════════
# ① Provider 配置 —— 每家厂商的"通讯录"条目
# ═══════════════════════════════════════════════════════════
#
# 字段说明：
#   base_url   → API 的网址，发 HTTP 请求的目的地
#   sdk_type   → "openai"（用 OpenAI SDK 调）还是 "anthropic"（用 Anthropic SDK 调）
#   env_key    → 从哪个环境变量读取 API Key
#
# 为什么 Anthropic 的 base_url 是 None？
#   Anthropic 有自己的官方 SDK，它内置了正确的 API 地址，
#   不需要我们手动指定。而 deepseek/zhipu/moonshot 都兼容 OpenAI 的接口格式，
#   所以用 OpenAI SDK 去调它们时，需要告诉 SDK "去这个地址"。

PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "sdk_type": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "sdk_type": "openai",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "sdk_type": "openai",
        "env_key": "ZHIPU_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "sdk_type": "openai",
        "env_key": "MOONSHOT_API_KEY",
    },
    "anthropic": {
        "base_url": None,
        "sdk_type": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
}

# ═══════════════════════════════════════════════════════════
# ② 默认模型 —— 每家厂商的"默认款"
# ═══════════════════════════════════════════════════════════
#
# 当用户不指定 model 时，自动选这个。
# 为什么不默认选最便宜的？因为 Day 6 的目标是"能用"，
# 后面 Week 14 会做智能模型路由（简单任务用小模型，复杂任务用大模型）。

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
    "zhipu": "glm-4-flash",
    "moonshot": "moonshot-v1-8k",
}


# ═══════════════════════════════════════════════════════════
# ③ 从环境变量加载配置
# ═══════════════════════════════════════════════════════════


def get_api_key(provider: str) -> str | None:
    """读取某个 provider 的 API Key。

    返回 None 表示没找到（而不是抛异常），
    让调用方自己决定"没 Key 时怎么办"。
    """
    cfg = PROVIDER_CONFIGS.get(provider)
    if cfg is None:
        return None
    return os.getenv(cfg["env_key"])


def get_default_provider() -> str:
    """从 LLM_PROVIDER 环境变量读取当前使用的厂商。

    没设置时默认用 deepseek。
    """
    return os.getenv("LLM_PROVIDER", "deepseek")


def list_available_providers() -> list[str]:
    """列出所有已配置 API Key 的 provider。

    用途：前端下拉菜单、自动检测可用模型等。
    """
    available = []
    for name, cfg in PROVIDER_CONFIGS.items():
        if os.getenv(cfg["env_key"]):
            available.append(name)
    return available


# ═══════════════════════════════════════════════════════════
# ④ 配置数据类 —— 把散落的配置字段打包成一个对象
# ═══════════════════════════════════════════════════════════
#
# @dataclass 是什么？
#   一个 Python 装饰器，自动帮你生成 __init__ 方法。
#   不用它的话，你要手写：
#     def __init__(self, provider, model, api_key, ...):
#         self.provider = provider
#         self.model = model
#         ...（8 行样板代码）
#   用了 @dataclass，只需要声明字段，Python 帮你生成以上所有代码。
#
#   类比：你去餐厅点菜，不需要自己进厨房炒菜。
#         @dataclass 就是餐厅的厨师——你点菜（声明字段），它做菜（生成 __init__）。


@dataclass
class LLMConfig:
    """一次 LLM 调用需要的全部配置，打包在一起。

    为什么要打包？
      之前这些参数散落在各处：provider 从环境变量读，
      model 在 DEFAULT_MODELS 里，api_key 在 os.getenv()...
      打包后：一个 LLMConfig 对象就是"一次调用的完整配置单"，
      传给任何函数都只需要一个参数。
    """

    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    sdk_type: str = "openai"
    max_retries: int = 3

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> LLMConfig:
        """从环境变量构建配置 —— 最常用的创建方式。

        用法：
          config = LLMConfig.from_env("deepseek")           # 自动读 Key + 默认模型
          config = LLMConfig.from_env("anthropic", "claude-sonnet-4-6")  # 指定模型
        """
        provider = provider or get_default_provider()
        cfg = PROVIDER_CONFIGS.get(provider)
        if cfg is None:
            raise ValueError(
                f"不支持的 provider: {provider}。" f"可选: {list(PROVIDER_CONFIGS.keys())}"
            )

        api_key = os.getenv(cfg["env_key"])
        if not api_key:
            raise ValueError(
                f"找不到 {provider} 的 API Key！\n"
                f"请在 .env 文件中设置 {cfg['env_key']}=\"your-key\""
            )

        model = model or DEFAULT_MODELS.get(provider, "unknown")

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=cfg["base_url"],
            sdk_type=cfg["sdk_type"],
        )
