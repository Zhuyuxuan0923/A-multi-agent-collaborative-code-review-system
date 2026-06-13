"""结构化输出提取器 —— 三种方式让 LLM 输出稳定的 JSON 数据

这个模块解决什么问题？
  LLM 天然输出"自由文本"，但应用程序需要"结构化数据"：
  - 存数据库需要固定字段
  - 前端图表需要 JSON 格式
  - 两个系统之间传数据需要严格 Schema

  直接说"请输出 JSON"就像让小孩"写字整齐点"——有时整齐，有时不。

三种解决方案的原理对比：

  ┌──────────────┬─────────────────────────┬──────────────────────┐
  │ 方案          │ 原理                    │ 适用 provider         │
  ├──────────────┼─────────────────────────┼──────────────────────┤
  │ JSON Mode    │ API 层保证输出合法 JSON  │ OpenAI, DeepSeek     │
  │ Tool Use     │ 把 Schema 伪装成工具参数 │ Anthropic Claude     │
  │ Function Call│ 让 LLM "调用"一个函数，  │ GLM-4, DeepSeek 等   │
  │              │ 参数就是结构化数据       │ 所有 OpenAI 兼容厂商 │
  └──────────────┴─────────────────────────┴──────────────────────┘

  共同思想：不给 LLM "可以自由发挥"的空间。用 API 参数锁死输出格式。

使用方法：
  from study_agent.llm.client import LLMClient
  from study_agent.llm.structured import StructuredExtractor, ExtractionSchema

  client = LLMClient(provider="deepseek")
  extractor = StructuredExtractor(client)

  schema = ExtractionSchema(
      name="product_info",
      description="从文本中提取产品信息",
      properties={
          "product_name": {"type": "string", "description": "产品名称"},
          "features": {"type": "array", "items": {"type": "string"}, "description": "功能列表"},
      },
      required=["product_name", "features"],
  )

  result = extractor.extract(text, schema, method="json_mode")
  # → {"product_name": "小记灵", "features": ["语音转文字", "智能摘要"]}
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ① ExtractionSchema —— 描述"要提取什么"
# ═══════════════════════════════════════════════════════════════


@dataclass
class ExtractionSchema:
    """一次结构化提取任务的定义。

    类比：你要填一张表，ExtractionSchema 就是"表头"——
      - name: 这张表叫什么
      - description: 什么时候填这张表
      - properties: 每一列的名字、类型、说明
      - required: 哪些列必须填，不能空着

    字段说明：
      name        → Schema 名称，生成 JSON Schema 时用作 title
      description → 告诉 LLM 这个提取任务的目的
      properties  → 字段定义，key 是字段名，value 是 {"type": ..., "description": ...}
      required    → 哪些字段是必填的（LLM 输出如果缺了这些字段就是失败）
    """

    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: list[str] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """转为标准 JSON Schema（OpenAI / GLM-4 都能理解）。"""
        return {
            "type": "object",
            "properties": self.properties,
            "required": self.required,
            "additionalProperties": False,
        }

    def to_system_prompt_fragment(self) -> str:
        """生成一段可以拼进 system prompt 的 Schema 描述。

        为什么需要这个？
          JSON Mode 的 response_format 只保证"输出是合法 JSON"，
          不保证"JSON 里有你想要的字段"。
          所以需要把 Schema 同时写进 system prompt，双重保险。
        """
        lines = [f"你需要从文本中提取以下信息（{self.description}）："]
        for field_name, field_info in self.properties.items():
            req_mark = "（必填）" if field_name in self.required else "（选填）"
            lines.append(f"  - {field_name}: {field_info.get('description', '')} {req_mark}")
            lines.append(f"    类型: {field_info.get('type', 'string')}")
            if field_info.get("items"):
                lines.append(f"    元素类型: {field_info['items'].get('type', 'string')}")

        lines.append("\n请严格按照上述字段输出 JSON，不要添加额外字段。")
        return "\n".join(lines)

    def to_openai_tool(self) -> dict[str, Any]:
        """转为 OpenAI Function Calling 的工具定义。

        OpenAI 要求的格式：
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """转为 Anthropic Tool Use 的工具定义。

        Anthropic 的格式与 OpenAI 不同：
          {"name": ..., "description": ..., "input_schema": {...}}
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema(),
        }


# ═══════════════════════════════════════════════════════════════
# ② StructuredExtractor —— 三种策略的统一入口
# ═══════════════════════════════════════════════════════════════


class StructuredExtractor:
    """用不同策略让 LLM 输出结构化数据。

    用法：
      from study_agent.llm.client import LLMClient
      client = LLMClient(provider="deepseek")
      extractor = StructuredExtractor(client)
      result = extractor.extract(text, schema, method="json_mode")

    method 可选值：
      - "prompt_only"    → 基线：仅靠 prompt 说"请输出 JSON"（不推荐，成功率低）
      - "json_mode"      → OpenAI JSON Mode（OpenAI / DeepSeek 支持）
      - "tool_use"       → Anthropic Tool Use（Claude 专用）
      - "function_call"  → OpenAI Function Calling（GLM-4 / DeepSeek 等兼容厂商）
    """

    def __init__(self, client: Any) -> None:
        """client 是 LLMClient 实例。"""
        self.client = client

    # ── 统一入口 ──────────────────────────────────────────

    def extract(
        self,
        text: str,
        schema: ExtractionSchema,
        method: str = "json_mode",
    ) -> dict[str, Any] | None:
        """用指定方法从文本中提取结构化数据。

        返回 None 表示提取失败（JSON 不合法 / 缺必填字段 / API 报错）。
        返回 dict 表示成功。
        """
        if method == "prompt_only":
            return self._extract_via_prompt_only(text, schema)
        elif method == "json_mode":
            return self._extract_via_json_mode(text, schema)
        elif method == "tool_use":
            return self._extract_via_tool_use(text, schema)
        elif method == "function_call":
            return self._extract_via_function_call(text, schema)
        else:
            raise ValueError(
                f"不支持的方法: {method}，可选: prompt_only, json_mode, tool_use, function_call"
            )

    # ── 方式 0：基线 —— 仅靠 prompt ───────────────────────

    def _extract_via_prompt_only(
        self, text: str, schema: ExtractionSchema
    ) -> dict[str, Any] | None:
        """基线方案：只靠 prompt 说"请输出 JSON"。

        这是最常见的"偷懒"写法，也是问题最多的写法：
          ❌ LLM 可能在 JSON 外包 ```json``` 标记
          ❌ LLM 可能在 JSON 后面继续聊天"好的，我已经为你提取了..."
          ❌ LLM 可能漏掉某些字段
          ❌ LLM 可能自己发明新字段

        这个方案的存在价值就是让你看到"不用工具有多不靠谱"。
        """
        system_prompt = (
            f"{schema.to_system_prompt_fragment()}\n\n"
            "重要：只输出 JSON，不要包含任何其他文字，不要用 ``` 包裹。"
        )

        try:
            raw = self.client.chat(text, system=system_prompt)
            return _parse_json_response(raw)
        except Exception as e:
            logger.warning(f"prompt_only 提取失败: {e}")
            return None

    # ── 方式 1：OpenAI JSON Mode ──────────────────────────

    def _extract_via_json_mode(self, text: str, schema: ExtractionSchema) -> dict[str, Any] | None:
        """OpenAI JSON Mode —— API 层保证输出是合法 JSON。

        原理：
          OpenAI 的 chat completions API 有一个 response_format 参数。
          设为 {"type": "json_object"} 后，模型在生成每个 token 时只考虑
          能形成合法 JSON 的 token，所以输出一定是合法的 JSON。

          但这只保证"是 JSON"，不保证"包含你想要的字段"。
          所以还需要在 system prompt 里描述 Schema（双重保险）。

        适用：OpenAI (gpt-4o, gpt-4o-mini)、DeepSeek
        不适用：Anthropic Claude（没有这个参数）
        """
        system_prompt = (
            f"{schema.to_system_prompt_fragment()}\n\n"
            "你必须输出一个 JSON 对象，不要包含任何其他文字。"
        )

        # 构建 messages
        messages: list[dict[str, str]] = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        try:
            if self.client.sdk_type == "anthropic":
                # Claude 不支持 JSON Mode，回退到 Tool Use
                logger.info("Anthropic 不支持 JSON Mode，自动切换到 tool_use")
                return self._extract_via_tool_use(text, schema)

            response = self.client._client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return _parse_json_response(raw)
        except Exception as e:
            logger.warning(f"json_mode 提取失败 ({self.client.provider}): {e}")
            return None

    # ── 方式 2：Anthropic Tool Use ────────────────────────

    def _extract_via_tool_use(self, text: str, schema: ExtractionSchema) -> dict[str, Any] | None:
        """Claude Tool Use —— 把输出 Schema 伪装成"工具的输入参数"。

        原理：
          Claude 没有 JSON Mode，但它有 Tool Use。
          Tool Use 原本的用途是让 Claude 调用外部工具，
          但我们可以"骗"Claude：定义一个工具，这个工具的
          输入参数 Schema 恰好是我们想要的输出格式。
          然后用 tool_choice 强制 Claude 必须调用这个工具，
          Claude 就会把结构化数据填进 tool_use.input。

          这有点"曲线救国"——工具参数是强 Schema 的，
          所以数据一定是结构化的。

        适用：Anthropic Claude
        """
        if self.client.sdk_type != "anthropic":
            logger.warning(f"{self.client.provider} 不是 Anthropic，tool_use 可能不可用")
            return None

        system_prompt = (
            "你的任务是从用户提供的文本中提取结构化信息。"
            "请调用 extract 工具，将提取结果填入参数。"
        )

        try:
            response = self.client._client.messages.create(
                model=self.client.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
                tools=[schema.to_anthropic_tool()],
                tool_choice={"type": "tool", "name": schema.name},
            )

            # 从 response 中提取 tool_use 的 input
            for block in response.content:
                if block.type == "tool_use":
                    return dict(block.input) if block.input else None

            # 如果 Claude 没调用工具（理论上前面的 tool_choice 能防止这个）
            logger.warning("Claude 未返回 tool_use，可能未强制调用工具")
            return None
        except Exception as e:
            logger.warning(f"tool_use 提取失败 ({self.client.provider}): {e}")
            return None

    # ── 方式 3：OpenAI Function Call ──────────────────────

    def _extract_via_function_call(
        self, text: str, schema: ExtractionSchema
    ) -> dict[str, Any] | None:
        """OpenAI Function Calling —— 让 LLM "调用"一个函数，参数即结构化数据。

        原理：
          和 Tool Use 一样的思路——定义一个函数，它的参数 Schema 就是
          我们要的输出格式。用 tool_choice 强制调用。
          LLM 以为自己在调用工具，实际上我们根本不执行这个函数，
          直接拿 tool_calls[0].function.arguments 里的 JSON。

        Function Call vs JSON Mode：
          - JSON Mode 是"我要输出 JSON"
          - Function Call 是"我要调用函数，参数是 JSON"
          结果都是 JSON，但 Function Call 因为工具定义的 Schema 是
          强约束的，所以字段更不容易漏。

        适用：GLM-4、DeepSeek、Moonshot 等所有 OpenAI 兼容厂商
        """
        if self.client.sdk_type != "openai":
            logger.warning(
                f"{self.client.provider} 的 sdk_type 不是 openai，无法使用 function_call"
            )
            return None

        system_prompt = (
            f"你的任务是从用户提供的文本中提取结构化信息。"
            f"请调用 {schema.name} 函数，将提取结果作为参数传入。"
        )

        messages: list[dict[str, str]] = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        try:
            response = self.client._client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                tools=[schema.to_openai_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": schema.name},
                },
            )

            # 提取 tool_calls 中的 JSON 参数
            choice = response.choices[0]
            if choice.message.tool_calls:
                raw_args = choice.message.tool_calls[0].function.arguments
                result: dict[str, Any] = json.loads(raw_args)
                return result
            else:
                # 某些模型可能忽略 tool_choice，直接文本回复
                logger.warning("模型未返回 tool_call，尝试从文本解析")
                raw = choice.message.content or ""
                return _parse_json_response(raw)
        except Exception as e:
            logger.warning(f"function_call 提取失败 ({self.client.provider}): {e}")
            return None

    # ── 工具方法 ──────────────────────────────────────────

    def available_methods(self) -> list[str]:
        """根据当前 client 的 provider 返回可用的提取方法。

        不同 provider 支持不同的方法：
          - openai    → prompt_only, json_mode, function_call
          - anthropic → prompt_only, tool_use
          - deepseek  → prompt_only, json_mode, function_call
          - zhipu     → prompt_only, json_mode, function_call
        """
        methods = ["prompt_only"]

        if self.client.sdk_type == "anthropic":
            methods.append("tool_use")
        else:
            methods.append("json_mode")
            methods.append("function_call")

        return methods


# ═══════════════════════════════════════════════════════════════
# ③ 辅助函数 —— 从 LLM 的原始回复里"抢救"JSON
# ═══════════════════════════════════════════════════════════════


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """从 LLM 的原始文本回复中提取 JSON 对象。

    这个函数解决的问题：
      LLM 输出的是"文本"，不一定直接是 JSON。
      可能被 ```json ... ``` 包裹，可能前面有废话，
      可能后面有"希望以上信息对你有帮助"。

    处理策略：
      1. 先尝试直接解析整个字符串
      2. 失败则尝试匹配 ```json ... ``` 代码块
      3. 失败则尝试匹配 ``` ... ``` 代码块
      4. 失败则尝试找到第一个 { 和最后一个 }
      5. 全失败返回 None
    """
    if not raw:
        return None

    raw = raw.strip()

    # 策略 1：直接解析
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 策略 2：提取 ```json ... ``` 代码块
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # 策略 3：提取 ``` ... ``` 代码块
    match = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # 策略 4：找第一个 { 和最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    return None
