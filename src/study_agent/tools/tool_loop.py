"""Tool Calling 循环 —— LLM 决策→工具执行→结果回传→继续决策的完整循环。

这个模块是 agent-core 的"大脑"。它让 LLM 不止能说话，还能做事。

Day 3 vs Day 4 的区别：
  Day 3（结构化输出）：单向提取
    文本 → LLM → JSON → 结束
    工具定义只是"约束输出格式"，不真正执行

  Day 4（Tool Calling 循环）：双向对话
    用户任务 → LLM → "我需要查数据" → 执行工具 → 结果喂回
    → LLM → "数据拿到了，还需要计算" → 执行工具 → 结果喂回
    → LLM → "好了，这是最终答案" → 输出

循环流程（ASCII 图）：

  用户任务
     │
     ▼
  ┌─────────┐    有 tool_call    ┌──────────┐
  │  LLM    │ ─────────────────► │ 执行工具  │
  │ (决策)  │                    │ (行动)   │
  └─────────┘ ◄───────────────── └──────────┘
     │        把结果喂回（循环）      │
     │ 没有 tool_call（最终答案）     │
     ▼
  最终回复

两个协议的关键区别：

  OpenAI 协议（DeepSeek / GLM-4 / Moonshot）：
    - tools 参数：list[{type: "function", function: {name, description, parameters}}]
    - LLM 返回：message.tool_calls[{id, function: {name, arguments}}]
    - 结果回传：messages.append({role: "tool", tool_call_id, content})

  Anthropic 协议（Claude）：
    - tools 参数：list[{name, description, input_schema}]
    - LLM 返回：content 中的 tool_use 块（type="tool_use", id, name, input）
    - 结果回传：messages.append({role: "user", content: [{type: "tool_result", ...}]})

使用方法：
  from study_agent.llm.client import LLMClient
  from study_agent.tools.tool_loop import ToolCallLoop
  from study_agent.tools.builtin_tools import CalculatorTool, DateTimeTool

  client = LLMClient(provider="deepseek")
  loop = ToolCallLoop(client, tools=[CalculatorTool(), DateTimeTool()])

  answer = loop.run("今天是几号？3天后是几号？")
  print(answer)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallLoop:
    """LLM 决策 → 工具执行 → 结果回传 → 继续决策的循环。

    这个类做的事：
      1. 把任务发给 LLM，同时告诉它"你有这些工具可以用"
      2. 如果 LLM 决定调用工具 → 执行工具 → 把结果喂回去
      3. 重复 2-5 轮，直到 LLM 给出最终答案
      4. 超过最大轮次还未完成 → 返回超时提示

    为什么要限制最多 5 轮？
      有些任务 LLM 可能陷入"查了又查"的死循环。
      5 轮是经验值——大部分多步骤任务 2-3 轮就能完成。
    """

    def __init__(self, client: Any, tools: list[Any], max_rounds: int = 5):
        """创建 Tool Calling 循环。

        参数：
          client     → LLMClient 实例
          tools      → BaseTool 子类实例的列表
          max_rounds → 最多循环几轮（默认 5，避免死循环）
        """
        self.client = client
        self.max_rounds = max_rounds

        # 工具名 → 工具实例的映射，方便按名字查找
        self.tool_map: dict[str, Any] = {}
        for tool in tools:
            name = tool.definition.name
            self.tool_map[name] = tool

        logger.info(
            "ToolCallLoop 初始化完成：provider=%s, model=%s, tools=%s, max_rounds=%d",
            client.provider,
            client.model,
            list(self.tool_map.keys()),
            max_rounds,
        )

    # ── 统一入口 ──────────────────────────────────────────

    def run(self, user_message: str, system: str | None = None) -> str:
        """执行 tool calling 循环，返回 LLM 的最终回答。

        如果 LLM 不需要工具，直接返回文本回复（0 轮）。
        如果需要工具，循环直到 LLM 不再请求工具或达到 max_rounds。
        """
        if self.client.sdk_type == "anthropic":
            return self._run_anthropic_loop(user_message, system)
        else:
            return self._run_openai_loop(user_message, system)

    # ═══════════════════════════════════════════════════════
    # OpenAI 协议循环（DeepSeek / GLM-4 / Moonshot）
    # ═══════════════════════════════════════════════════════

    def _run_openai_loop(self, user_message: str, system: str | None) -> str:
        """OpenAI 风格的 tool calling 循环。

        OpenAI 协议的消息流：
          1. [system?, user] → LLM
          2. LLM 返回 assistant msg（可能带 tool_calls）
          3. 如果有 tool_calls → 执行 → 添加 tool msg → 回到步骤 2
          4. 如果没有 tool_calls → assistant 的 content 就是最终答案
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        # 把 BaseTool 列表转成 OpenAI 的 tools 格式
        openai_tools = [t.definition.to_openai_function() for t in self.tool_map.values()]

        for round_num in range(1, self.max_rounds + 1):
            logger.info("OpenAI 循环 Round %d/%d", round_num, self.max_rounds)

            response = self.client._client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                tools=openai_tools,
            )
            msg = response.choices[0].message

            # 没有 tool_calls → LLM 认为任务完成了，直接回复
            if not msg.tool_calls:
                return msg.content or ""

            # 有 tool_calls → 记录本轮调了什么工具
            tool_call_info = [
                f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls
            ]
            logger.info("  LLM 请求调用工具: %s", tool_call_info)

            # 把 assistant 消息（含 tool_calls）加入对话历史
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # 执行每个 tool_call，把结果加回对话
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool = self.tool_map.get(tool_name)

                if tool is None:
                    result = f"错误：未知工具 '{tool_name}'。可用工具：{list(self.tool_map.keys())}"
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = tool.execute(**args)
                    except json.JSONDecodeError as e:
                        result = f"错误：工具参数 JSON 解析失败 —— {e}"
                    except Exception as e:
                        result = f"错误：工具执行失败 —— {type(e).__name__}: {e}"

                logger.info("  工具 %s 执行结果: %s", tool_name, result[:100])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return f"已达到最大轮次 {self.max_rounds}，任务未完成。请简化你的请求。"

    # ═══════════════════════════════════════════════════════
    # Anthropic 协议循环（Claude）
    # ═══════════════════════════════════════════════════════

    def _run_anthropic_loop(self, user_message: str, system: str | None) -> str:
        """Anthropic 风格的 tool calling 循环。

        和 OpenAI 协议的关键区别：
          1. system prompt 是 messages.create() 的独立参数，不在 messages 里
          2. 工具结果以 tool_result 内容块形式放在 user 消息里
          3. response.content 是内容块列表（不是单一 message），需要按 type 分类处理
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        # 把 BaseTool 列表转成 Anthropic 的 tools 格式
        anthropic_tools = [
            {
                "name": t.definition.name,
                "description": t.definition.description,
                "input_schema": _build_anthropic_input_schema(t.definition),
            }
            for t in self.tool_map.values()
        ]

        for round_num in range(1, self.max_rounds + 1):
            logger.info("Anthropic 循环 Round %d/%d", round_num, self.max_rounds)

            kwargs: dict[str, Any] = {
                "model": self.client.model,
                "max_tokens": 1024,
                "messages": messages,
                "tools": anthropic_tools,
            }
            if system:
                kwargs["system"] = system

            response = self.client._client.messages.create(**kwargs)

            # content 是列表，按 type 分类："text" 是文字，"tool_use" 是工具调用
            text_blocks = [b for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # 没有 tool_use → LLM 给出最终答案
            if not tool_use_blocks:
                return "".join(b.text for b in text_blocks) if text_blocks else ""

            tool_names = [
                f"{b.name}({json.dumps(b.input, ensure_ascii=False)})" for b in tool_use_blocks
            ]
            logger.info("  Claude 请求调用工具: %s", tool_names)

            # 把 assistant 消息加入历史
            # Anthropic SDK 的 content block 对象需要转成 dict 才能存进 messages
            messages.append(
                {
                    "role": "assistant",
                    "content": [_block_to_dict(b) for b in response.content],
                }
            )

            # 执行工具，构建 tool_result 列表
            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool = self.tool_map.get(tool_name)

                if tool is None:
                    result_text = f"错误：未知工具 '{tool_name}'"
                else:
                    try:
                        result_text = tool.execute(**block.input)
                    except Exception as e:
                        result_text = f"错误：工具执行失败 —— {type(e).__name__}: {e}"

                logger.info("  工具 %s 执行结果: %s", tool_name, result_text[:100])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            # Anthropic 要求 tool_result 包在 user 消息里
            messages.append({"role": "user", "content": tool_results})

        return f"已达到最大轮次 {self.max_rounds}，任务未完成。请简化你的请求。"


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════


def _build_anthropic_input_schema(definition: Any) -> dict[str, Any]:
    """从 ToolDefinition 构建 Anthropic 的 input_schema。

    Anthropic 的 format 和 OpenAI 的 parameters 本质相同（都是 JSON Schema），
    但 Anthropic 把这个字段叫 input_schema 而不是 parameters。
    """
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param in definition.parameters:
        prop: dict[str, Any] = {
            "type": param.type,
            "description": param.description,
        }
        properties[param.name] = prop
        if param.required:
            required.append(param.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _block_to_dict(block: Any) -> dict[str, Any]:
    """把 Anthropic SDK 的 content block 对象转成普通 dict。

    为什么需要这个？
      Anthropic SDK 返回的 content 是 TypedDict / 对象列表。
      存进 messages 时需要用纯 dict，否则下次发送会序列化失败。

    支持的 block 类型：
      - text → {"type": "text", "text": "..."}
      - tool_use → {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input) if block.input else {},
        }
    else:
        # 未知类型，原样返回（未来兼容）
        logger.warning("未知的 content block 类型: %s", block.type)
        return {"type": block.type}
