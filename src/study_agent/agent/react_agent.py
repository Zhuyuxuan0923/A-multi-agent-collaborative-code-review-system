"""ReAct Agent —— Reasoning + Acting 循环实现。

ReAct 范式：让 LLM 在文本中显式输出"思考过程"（Thought），
然后声明要执行的动作（Action），框架解析文本后执行工具，
将结果（Observation）喂回 LLM，循环直到 LLM 输出 Final Answer。

对比你之前的 ToolCallLoop（Function Calling 风格）：
  FC：LLM 内部思考 -> 输出 tool_calls JSON -> 框架执行 -> 结果回传
  ReAct：LLM 输出 Thought 文本 -> Action 文本 -> 框架解析文本 -> 执行 -> Observation 回传

关键区别：ReAct 的推理过程是"可读的文本"，FC 是"不可见的结构化 JSON"。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from study_agent.agent.research_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# ReAct Prompt 模板 —— 这是整个 ReAct 范式的核心
# ═══════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """你是一个研究助手 Agent，能够使用工具来搜索信息、计算和总结。

你可以使用以下工具：

{tool_descriptions}

你必须严格遵循以下格式来响应：

Question: 用户的问题
Thought: 你需要思考下一步该做什么
Action: 你要调用的工具，格式为 tool_name(param_name="value")
（此时停止，等待系统返回 Observation）
Observation: 系统返回的工具执行结果
... (这个 Thought/Action/Observation 可以重复多次)
Thought: 我现在有足够的信息来回答问题了
Final Answer: 用中文给用户的最终答案

重要规则：
1. 每次只调用一个工具
2. 工具名必须是上面列出的之一
3. Action 行的格式必须是: Action: tool_name(key="value")
4. 你必须自己写 Thought 和 Action，但绝不能自己写 Observation -- Observation 是系统自动注入的
5. 当你认为信息足够时，用 Final Answer 结束
6. 搜索时使用中文关键词能获得更好的结果
"""

# ═══════════════════════════════════════════════════════════
# 方案 A: 简单版解析器 —— 容易出错，让你看到问题
# ═══════════════════════════════════════════════════════════


def parse_action_naive(llm_output: str) -> dict[str, Any] | None:
    """[方案A] 用简单的字符串查找来解析 Action。

    问题在哪？
      - LLM 可能输出 "Action: search(query=React 19)" → 引号被吃了
      - 可能输出 "Action: search("React 19")" → 用括号而不是等号
      - 可能输出 "Action:  search(query='test')" → 多个空格
      - 可能输出 "action: search(...)" → 大小写不一致
    """
    for line in llm_output.split("\n"):
        line_stripped = line.strip()
        # 大小写不敏感匹配
        separator = None
        if "Action:" in line_stripped:
            separator = "Action:"
        elif "action:" in line_stripped:
            separator = "action:"

        if separator:
            # 提取 "Action: " 后面的内容
            action_part = line_stripped.split(separator, 1)[1].strip()
            # 简单按 "(" 分割工具名和参数
            if "(" in action_part:
                tool_name = action_part.split("(", 1)[0].strip()
                args_str = action_part.split("(", 1)[1].rstrip(")")
                return {"tool": tool_name, "args_str": args_str, "raw": line_stripped}
    return None


# ═══════════════════════════════════════════════════════════
# 方案 B: 正则版解析器 —— 更健壮
# ═══════════════════════════════════════════════════════════

# 匹配 Action: tool_name(key="value", ...) 模式
ACTION_PATTERN = re.compile(
    r"Action:\s*"  # Action: 后跟任意空格
    r"(\w+)"  # 工具名 (字母+数字+下划线)
    r"\s*\(\s*"  # 左括号，前后可带空格
    r"(.*?)"  # 参数内容 (非贪婪匹配)
    r"\s*\)\s*$",  # 右括号，前后可带空格，行尾
    re.MULTILINE | re.IGNORECASE,  # 支持 action: 和 ACTION: 等大小写变体
)

# 匹配 key="value" 形式的参数（value 可以是带引号的字符串或裸数字/单词）
KV_PATTERN = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_action_regex(llm_output: str) -> dict[str, Any] | None:
    """[方案B] 用正则表达式解析 Action 行。

    比方案 A 更健壮，但仍有限制：
      - 参数值必须用双引号包裹含空格的字符串
      - 单参数工具如 current_time() 也能匹配
    """
    match = ACTION_PATTERN.search(llm_output)
    if not match:
        return None

    tool_name = match.group(1)
    args_content = match.group(2).strip()

    # 解析 key="value" 参数
    params: dict[str, str] = {}
    if args_content:
        for kv_match in KV_PATTERN.finditer(args_content):
            params[kv_match.group(1)] = kv_match.group(2)

    return {"tool": tool_name, "params": params, "raw": match.group(0)}


# ═══════════════════════════════════════════════════════════
# 方案 C: JSON 格式 —— 最可靠，但依赖 LLM 输出正确 JSON
# ═══════════════════════════════════════════════════════════

JSON_ACTION_PATTERN = re.compile(
    r"```json\s*(.*?)\s*```",
    re.DOTALL,
)


def parse_action_json(llm_output: str) -> dict[str, Any] | None:
    """[方案C] 从 LLM 输出中提取 JSON 格式的 Action。

    比文本解析可靠得多，因为 JSON 有严格语法。
    缺点：不是所有模型都能稳定输出合法 JSON。
    """
    # 尝试提取 ```json ... ``` 代码块
    json_match = JSON_ACTION_PATTERN.search(llm_output)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试直接解析整个输出为 JSON（适用于只输出 JSON 的模型）
    try:
        return json.loads(llm_output.strip())
    except json.JSONDecodeError:
        pass

    return None


# ═══════════════════════════════════════════════════════════
# ReAct 循环 —— 核心引擎
# ═══════════════════════════════════════════════════════════


def build_tool_descriptions() -> str:
    """构建工具描述文本，注入到 prompt 中。"""
    lines: list[str] = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  用法: {info['param_description']}")
    return "\n".join(lines)


def execute_tool(tool_name: str, params: dict[str, str]) -> str:
    """根据工具名和参数执行工具，返回结果字符串。"""
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        return f"错误：未知工具 '{tool_name}'。可用的工具：{available}"

    tool_info = TOOL_REGISTRY[tool_name]
    func = tool_info["function"]

    try:
        if not params:
            result = func()
        else:
            result = func(**params)
        return str(result)
    except TypeError as e:
        return f"参数错误：{e}。正确用法：{tool_info['param_description']}"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"


class ReactAgent:
    """ReAct 范式的 Agent —— 通过 Thought/Action/Observation 循环完成任务。

    和你的 ToolCallLoop 对比：
      - ToolCallLoop 依赖模型的 tool_calls 能力（OpenAI/Anthropic 原生协议）
      - ReactAgent 只需要模型能生成文本（任何模型都能用）
      - ReactAgent 的推理过程完全透明，方便调试和审计
    """

    def __init__(self, client: Any, max_rounds: int = 8, parse_method: str = "regex"):
        """
        参数：
          client       → LLMClient 实例
          max_rounds   → 最大循环轮次（默认 8，ReAct 通常比 FC 需要更多轮次）
          parse_method → Action 解析方式："naive" | "regex" | "json"
        """
        self.client = client

        # ReAct 通常比 Function Calling 需要更多轮次，
        # 因为每轮只有一次 Thought + Action（而 FC 可以一次调用多个工具）
        self.max_rounds = max_rounds

        if parse_method == "naive":
            self.parse_action = parse_action_naive
        elif parse_method == "json":
            self.parse_action = parse_action_json
        else:
            self.parse_action = parse_action_regex

        # 构建 system prompt
        self.system_prompt = REACT_SYSTEM_PROMPT.format(tool_descriptions=build_tool_descriptions())

        # 记录最后一次运行的轮数（供评测框架使用）
        self.last_round_count: int = 0

        logger.info(
            "ReactAgent 初始化: provider=%s, model=%s, max_rounds=%d, parse=%s",
            client.provider,
            client.model,
            max_rounds,
            parse_method,
        )

    def run(self, question: str) -> str:
        """执行 ReAct 循环，返回最终答案。

        循环流程：
          1. 把 Question + 历史 (Thought/Action/Observation) 发给 LLM
          2. LLM 返回 Thought + Action 或 Final Answer
          3. 如果是 Action -> 执行工具 -> 把 Observation 加入对话 -> 回到步骤 1
          4. 如果是 Final Answer -> 返回答案
        """
        # 构建初始 prompt：Question + Thought: 引导 LLM 开始推理
        # 这里 "Thought: " 是关键 —— 它强制 LLM 先思考，而不是直接跳到 Final Answer
        current_prompt = f"Question: {question}\nThought: "

        for round_num in range(1, self.max_rounds + 1):
            logger.info("ReAct Round %d/%d", round_num, self.max_rounds)

            # 把 system prompt + 当前上下文发给 LLM
            full_prompt = self.system_prompt + "\n\n" + current_prompt

            # 调用 LLM —— 注意：这里不传 tools 参数！
            # ReAct 的核心就是"不用原生 tool_calls，用文本格式驱动工具"
            # stop 序列是关键：防止 LLM 在 Action 之后自己编造 Observation
            response = self._call_llm(full_prompt)
            # 补上 "Thought: " 前缀（prompt 以 "Thought: " 结尾，LLM 的输出应该从思考内容开始）
            # 但有些模型自己也会输出 "Thought: "，需要去重
            if not response.strip().startswith("Thought:"):
                response = "Thought: " + response

            logger.info("LLM 原始输出:\n%s", response[:300])

            # 检查是否包含 Final Answer
            final_answer = self._extract_final_answer(response)
            if final_answer:
                self.last_round_count = round_num
                logger.info("ReAct 循环完成, 共 %d 轮", round_num)
                return final_answer

            # 尝试解析 Action
            parsed = self.parse_action(response)
            if parsed is None:
                # 无法解析 —— 可能是 LLM 格式不对，把 LLM 输出作为 observation 喂回去
                logger.warning("无法解析 Action，LLM 输出:\n%s", response[:200])
                current_prompt += response + "\n"
                current_prompt += (
                    'Observation: 格式错误。请使用 Action: tool_name(key="value") 格式。\n'
                )
                # 下一轮还是要引导
                current_prompt += "Thought: "
                continue

            tool_name = parsed.get("tool", "")
            # 统一获取参数（不同解析器返回不同 key）
            params = parsed.get("params") or parsed.get("args_str") or {}
            if isinstance(params, str):
                # naive 解析器返回原始参数字符串，这里简单处理
                params = _parse_naive_args(params)

            # 把 LLM 的完整输出（包括 Thought 和 Action）加入上下文
            # 但需要截断到 Action 行为止（去掉 LLM 可能编造的 Observation）
            # 正则匹配从开头到 Action 行（包含）的部分
            action_until = response
            action_match = re.search(r"(Action:\s*[^\n]+)", response)
            if action_match:
                action_until = response[: action_match.end()]
            current_prompt += action_until + "\n"

            # 执行工具
            observation = execute_tool(tool_name, params)
            logger.info("  工具 %s(%s) -> %s", tool_name, params, observation[:80])

            # 把工具结果喂回上下文 —— 这是 ReAct 循环的关键！
            # Observation 让 LLM "知道"刚才的动作产生了什么结果
            current_prompt += f"Observation: {observation}\n"
            # 引导 LLM 继续思考下一步
            current_prompt += "Thought: "

        return f"经过 {self.max_rounds} 轮思考仍未得出最终答案。请简化问题。"

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM，返回文本回复。

        stop 序列是 ReAct 的关键机制：
          LLM 看到 prompt 以 "Thought: " 结尾，会输出推理过程，
          然后输出 "Action: tool_name(...)"。
          如果在 "Observation:" 处停止，LLM 就不会自己编造工具结果。
        """
        stop_tokens = ["Observation:", "observation:"]
        if self.client.sdk_type == "anthropic":
            response = self.client._client.messages.create(
                model=self.client.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=stop_tokens,
            )
            return "".join(b.text for b in response.content if hasattr(b, "text") and b.text)
        else:
            response = self.client._client.chat.completions.create(
                model=self.client.model,
                messages=[{"role": "user", "content": prompt}],
                stop=stop_tokens,
            )
            return response.choices[0].message.content or ""

    def _extract_final_answer(self, text: str) -> str | None:
        """从 LLM 输出中提取 Final Answer。"""
        # 支持多种写法：Final Answer: / Final answer: / Answer: / 最终答案:
        patterns = [
            r"Final Answer:\s*(.*?)$",
            r"Final answer:\s*(.*?)$",
            r"最终答案：\s*(.*?)$",
            r"Answer:\s*(.*?)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def get_trace(self) -> dict[str, Any]:
        """返回最后一次运行的完整上下文（用于调试对比）。"""
        return {"system_prompt": self.system_prompt}


def _parse_naive_args(args_str: str) -> dict[str, str]:
    """解析 naive 解析器提取的原始参数字符串。"""
    params: dict[str, str] = {}
    if not args_str:
        return params

    # 尝试 key=value 格式
    for match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
        params[match.group(1)] = match.group(2)

    # 如果没匹配到，把整个字符串作为位置参数
    if not params and args_str:
        params = {"query": args_str} if "query" not in params else params

    return params


# ═══════════════════════════════════════════════════════════
# LangChain 版本 —— 用 create_react_agent
# ═══════════════════════════════════════════════════════════


def create_langchain_react_agent(client: Any):
    """用 LangChain 的 create_react_agent 创建 ReAct Agent。

    这个函数演示了 LangChain 版 ReAct Agent 的创建过程。
    和上面手写版本的对比：
      - LangChain 版：框架处理循环、解析、错误
      - 手写版：一切透明，每行都是你控制的
    """
    from langchain.agents import create_react_agent
    from langchain.agents.agent import AgentExecutor
    from langchain_core.tools import BaseTool as LCBaseTool

    # 把我们的工具转成 LangChain 的 Tool 格式
    class ResearchSearchTool(LCBaseTool):
        name: str = "search"
        description: str = "搜索互联网获取信息。输入搜索关键词，返回相关结果。"

        def _run(self, query: str) -> str:
            from study_agent.agent.research_tools import mock_search

            return mock_search(query)

    class ResearchCalculatorTool(LCBaseTool):
        name: str = "calculator"
        description: str = "计算数学表达式。输入如 '(3+5)*2'。"

        def _run(self, expression: str) -> str:
            from study_agent.agent.research_tools import safe_calculate

            return safe_calculate(expression)

    tools = [ResearchSearchTool(), ResearchCalculatorTool()]

    # 获取（或手写）ReAct prompt 模板
    # LangChain hub 上有预制模板，也可以自己写
    from langchain_core.prompts import PromptTemplate

    react_template = """回答以下问题，尽可能使用工具。

你可以使用以下工具：

{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 你应该思考要做什么
Action: 要采取的行动，格式为 [{tool_names}]
Action Input: 行动的输入
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(react_template)

    # 包装 LLMClient 成 LangChain 兼容的 ChatModel
    lc_llm = _wrap_client_for_langchain(client)

    agent = create_react_agent(lc_llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )

    return executor


def _wrap_client_for_langchain(client: Any):
    """把 LLMClient 包装成 LangChain 兼容的 ChatModel。"""

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _LCWrapper(BaseChatModel):
        """把我们的 LLMClient 包一层，适配 LangChain 的 BaseChatModel 接口。"""

        _client: Any

        def __init__(self, lc_client: Any):
            super().__init__()
            self._client = lc_client

        def _generate(self, messages: list, stop=None, run_manager=None, **kwargs):
            api_messages = []
            for msg in messages:
                role = "assistant" if msg.type == "ai" else msg.type
                api_messages.append({"role": role, "content": msg.content})

            if self._client.sdk_type == "anthropic":
                resp = self._client._client.messages.create(
                    model=self._client.model,
                    max_tokens=1024,
                    messages=api_messages,
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.text)
            else:
                resp = self._client._client.chat.completions.create(
                    model=self._client.model,
                    messages=api_messages,
                )
                text = resp.choices[0].message.content or ""

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        @property
        def _llm_type(self) -> str:
            return "study-agent-llm-client"

    return _LCWrapper(client)
