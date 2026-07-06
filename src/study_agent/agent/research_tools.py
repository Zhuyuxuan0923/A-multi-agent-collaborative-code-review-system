"""研究 Agent 专用工具集 —— 模拟搜索 + 计算 + 总结能力。

这些工具是 Week 5 Day 2 的一部分，为 ReAct Agent 提供行动能力。
搜索工具使用模拟数据，不需要外部 API Key。
"""

from __future__ import annotations

import math
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# ① 模拟搜索工具
# ═══════════════════════════════════════════════════════════

# 模拟的知识库 —— 当工具"搜索"时，从这个字典里模糊匹配
MOCK_KNOWLEDGE: dict[str, str] = {
    "react 19": (
        "React 19 于 2024 年 12 月发布，主要新特性：1) Server Components 稳定版，"
        "支持在服务器端渲染组件以减少客户端 JS 体积；2) Actions 机制，用 form action "
        "替代 useEffect 处理数据提交；3) Document Metadata 原生支持，无需 react-helmet；"
        "4) 改进的 ref 处理，支持 ref 作为 prop 传递；5) use() hook 允许在 render 中"
        "读取 Promise 和 Context。性能方面，React 19 的客户端 bundle 比 React 18 减小约 20%。"
    ),
    "langchain": (
        "LangChain 是一个用于构建 LLM 应用的开源框架。核心模块："
        "1) Model I/O —— 统一的 LLM 调用接口；2) Retrieval —— 文档加载、分块、向量存储；"
        "3) Agents —— AgentExecutor 循环、工具绑定、ReAct/Plan-Execute 等范式；"
        "4) Chains —— 可组合的调用链。当前稳定版本是 0.3.x，"
        "LangGraph 是 LangChain 生态中专门用于构建有状态、多步骤 Agent 工作流的库。"
    ),
    "fastapi": (
        "FastAPI 是一个现代 Python Web 框架，用于构建 REST API。"
        "核心特性：1) 基于 Python 类型提示的自动参数校验（Pydantic）；"
        "2) 自动生成 OpenAPI/Swagger 文档；3) 异步支持（async/await）；"
        "4) 依赖注入系统。性能与 Node.js 和 Go 接近，适合构建高性能后端。"
    ),
    "ai agent": (
        "AI Agent 是能够自主使用工具、制定计划、执行多步骤任务的 AI 系统。"
        "核心架构组件：1) LLM 大脑 —— 决策和推理；2) 工具层 —— 搜索、计算、API 调用等；"
        "3) 记忆系统 —— 短期（对话历史）+ 长期（向量数据库）；"
        "4) 编排器 —— 循环控制、错误处理、超时。"
        "常见范式包括 ReAct（推理+行动交替）、Plan-Execute（先计划后执行）、"
        "以及多 Agent 协作（CrewAI、AutoGen）。"
    ),
}


def mock_search(query: str) -> str:
    """模拟搜索引擎 —— 在 MOCK_KNOWLEDGE 中做模糊匹配。

    真实场景中这里会调用 SerpAPI / Tavily / Brave Search 等 API。
    """
    query_lower = query.lower()
    results: list[str] = []

    for keyword, content in MOCK_KNOWLEDGE.items():
        # 简单的关键词匹配：查询词中的任意词出现在 key 或 content 中
        query_words = set(query_lower.split())
        keyword_words = set(keyword.split())
        # 检查是否有交集
        if query_words & keyword_words or any(w in content.lower() for w in query_words):
            results.append(f"[{keyword}] {content}")

    if not results:
        # 尝试更模糊的匹配：逐词检查
        for keyword, content in MOCK_KNOWLEDGE.items():
            for word in query_lower.split():
                if len(word) > 2 and word in keyword or word in content.lower():
                    results.append(f"[{keyword}] {content[:200]}...")
                    break

    if results:
        return "\n\n---\n\n".join(results)
    return f"未找到与 '{query}' 相关的结果。请尝试更换搜索关键词。"


# ═══════════════════════════════════════════════════════════
# ② 安全计算器
# ═══════════════════════════════════════════════════════════


def safe_calculate(expression: str) -> str:
    """安全地计算数学表达式。只允许基本算术和数学函数。"""
    import ast

    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else None
                )
                if func_name and func_name not in allowed_names:
                    return f"错误：不允许调用函数 '{func_name}'"
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                return f"错误：不允许使用变量 '{node.id}'"

        code = compile(tree, "<calculator>", "eval")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


# ═══════════════════════════════════════════════════════════
# ③ 文本摘要工具
# ═══════════════════════════════════════════════════════════


def summarize_text(text: str, max_length: int = 100) -> str:
    """简单的文本摘要 —— 提取关键句并截断。

    这不是真正的 AI 摘要，只是用规则提取文本中最"有信息量"的句子。
    真实场景会用 LLM 做摘要。
    """
    sentences = text.replace("；", "。").replace(";", ".").split("。")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return text[:max_length] + ("..." if len(text) > max_length else "")

    # 按句子长度排序（假设长句信息量更大），取前几条
    scored = sorted(sentences, key=len, reverse=True)
    result = ""
    for s in scored:
        if len(result) + len(s) + 1 <= max_length:
            result += s + "。"
        else:
            break

    return result if result else text[:max_length] + "..."


# ═══════════════════════════════════════════════════════════
# ④ 工具注册表 —— 统一管理所有工具
# ═══════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "search": {
        "function": mock_search,
        "description": "搜索互联网获取信息。输入：搜索关键词。输出：相关搜索结果。",
        "param_description": "search(query: str) —— query 是搜索关键词",
    },
    "calculator": {
        "function": safe_calculate,
        "description": "安全计算数学表达式。支持加减乘除、幂运算、sqrt等。",
        "param_description": "calculator(expression: str) —— expression 是数学表达式",
    },
    "summarize": {
        "function": summarize_text,
        "description": "对长文本进行摘要，提取关键信息。",
        "param_description": "summarize(text: str, max_length: int) —— text 是要摘要的文本",
    },
    "current_time": {
        "function": lambda: datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
        "description": "获取当前日期和时间。无需参数。",
        "param_description": "current_time() —— 无需参数",
    },
}
