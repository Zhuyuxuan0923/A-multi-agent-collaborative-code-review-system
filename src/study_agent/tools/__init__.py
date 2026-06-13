"""tools 模块 —— agent-core 的工具抽象层。

从这里可以拿到：
- BaseTool       → 所有工具的基类（继承它来创建新工具）
- ToolDefinition → 工具的定义/元数据（告诉 LLM "我能做什么"）
- ToolParameter  → 工具参数的定义
- ToolCallLoop   → LLM 决策 → 工具执行 → 结果回传的循环（Agent 核心）
- CalculatorTool → 安全计算器
- DateTimeTool   → 日期时间工具
- TextStatsTool  → 文本统计工具
"""

from study_agent.tools.base import BaseTool, ToolDefinition, ToolParameter
from study_agent.tools.builtin_tools import CalculatorTool, DateTimeTool, TextStatsTool
from study_agent.tools.tool_loop import ToolCallLoop

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParameter",
    "ToolCallLoop",
    "CalculatorTool",
    "DateTimeTool",
    "TextStatsTool",
]
