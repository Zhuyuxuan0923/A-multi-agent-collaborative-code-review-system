"""内置示例工具 —— 供 Tool Calling 循环使用的具体工具实现。

这里的工具都是"模拟"的——不需要外部 API Key，不联网。
目的是演示 Tool Calling 循环的工作流程，而不是实现真正的生产级工具。

工具清单：
  CalculatorTool   → 安全计算器（加减乘除、幂运算、sqrt 等）
  DateTimeTool     → 获取当前日期时间、计算日期差、格式化
  TextStatsTool    → 文本统计（字数、词数、字符数、反转等）

每个工具都是 BaseTool 的子类，所以你可以复用 ToolCallLoop。
添加新工具只需要：
  1. 继承 BaseTool
  2. 实现 definition 属性
  3. 实现 execute() 方法
  4. 把实例传给 ToolCallLoop(tools=[...])
"""

from __future__ import annotations

import ast
import math
from datetime import datetime, timedelta

from study_agent.tools.base import BaseTool, ToolDefinition, ToolParameter

# ═══════════════════════════════════════════════════════════
# ① 安全计算器
# ═══════════════════════════════════════════════════════════


class CalculatorTool(BaseTool):
    """安全地计算数学表达式。

    为什么叫"安全"计算器？
      不能用 Python 的 eval()——如果 LLM 传出 __import__('os').system('rm -rf /')
      你的电脑就没了。这里用 ast.literal_eval 限制只能做数学运算。

    支持：+、-、*、/、**（幂）、//（整除）、%（取余）、sqrt、abs
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description=(
                "安全地计算数学表达式。支持加减乘除、幂运算、括号。"
                "例如：calculator(expression='(3+5)*2') 返回 16。"
                "当你需要进行数学计算时使用。"
            ),
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="数学表达式，如 '(3+5)*2'、'sqrt(144)'、'2**10'",
                    required=True,
                ),
            ],
        )

    def execute(self, expression: str) -> str:  # type: ignore[override]
        # 允许的函数白名单——除了这些，其他任何函数调用都不允许
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
            # Step 1：把字符串解析成 AST（抽象语法树）
            tree = ast.parse(expression, mode="eval")

            # Step 2：遍历 AST，检查每个节点是否安全
            for node in ast.walk(tree):
                # 禁止属性访问（a.b）、调用（f()）除了白名单以外的函数
                if isinstance(node, ast.Call):
                    func_name = _get_func_name(node.func)
                    if func_name and func_name not in allowed_names:
                        return f"错误：不允许调用函数 '{func_name}'"
                # 禁止使用变量名（除了白名单）
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    return f"错误：不允许使用变量 '{node.id}'"

            # Step 3：编译并执行
            code = compile(tree, "<calculator>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)

            if isinstance(result, float):
                # 浮点数保留 6 位小数，去掉末尾的 0
                result = round(result, 6)
                if result == int(result):
                    result = int(result)

            return f"计算结果：{result}"
        except SyntaxError as e:
            return f"错误：表达式语法不对 —— {e}"
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"错误：{type(e).__name__}: {e}"


def _get_func_name(node: ast.expr) -> str | None:
    """从 AST 节点提取函数名。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# ═══════════════════════════════════════════════════════════
# ② 日期时间工具
# ═══════════════════════════════════════════════════════════


class DateTimeTool(BaseTool):
    """获取当前时间，或计算日期偏移（如"3天后""一周前"）。"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="datetime",
            description=(
                "获取当前日期和时间，或计算相对日期。"
                "如果不传任何参数，返回当前日期时间。"
                "可以计算「3天后」「一周前」等相对日期。"
                "例如：datetime() 返回当前时间；datetime(offset_days=3) 返回3天后。"
            ),
            parameters=[
                ToolParameter(
                    name="offset_days",
                    type="number",
                    description="日期偏移天数。正数=未来，负数=过去，0=今天。默认为0。",
                    required=False,
                    default="0",
                ),
                ToolParameter(
                    name="format_type",
                    type="string",
                    description=(
                        "输出格式：'full'=完整日期时间，"
                        "'date'=仅日期，'weekday'=日期+星期几。默认 'full'。"
                    ),
                    required=False,
                    default="full",
                ),
            ],
        )

    def execute(self, offset_days: int = 0, format_type: str = "full") -> str:  # type: ignore[override]
        target = datetime.now() + timedelta(days=int(offset_days))

        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        if format_type == "date":
            return target.strftime("%Y年%m月%d日")
        elif format_type == "weekday":
            return f"{target.strftime('%Y年%m月%d日')} {weekday_names[target.weekday()]}"
        else:  # full
            return (
                f"{target.strftime('%Y年%m月%d日')} "
                f"{weekday_names[target.weekday()]} "
                f"{target.strftime('%H:%M:%S')}"
            )


# ═══════════════════════════════════════════════════════════
# ③ 文本统计工具
# ═══════════════════════════════════════════════════════════


class TextStatsTool(BaseTool):
    """分析文本的统计信息或做文本变换。"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="text_stats",
            description=(
                "分析文本的统计信息或进行文本变换。支持的操作："
                "'count'=统计字数/词数/字符数，"
                "'reverse'=反转文本，"
                "'uppercase'=转大写，"
                "'lowercase'=转小写。"
                "例如：text_stats(text='Hello World', operation='count') 返回字数和词数。"
            ),
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="要分析的文本",
                    required=True,
                ),
                ToolParameter(
                    name="operation",
                    type="string",
                    description="操作类型：'count'=统计，'reverse'=反转，'uppercase'=转大写，'lowercase'=转小写",
                    required=False,
                    default="count",
                ),
            ],
        )

    def execute(self, text: str, operation: str = "count") -> str:  # type: ignore[override]
        if operation == "count":
            char_count = len(text)
            # 按空白分词算词数（简化实现）
            word_count = len(text.split())
            # 中文字数（Unicode 范围 一-鿿）
            chinese_count = sum(1 for c in text if "一" <= c <= "鿿")
            return (
                f"统计结果：总字符数={char_count}，词数={word_count}，"
                f"中文字数={chinese_count}，英文/数字/符号={char_count - chinese_count}"
            )
        elif operation == "reverse":
            return f"反转结果：{text[::-1]}"
        elif operation == "uppercase":
            return f"大写结果：{text.upper()}"
        elif operation == "lowercase":
            return f"小写结果：{text.lower()}"
        else:
            return f"不支持的操作：'{operation}'。支持：count, reverse, uppercase, lowercase"
