"""框架选型指南 —— 演示脚本。

运行方式:
  poetry run python -m study_agent.framework_guide_demo

输出完整的 Agent 框架选型指南，包括:
  1. 7 种框架/范式信息卡片
  2. 6 维度对比矩阵
  3. 选型决策树
  4. 同一任务 5 种实现对比
  5. 6 个面试高频问题 + 回答要点
  6. 一页纸速查表
"""

from __future__ import annotations

from study_agent.agent.framework_guide import (
    print_full_guide,
)

if __name__ == "__main__":
    print_full_guide()

    # 额外: 打印本周所有框架的代码文件位置
    print("=" * 60)
    print("  Week 5 全部代码索引")
    print("=" * 60)
    print()
    print("学习顺序回顾:")
    print()
    files = [
        ("Day 1", "LangChain AgentExecutor 分析", "docs/week5/day1_sequence.html (调用时序图)"),
        ("Day 2", "ReAct Agent 手写", "src/study_agent/agent/react_agent.py"),
        ("Day 3", "Plan-Execute Agent 手写", "src/study_agent/agent/plan_execute_agent.py"),
        ("Day 4", "LangGraph 条件分支", "src/study_agent/agent/langgraph_router.py"),
        ("Day 5", "CrewAI vs AutoGen 对比", "src/study_agent/agent/autogen_research_team.py"),
        ("Day 5", "CrewAI API 展示", "src/study_agent/agent/crewai_research_crew.py"),
        ("Day 6", "框架选型指南 (本文档)", "src/study_agent/agent/framework_guide.py"),
    ]
    for day, desc, path in files:
        print(f"  {day:6s}  {desc:30s}  {path}")
    print()
    print("Week 5 核心收获:")
    print("  从底层的 ReAct 循环，到中层的 LangGraph 编排，")
    print("  到高层的 CrewAI/AutoGen 多 Agent 协作——")
    print("  每一层都手写过，也都用过框架。")
    print("  理解了'框架在帮我做什么'，也知道了'框架在限制我什么'。")
