"""Week 5 Day 5 演示 —— CrewAI vs AutoGen 双框架对比。

运行 AutoGen 版本:
  poetry run python -m study_agent.multi_agent_demo

本脚本会:
  1. 启动 AutoGen 的研究员+写手团队
  2. 实时展示每个 Agent 的输出
  3. 输出两种框架的 API 对比总结
"""

from __future__ import annotations

import asyncio


async def demo_autogen(topic: str) -> None:
    """演示 AutoGen 版本（可运行）。"""
    from study_agent.agent.autogen_research_team import run_and_print

    print("=" * 60)
    print("AutoGen: 研究员 + 写手 双 Agent 协作")
    print("=" * 60)
    print()
    print("架构:")
    print("  RoundRobinGroupChat")
    print("    +-- researcher (AssistantAgent)")
    print("    +-- writer (AssistantAgent)")
    print("  终止条件: [ARTICLE_COMPLETE] 文本标记")
    print()
    print("执行流程: 用户提需求 -> researcher 调研 -> writer 写文章 -> 终止")
    print()

    await run_and_print(topic)


def show_crewai_api() -> None:
    """展示 CrewAI 的 API 模式（因依赖冲突无法运行）。"""
    print("=" * 60)
    print("CrewAI: 研究员 + 写手 双 Agent 协作 (API 展示)")
    print("=" * 60)
    print()
    print("架构:")
    print("  Crew (Process.sequential)")
    print("    +-- Task 1: research_task -> Agent: Senior Researcher")
    print("    +-- Task 2: write_task     -> Agent: Technical Writer")
    print("  执行顺序: research_task -> write_task (严格串行)")
    print()
    print("CrewAI 的核心 API 模式:")
    print()
    print("  # 1. 定义 Agent (角色扮演)")
    print("  researcher = Agent(")
    print("      role='Senior Researcher',")
    print("      goal='Research {topic} thoroughly',")
    print("      backstory='Veteran researcher with 15 years...',")
    print("      llm=llm,")
    print("  )")
    print()
    print("  # 2. 定义 Task (任务描述 + 期望输出)")
    print("  research_task = Task(")
    print("      description='Research: {topic}...',")
    print("      expected_output='A structured report with...',")
    print("      agent=researcher,")
    print("  )")
    print()
    print("  # 3. 创建 Crew 并启动")
    print("  crew = Crew(")
    print("      agents=[researcher, writer],")
    print("      tasks=[research_task, write_task],")
    print("      process=Process.sequential,")
    print("  )")
    print("  result = crew.kickoff(inputs={'topic': topic})")
    print()


def show_comparison() -> None:
    """CrewAI vs AutoGen 核心对比。"""
    print("=" * 60)
    print("CrewAI vs AutoGen: 核心对比")
    print("=" * 60)
    print()

    comparisons = [
        (
            "设计哲学",
            "CrewAI",
            "AutoGen",
            "角色扮演 (Role-Playing)\nAgent = 职位+目标+背景故事",
            "对话驱动 (Conversation-Driven)\nAgent = 名字+系统消息",
        ),
        (
            "Agent 定义",
            "CrewAI",
            "AutoGen",
            "Agent(role=..., goal=..., backstory=...)",
            "AssistantAgent(name=..., system_message=...)",
        ),
        (
            "任务管理",
            "CrewAI",
            "AutoGen",
            "Task 对象: description + expected_output\n预定义执行顺序",
            "对话流: Agent 之间自由交流\n通过终止条件控制流程",
        ),
        (
            "执行模型",
            "CrewAI",
            "AutoGen",
            "Process.sequential / Process.hierarchical\n任务级编排",
            "RoundRobinGroupChat / SelectorGroupChat\n对话轮次级编排",
        ),
        (
            "工具支持",
            "CrewAI",
            "AutoGen",
            "通过 langchain tools 集成",
            "原生 FunctionTool + 函数直接注册",
        ),
        (
            "学习曲线",
            "CrewAI",
            "AutoGen",
            "低: API 直观，概念少 (Agent/Task/Crew)",
            "中: 概念较多 (Agent/Team/Termination/Runtime)",
        ),
        (
            "适合场景",
            "CrewAI",
            "AutoGen",
            "固定流程的多 Agent 协作\n(先A后B再C，流程明确)",
            "需要 Agent 间动态协商的场景\n(讨论、辩论、迭代优化)",
        ),
        (
            "输出可见性",
            "CrewAI",
            "AutoGen",
            "verbose=True 打印日志\n最终拿到 result 字符串",
            "run_stream() 逐条获取每轮对话\n可实时看到每个 Agent 说什么",
        ),
    ]

    for title, left_name, right_name, left_val, right_val in comparisons:
        print(f"--- {title} ---")
        print(f"  {left_name:12s} | {right_name}")
        print(
            f"  {left_val.replace(chr(10), chr(10) + ' ' * 14):12s}   | {right_val.replace(chr(10), chr(10) + ' ' * 14)}"
        )
        print()

    print("=" * 60)
    print("选型建议")
    print("=" * 60)
    print()
    print("选 CrewAI 如果你:")
    print("  - 任务流程固定且清晰（第一步做什么、第二步做什么）")
    print("  - 喜欢「角色扮演」式的 Agent 定义")
    print("  - 需要 low-code / 拖拽式的可视化（CrewAI Enterprise）")
    print("  - 团队 Agent 数量 2-5 个")
    print()
    print("选 AutoGen 如果你:")
    print("  - 需要 Agent 之间灵活协商（而不是固定流程）")
    print("  - 需要精细控制每轮对话（streaming 粒度为单条消息）")
    print("  - 需要嵌套团队（Team of Teams）")
    print("  - 需要复杂的终止条件（多种条件组合）")
    print()
    print("两者都学为什么？")
    print("  - 面试中两个框架都可能被问到")
    print("  - 理解了两种设计哲学后，换第三个框架只需 1-2 天")
    print("  - 真实项目中可能混用：CrewAI 做固定流程，AutoGen 做开放讨论")
    print()
    print("LangGraph 的位置:")
    print("  LangGraph 是更底层的编排框架。")
    print("  CrewAI 和 AutoGen 是更高层的封装（内置了角色、任务、对话管理等概念）。")
    print("  你可以在 CrewAI Agent 内部使用 LangGraph 做复杂推理，")
    print("  也可以在 AutoGen Agent 内部使用 LangGraph 做状态机流程。")
    print("  三者不是替代关系，而是不同抽象层次的工具。")


async def main() -> None:
    """主入口。"""
    topic = "Python 异步编程 (asyncio) —— 协程、事件循环、await 的工作原理"

    # 1. 展示 CrewAI API（概念层面）
    show_crewai_api()

    # 2. 运行 AutoGen（实际执行）
    await demo_autogen(topic)

    # 3. 输出对比总结
    show_comparison()


if __name__ == "__main__":
    asyncio.run(main())
