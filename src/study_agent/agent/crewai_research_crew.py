"""CrewAI 多 Agent 协作 —— 研究员 + 写手 双 Agent 团队。

用 CrewAI 实现:
  - Researcher Agent: 角色扮演式定义（role + goal + backstory）
  - Writer Agent:    同样的角色定义模式
  - Sequential Process: 先研究 -> 再写作，严格串行

核心概念（按学习顺序）：
  1. Agent   —— 通过 role/goal/backstory 定义角色，比 AutoGen 更"拟人化"
  2. Task    —— 独立的任务对象，绑定到特定 Agent，包含 expected_output
  3. Crew    —— Agent + Task 的容器，用 Process 控制执行顺序
  4. Process —— sequential（串行）或 hierarchical（层级委派）

注意：此文件因 CrewAI 与 langgraph 的依赖冲突无法在当前环境运行。
      langgraph 需要 langchain-core >= 1.4, CrewAI 需要 langchain < 0.2.0。
      两个需求互斥。此代码展示 API 模式供学习参考。
"""

from __future__ import annotations

import os

# ═══════════════════════════════════════════════════════════════════
# CrewAI vs AutoGen: 两种不同的 Agent 定义方式
# ═══════════════════════════════════════════════════════════════════
#
# CrewAI 的设计哲学是"角色扮演"（Role-Playing）。
# 每个 Agent 不是冷冰冰的"名字+系统提示"，而是有职位、目标、背景故事的"角色"。
#
# 对比:
#
#   AutoGen:
#     agent = AssistantAgent(
#         name="researcher",
#         system_message="你是一个研究员...",
#     )
#     -> 定义方式: 名字 + 指令
#     -> 风格: 工具型——"我告诉你怎么做"
#
#   CrewAI:
#     agent = Agent(
#         role="Senior Researcher",
#         goal="Uncover cutting-edge insights about {topic}",
#         backstory="Driven by curiosity, you are a veteran researcher...",
#     )
#     -> 定义方式: 职位 + 目标 + 背景故事
#     -> 风格: 角色型——"我知道我是谁，我知道我要做什么"
#
#  哪种更好？
#    - CrewAI 的角色定义更丰富，LLM 更容易"入戏"，输出风格更稳定
#    - AutoGen 的系统提示更直接，适合需要精确控制行为的场景
#    - 没有绝对的好坏，取决于你的任务需要多强的"角色一致性"

# 以下代码按照 CrewAI 官方 API 编写，展示了完整的 CrewAI 使用模式。
# 由于依赖冲突，这些 import 在当前环境无法执行，但代码结构是标准的。


def build_crewai_team():
    """构建 CrewAI 研究员+写手团队。

    这个函数展示了 CrewAI 的完整工作流:
      1. 定义 Agent（角色扮演）
      2. 定义 Task（任务描述 + 期望输出）
      3. 创建 Crew（Agent + Task 的编排器）
      4. kickoff() 启动执行

    返回:
      (crew, research_task, write_task) 元组
    """
    # 避免 import 错误导致整个模块无法加载
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError:
        raise ImportError(
            "CrewAI 未安装或存在依赖冲突。\n"
            "当前环境: langgraph 需要 langchain-core >= 1.4, "
            "但 CrewAI 需要 langchain < 0.2.0。\n"
            "二者无法共存。如需运行此代码，请在新虚拟环境中独立安装 crewai。"
        )

    # ── Step 1: 定义 LLM ──
    # CrewAI 0.11.x 使用 langchain 的 ChatOpenAI 作为 LLM 后端
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    )

    # ── Step 2: 定义 Agent ──
    # 每个 Agent 需要三个核心字段:
    #   role:      职位名称，定义了"我是做什么的"
    #   goal:      工作目标，定义了"我要达成什么"
    #   backstory: 背景故事，定义了"我的经验/风格/偏好"
    #
    # LLM 会基于这三个字段来"扮演"这个角色。
    # backstory 越具体，Agent 的行为越一致。

    researcher = Agent(
        role="Senior Technology Researcher",
        goal="Research {topic} thoroughly and produce a comprehensive, structured report",
        backstory=(
            "You are a veteran technology researcher with 15 years of experience. "
            "You have a knack for breaking down complex technical topics into "
            "clear categories: definitions, core concepts, how-it-works, use cases, "
            "and trade-offs. Your reports are known for being well-organized and "
            "always distinguishing between established facts and informed opinions."
        ),
        verbose=True,
        allow_delegation=False,  # 研究员不委派任务给别人
        llm=llm,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal=(
            "Transform the research report into an engaging, beginner-friendly "
            "article in Chinese, under 1200 words"
        ),
        backstory=(
            "You are a skilled technical writer who specializes in making complex "
            "topics accessible to beginners. Your superpower is finding the perfect "
            "real-life analogy for any abstract concept. You write in Chinese and "
            "your articles always follow a clear structure: hook -> problem -> "
            "concepts with analogies -> code examples -> summary."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Step 3: 定义 Task ──
    # Task 不是函数调用，而是一个"任务描述 + 期望输出"的声明。
    # CrewAI 会把这个描述和期望输出格式一起发给 Agent。
    #
    # 关键参数:
    #   description:      任务描述（会发给 Agent）
    #   expected_output:  期望输出格式（Agent 会尽量按这个格式回复）
    #   agent:            绑定到哪个 Agent
    #
    # 注意 description 里的 {topic} —— 这是模板变量，
    # 在 kickoff() 时传入，会被替换为实际值。

    research_task = Task(
        description=(
            "Research the following topic: {topic}\n\n"
            "Please provide a structured report covering:\n"
            "1. What is it? (definition and background)\n"
            "2. Core concepts and terminology\n"
            "3. How it works (mechanism, lifecycle)\n"
            "4. Common use cases and real-world examples\n"
            "5. Pros and cons / trade-offs\n"
            "6. Code example (if applicable)\n\n"
            "Label each piece of information as [FACT] or [OPINION]."
        ),
        expected_output=(
            "A structured research report with sections for: "
            "Definition, Core Concepts, Mechanism, Use Cases, Trade-offs, "
            "and Code Example. Each section clearly labeled."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Write a beginner-friendly article based on the research report.\n"
            "Target audience: people who know basic Python but nothing about {topic}.\n\n"
            "Requirements:\n"
            "- Title should be catchy\n"
            "- Start with a real-life scenario showing why this matters\n"
            "- Explain each concept with a real-life analogy\n"
            "- Include and explain the code example line by line\n"
            "- Keep it under 1200 words\n"
            "- Write in Chinese"
        ),
        expected_output=(
            "A well-structured Chinese article with: catchy title, "
            "hook/scenario intro, concepts with analogies, "
            "code walkthrough, and a summary."
        ),
        agent=writer,
    )

    # ── Step 4: 创建 Crew 并配置执行流程 ──
    # Process.sequential: Task 按添加顺序依次执行
    #   先 research_task (研究员) -> 再 write_task (写手)
    #
    # Process.hierarchical: 有一个"经理"Agent 自动分配任务
    #   适合 3+ 个 Agent 的复杂协作场景

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,  # 串行执行
        verbose=True,  # 打印详细日志，便于调试
    )

    return crew, research_task, write_task


# ═══════════════════════════════════════════════════════════════════
# 运行接口
# ═══════════════════════════════════════════════════════════════════


def run_crewai(topic: str) -> str:
    """运行 CrewAI 研究+写作任务。

    参数:
      topic: 研究主题

    返回:
      Crew 执行结果的字符串表示
    """
    crew, _, _ = build_crewai_team()

    # kickoff() 启动任务执行。
    # inputs 参数传入模板变量，替换 Task description 中的 {topic}。
    result = crew.kickoff(inputs={"topic": topic})

    return str(result)


# ═══════════════════════════════════════════════════════════════════
# 两种 Process 的区别
# ═══════════════════════════════════════════════════════════════════
#
# Process.sequential (串行):
#   research_task -> write_task
#   Task 严格按照定义顺序执行，前一个完成才开始后一个。
#   适合: 有明确依赖关系的任务（写文章依赖研究结果）
#
# Process.hierarchical (层级委派):
#   自动创建一个"Crew Manager"Agent，
#   它分析所有 Task 后决定"谁来做哪个、先做什么后做什么"。
#   适合: 3+ Agent 的复杂协作、任务间有复杂依赖关系
#
# 今天的 2-Agent 场景用 sequential 就够。
# hierarchical 在 3+ Agent 时更有价值。


if __name__ == "__main__":
    print("CrewAI 实现因依赖冲突无法直接运行。")
    print("请查看代码中的 API 模式说明和注释。")
    print()
    print("如需运行，请在新虚拟环境中:")
    print("  pip install crewai langchain-openai")
    print("然后运行: python -m study_agent.agent.crewai_research_crew")
