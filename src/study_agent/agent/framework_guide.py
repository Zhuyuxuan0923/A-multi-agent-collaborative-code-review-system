"""Agent 框架选型指南 —— Week 5 收官之作。

对 Week 5 学过的所有 Agent 框架/范式做系统性横向对比：
  - LangChain AgentExecutor: 高层封装，开箱即用
  - ReAct (手写): 推理+行动循环，灵活但需自己写
  - Plan-Execute (手写): 先计划后执行，结构化任务首选
  - LangGraph: 有状态图编排，条件分支
  - CrewAI: 角色扮演式多 Agent 协作
  - AutoGen: 对话驱动式多 Agent 协作
  - 自研 (OpenAI SDK): 零依赖，完全控制

本模块提供：
  1. 框架分类体系（抽象层级）
  2. 多维度对比（易用性/灵活性/调试/生态/成本）
  3. 决策矩阵（什么场景选什么框架）
  4. 同一任务的 5 种实现对比
  5. 面试高频问题 + 回答要点
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# 1. 框架分类体系 —— 按抽象层级排序
# ═══════════════════════════════════════════════════════════════════


class AbstractionLevel(Enum):
    """框架的抽象层级。

    层级越低 = 越灵活、越需要自己写代码
    层级越高 = 越开箱即用、越受框架约束
    """

    RAW_SDK = "原生 SDK"  # 直接调 OpenAI / Anthropic API
    PATTERN = "设计范式"  # ReAct / Plan-Execute 等模式
    ORCHESTRATOR = "编排引擎"  # LangGraph 等图编排
    AGENT_FRAMEWORK = "Agent 框架"  # LangChain AgentExecutor
    MULTI_AGENT = "多 Agent 框架"  # CrewAI / AutoGen


@dataclass
class FrameworkInfo:
    """一个框架/范式的完整信息卡片。"""

    name: str
    tagline: str  # 一句话定位
    level: AbstractionLevel
    design_philosophy: str  # 设计哲学
    best_for: list[str]  # 最适合的场景
    not_for: list[str]  # 不适合的场景
    learning_curve: str  # 学习曲线: 低/中/高
    our_experience: str  # 我们在 Week 5 的实际体验


# ── 7 个框架/范式信息卡片 ──

FRAMEWORKS: list[FrameworkInfo] = [
    FrameworkInfo(
        name="自研 (OpenAI/Anthropic SDK)",
        tagline="零依赖，完全控制，一行不多",
        level=AbstractionLevel.RAW_SDK,
        design_philosophy=(
            "不引入任何 Agent 框架。直接使用 LLM SDK 的 chat/completion 接口，"
            "自己管理对话历史、工具调用循环、重试逻辑。框架代码就是你的业务代码。"
        ),
        best_for=[
            "原型验证（快速试一个想法能不能 work）",
            "简单 Agent（单轮工具调用、固定流程）",
            "对依赖敏感的项目（安全审查、嵌入式部署）",
            "学习和理解 Agent 底层机制（Week 5 Day 2/3 就是这样做的）",
        ],
        not_for=[
            "复杂多步推理（手写状态管理容易出错）",
            "多 Agent 协作（需要自己实现通信协议）",
            "频繁变更的流程（改流程 = 改代码 = 重新测试）",
        ],
        learning_curve="低（只需了解 SDK，但实现复杂功能需要强工程能力）",
        our_experience=(
            "Day 2 手写 ReAct 时：约 100 行代码实现 Thought/Action/Observation 循环。"
            "灵活但容易写出 bug（比如忘了把 observation 拼回 messages）。"
            "适合理解原理，不适合生产环境。"
        ),
    ),
    FrameworkInfo(
        name="LangChain AgentExecutor",
        tagline="Agent 框架的'Spring Boot'——功能全但很重",
        level=AbstractionLevel.AGENT_FRAMEWORK,
        design_philosophy=(
            "高层抽象：Agent + Tool + AgentExecutor 三个概念覆盖 80% 的 Agent 场景。"
            "内置 ReAct / OpenAI Functions / Structured Chat 等多种 Agent 类型。"
            "丰富的 Tool 生态（Google Search / Wikipedia / SQL / Python REPL）。"
        ),
        best_for=[
            "快速原型（几行代码跑通一个完整 Agent）",
            "需要大量内置工具的场景（搜索/数据库/文件系统）",
            "团队对 Agent 不熟悉（框架帮你做了很多决策）",
            "短期项目或 Hackathon",
        ],
        not_for=[
            "需要精细控制 Agent 行为的场景（框架的「魔法」太多）",
            "性能敏感场景（框架有额外开销）",
            "需要自定义 Agent 循环的场景（绕过框架比直接用 SDK 还麻烦）",
            "长期维护的项目（LangChain 版本更新频繁，API break 多）",
        ],
        learning_curve="中（API 简单，但调试和定制困难）",
        our_experience=(
            "Day 1 研究 AgentExecutor 源码时发现：框架在背后做了太多事——"
            "自动管理 message 历史、自动解析 tool call、自动拼 prompt。"
            "方便是真方便，但出了 bug 要看源码才能定位。"
        ),
    ),
    FrameworkInfo(
        name="ReAct 范式 (手写)",
        tagline="推理+行动循环——Agent 的'Hello World'",
        level=AbstractionLevel.PATTERN,
        design_philosophy=(
            "Thought -> Action -> Observation 三步循环。"
            "LLM 在每个回合先思考（Thought），再选择行动（Action），"
            "执行后得到观察（Observation），喂回 LLM 继续。"
            "这是一个设计模式，不绑定任何框架。"
        ),
        best_for=[
            "需要多步推理的探索性任务（'帮我调研 X'）",
            "工具调用场景（搜索/计算/API 调用）",
            "理解和教学 Agent 原理（ReAct 论文是必读）",
        ],
        not_for=[
            "流程确定的批量任务（ReAct 的自由循环是浪费）",
            "对延迟敏感的场景（每步都要调 LLM）",
            "需要结构化输出的场景（ReAct 输出格式不稳定）",
        ],
        learning_curve="低（概念少，实现简单）",
        our_experience=(
            "Day 2 手写实现：核心循环不到 50 行。"
            "最大的坑是 LLM 的输出解析——有时 Thought 和 Action 混在一起，"
            "有时直接跳过 Thought 给 Action。需要做 robust parsing。"
        ),
    ),
    FrameworkInfo(
        name="Plan-Execute 范式 (手写)",
        tagline="先计划后执行——结构化任务的最佳选择",
        level=AbstractionLevel.PATTERN,
        design_philosophy=(
            "Plan -> Execute -> Replan 循环。"
            "先让 LLM 生成 JSON 格式的执行计划，然后逐步执行每个步骤，"
            "每步执行后校验结果，如果偏离计划就重新规划。"
            "核心优势：可审计（计划是 JSON，每一步都可追溯）。"
        ),
        best_for=[
            "结构化任务（代码审查、文档生成、数据分析报告）",
            "需要可审计性的场景（每一步都能追溯「为什么这样做」）",
            "任务步骤之间有依赖关系（第 3 步依赖第 1 步的结果）",
        ],
        not_for=[
            "探索性任务（无法提前规划所有步骤）",
            "对话式交互（用户随时可能改变方向）",
            "简单任务（搭框架的成本比任务本身还高）",
        ],
        learning_curve="中（需要设计 JSON 计划结构和校验逻辑）",
        our_experience=(
            "Day 3 实现：核心价值在'Replan'。第一次计划可能不全，"
            "但执行到一半发现信息不够时，自动触发的 Replan 让 Agent 更鲁棒。"
            "JSON 计划让调试变得简单——直接看计划就知道 Agent 在想什么。"
        ),
    ),
    FrameworkInfo(
        name="LangGraph",
        tagline="Agent 工作流的'图编辑器'——状态+节点+边",
        level=AbstractionLevel.ORCHESTRATOR,
        design_philosophy=(
            "有状态图（StateGraph）：State 在节点之间流动，条件边决定分支。"
            "比 ReAct/Plan-Execute 更底层（它们是 Agent 范式，LangGraph 是编排引擎）。"
            "你可以在 LangGraph 的节点里放 ReAct Agent，也可以用 Plan-Execute。"
        ),
        best_for=[
            "有明确分支的 Agent 工作流（不同问题走不同路径）",
            "需要状态持久化的场景（中断后恢复）",
            "复杂多步骤流程（A->B->根据B的结果走C或D）",
            "作为多个 Agent 范式的'胶水层'",
        ],
        not_for=[
            "简单的线性流程（用 LangGraph 是杀鸡用牛刀）",
            "团队不熟悉图/状态机概念（学习曲线陡）",
            "快速原型（搭图比写 if/else 慢）",
        ],
        learning_curve="高（需要理解 StateGraph / Node / Edge / Conditional Edge / Compile）",
        our_experience=(
            "Day 4 实现：核心价值在条件边。router_node 根据意图分叉到 "
            "search_node 或 chat_node，这个能力是 ReAct/Plan-Execute 做不到的。"
            "但概念多（5 个核心 API），学习曲线是 Week 5 最陡的。"
        ),
    ),
    FrameworkInfo(
        name="CrewAI",
        tagline="多 Agent 协作的'公司组织架构'",
        level=AbstractionLevel.MULTI_AGENT,
        design_philosophy=(
            "角色扮演（Role-Playing）：Agent 有 role/goal/backstory，"
            "Task 有 description/expected_output。"
            "用 Process（sequential/hierarchical）控制执行顺序。"
            "设计哲学：让 LLM '入戏'，通过角色定义来约束行为。"
        ),
        best_for=[
            "固定流程的多 Agent 协作（研究->写作->审核）",
            "角色分工明确的场景（每个人做什么很清楚）",
            "需要低代码/拖拽式界面的场景（CrewAI Enterprise）",
            "2-5 个 Agent 的小团队协作",
        ],
        not_for=[
            "需要 Agent 间自由协商的场景（CrewAI 的流程是预定义的）",
            "单个 Agent 内部需要复杂推理（应该配合 LangGraph 使用）",
            "高频调用的场景（每次 kickoff 都创建新的执行上下文）",
            "需要精细控制每个 Agent 的 LLM 参数的场景",
        ],
        learning_curve="低（Agent/Task/Crew 三个概念）",
        our_experience=(
            "Day 5 尝试安装但遇到 langchain-core 版本冲突。"
            "代码层面：API 很直观，role+goal+backstory 的角色定义方式"
            "让 LLM 的输出风格更稳定。但依赖管理是一个实际问题。"
        ),
    ),
    FrameworkInfo(
        name="AutoGen",
        tagline="多 Agent 协作的'圆桌会议'",
        level=AbstractionLevel.MULTI_AGENT,
        design_philosophy=(
            "对话驱动（Conversation-Driven）：Agent 通过自由对话来协作。"
            "不是预定义流程，而是 Agent 互相交谈、追问、修正。"
            "终止条件（TerminationCondition）控制对话何时结束。"
            "GroupChat 管理多 Agent 的发言顺序。"
        ),
        best_for=[
            "需要 Agent 间协商的任务（讨论方案、迭代优化）",
            "Agent 需要追问/澄清的场景（写手可以反问研究员）",
            "需要嵌套团队（Team of Teams）的复杂场景",
            "需要实时流式输出每个 Agent 发言的场景",
        ],
        not_for=[
            "固定流程的任务（AutoGen 的灵活性反而增加不确定性）",
            "对延迟敏感的场景（Agent 间的来回对话增加总耗时）",
            "简单的单 Agent 任务（AutoGen 的团队机制是多余的）",
        ],
        learning_curve="中（概念多：Agent/Team/Termination/Runtime/ModelClient）",
        our_experience=(
            "Day 5 实际运行成功：研究员+写手团队完成了异步编程科普文章。"
            "最大的坑是终止条件——第一次忘记了设计'刹车'，"
            "Agent 陷入互相感谢的聊天循环。"
            "接入 DeepSeek 需要手动配置 ModelInfo。"
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════
# 2. 多维度对比矩阵
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ComparisonMatrix:
    """框架对比矩阵 —— 每个维度一个评分（1-5）加解释。"""

    dimension: str
    scores: dict[str, int]  # framework_name -> score (1-5)
    explanation: dict[str, str]  # framework_name -> 一句话解释


COMPARISONS: list[ComparisonMatrix] = [
    ComparisonMatrix(
        dimension="易用性",
        scores={
            "自研": 2,
            "LangChain": 4,
            "ReAct": 3,
            "Plan-Execute": 3,
            "LangGraph": 2,
            "CrewAI": 4,
            "AutoGen": 3,
        },
        explanation={
            "自研": "一切自己写，简单任务容易，复杂任务难",
            "LangChain": "几行代码跑通 Agent，但定制困难",
            "ReAct": "循环逻辑简单，解析 LLM 输出需要技巧",
            "Plan-Execute": "JSON 计划结构需要精心设计",
            "LangGraph": "5 个核心概念，需要图/状态机基础",
            "CrewAI": "Agent/Task/Crew 三个概念，API 直观",
            "AutoGen": "概念多，但流式 API 调试友好",
        },
    ),
    ComparisonMatrix(
        dimension="灵活性",
        scores={
            "自研": 5,
            "LangChain": 2,
            "ReAct": 4,
            "Plan-Execute": 4,
            "LangGraph": 5,
            "CrewAI": 2,
            "AutoGen": 4,
        },
        explanation={
            "自研": "没有框架约束，想怎么改就怎么改",
            "LangChain": "高度封装，绕过框架的「魔法」很痛苦",
            "ReAct": "循环骨架简单，容易魔改",
            "Plan-Execute": "计划格式和执行逻辑都可定制",
            "LangGraph": "图结构完全自由，节点里可以放任何逻辑",
            "CrewAI": "流程固定（sequential/hierarchical），不易跳出",
            "AutoGen": "对话模式自由，Agent 可追问、可协商",
        },
    ),
    ComparisonMatrix(
        dimension="调试体验",
        scores={
            "自研": 5,
            "LangChain": 1,
            "ReAct": 4,
            "Plan-Execute": 5,
            "LangGraph": 4,
            "CrewAI": 2,
            "AutoGen": 3,
        },
        explanation={
            "自研": "代码全是你的，加 print/日志非常容易",
            "LangChain": "框架内部有大量隐式行为，出问题要看源码",
            "ReAct": "循环简单，每步打印 Thought/Action/Observation 即可",
            "Plan-Execute": "JSON 计划就是天然的 trace，可审计",
            "LangGraph": "内置 ASCII/Mermaid 图可视化 + State 追踪",
            "CrewAI": "verbose=True 有日志，但内部行为仍不透明",
            "AutoGen": "run_stream() 逐条看到每个 Agent 说什么",
        },
    ),
    ComparisonMatrix(
        dimension="生产就绪度",
        scores={
            "自研": 3,
            "LangChain": 3,
            "ReAct": 2,
            "Plan-Execute": 2,
            "LangGraph": 4,
            "CrewAI": 3,
            "AutoGen": 4,
        },
        explanation={
            "自研": "可靠性取决于你的工程能力",
            "LangChain": "社区大但版本不稳定，生产环境需锁定版本",
            "ReAct": "只是模式，需要大量工程化才能上生产",
            "Plan-Execute": "同上，校验逻辑是生产部署的关键",
            "LangGraph": "有 checkpoint 持久化、streaming、错误恢复",
            "CrewAI": "功能全但生态仍在快速变化",
            "AutoGen": "Microsoft 维护，有成熟的 runtime 和 streaming",
        },
    ),
    ComparisonMatrix(
        dimension="多 Agent 支持",
        scores={
            "自研": 1,
            "LangChain": 2,
            "ReAct": 1,
            "Plan-Execute": 1,
            "LangGraph": 3,
            "CrewAI": 5,
            "AutoGen": 5,
        },
        explanation={
            "自研": "需要自己实现 Agent 间通信协议",
            "LangChain": "有 RunnableBranch 等基础支持，但非主打",
            "ReAct": "单 Agent 范式",
            "Plan-Execute": "单 Agent 范式（但可嵌套）",
            "LangGraph": "可构建多 Agent 图，但不如 CrewAI/AutoGen 方便",
            "CrewAI": "天生为多 Agent 设计，角色扮演+任务分配",
            "AutoGen": "天生为多 Agent 对话设计，团队+终止条件",
        },
    ),
    ComparisonMatrix(
        dimension="学习成本",
        scores={
            "自研": 1,
            "LangChain": 3,
            "ReAct": 2,
            "Plan-Execute": 2,
            "LangGraph": 4,
            "CrewAI": 2,
            "AutoGen": 3,
        },
        explanation={
            "自研": "只需了解 LLM SDK，概念最简",
            "LangChain": "概念繁杂（Chain/Agent/Tool/Memory/Callback）",
            "ReAct": "一个论文的概念，半天可掌握",
            "Plan-Execute": "一个模式，JSON schema 设计是主要工作",
            "LangGraph": "StateGraph/Node/Edge/ConditionalEdge/Compile 五概念",
            "CrewAI": "Agent/Task/Crew 三概念，API 符合直觉",
            "AutoGen": "Agent/Team/Termination/Runtime/ModelClient 五概念",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════
# 3. 决策树 —— 根据需求选择框架
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DecisionNode:
    """决策树的一个节点。"""

    question: str  # 问自己的问题
    hint: str  # 帮助你回答的提示
    yes: str | None = None  # 回答"是"时的建议
    no: str | None = None  # 回答"否"时的下一步问题 key


DECISION_TREE: dict[str, DecisionNode] = {
    "start": DecisionNode(
        question="任务需要多个 Agent 协作吗？",
        hint=(
            "多 Agent = 至少 2 个不同角色的 Agent 配合完成任务。"
            "例如：研究员+写手、前端+后端+设计。"
            "单 Agent = 一个 Agent 包揽所有事。"
        ),
        yes="multi_agent",
        no="single_agent",
    ),
    "multi_agent": DecisionNode(
        question="协作流程是固定的还是需要协商？",
        hint=(
            "固定流程 = 第一步 A 做 X，第二步 B 做 Y，流程不会变。"
            "需要协商 = A 和 B 需要讨论、追问、修正才能完成任务。"
        ),
        yes="选 CrewAI（固定流程，角色扮演，有明确的执行顺序）",
        no="选 AutoGen（开放对话，Agent 之间可以追问、协商、迭代优化）",
    ),
    "single_agent": DecisionNode(
        question="任务流程有明确的步骤和分支吗？",
        hint=(
            "有分支 = 根据中间结果走不同路径。"
            "例如：技术问题走搜索，闲聊走对话。"
            "无分支 = 线性流程，没有条件判断。"
        ),
        yes="langgraph_or_pattern",
        no="simple_choice",
    ),
    "langgraph_or_pattern": DecisionNode(
        question="状态管理和可恢复性重要吗？",
        hint=(
            "重要 = 任务可能中断后需要恢复（长时间运行的任务）。"
            "需要可视化工作流、需要对流程做审计。"
            "不重要 = 任务每次从头开始，不需要保存中间状态。"
        ),
        yes="选 LangGraph（状态持久化、条件分支、图可视化）",
        no="simple_choice",
    ),
    "simple_choice": DecisionNode(
        question="需要快速原型还是精细控制？",
        hint=(
            "快速原型 = 几小时内要跑通，不在乎框架开销。"
            "精细控制 = 需要控制每个细节（prompt 格式、重试策略、错误处理）。"
        ),
        yes="选 LangChain AgentExecutor（内置工具多、几行代码出 Demo）",
        no="选 自研 (SDK) 或手写 ReAct/Plan-Execute（完全可控、零框架依赖）",
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 4. 同一任务的 5 种实现对比
# ═══════════════════════════════════════════════════════════════════
#
# 任务: "用户问 'Python asyncio 是什么'，搜索相关信息并给出回答"

SAME_TASK_COMPARISON = """
同一任务: "用户问 'Python asyncio 是什么' -> 搜索 -> 回答"

┌──────────────────┬──────────────────────────────────────────────┐
│ 方案              │ 核心代码                                      │
├──────────────────┼──────────────────────────────────────────────┤
│ 自研 (SDK)        │ messages = [user_msg]                        │
│                   │ reply = client.chat(messages)                │
│                   │ # 手动拼 search result 到 prompt              │
│                   │ # 代码量: ~30 行                              │
├──────────────────┼──────────────────────────────────────────────┤
│ LangChain         │ agent = create_react_agent(llm, tools)       │
│ AgentExecutor     │ executor = AgentExecutor(agent, tools)       │
│                   │ result = executor.invoke({"input": q})       │
│                   │ # 代码量: ~10 行（但内部做了大量隐式操作）      │
├──────────────────┼──────────────────────────────────────────────┤
│ ReAct (手写)      │ for step in range(max_steps):                │
│                   │     thought = llm.think(messages)            │
│                   │     action = parse_action(thought)            │
│                   │     observation = execute(action)             │
│                   │     messages.append(observation)              │
│                   │ # 代码量: ~50 行（每步都可见可控）             │
├──────────────────┼──────────────────────────────────────────────┤
│ Plan-Execute      │ plan = llm.generate_plan(task)  # JSON       │
│ (手写)            │ for step in plan["steps"]:                   │
│                   │     result = execute_step(step)              │
│                   │     if needs_replan(result):                 │
│                   │         plan = llm.replan(task, result)      │
│                   │ # 代码量: ~80 行（JSON 校验占一半）           │
├──────────────────┼──────────────────────────────────────────────┤
│ LangGraph         │ graph = StateGraph(State)                    │
│                   │ graph.add_node("search", search_node)        │
│                   │ graph.add_node("answer", answer_node)        │
│                   │ graph.add_conditional_edges(...)             │
│                   │ app = graph.compile()                        │
│                   │ result = app.invoke(initial_state)           │
│                   │ # 代码量: ~100 行（图结构定义占一半）          │
└──────────────────┴──────────────────────────────────────────────┘

关键洞察: 代码量不是越少越好。
  - LangChain 10 行 = 但你看不到里面做了什么
  - 自研 30 行 = 每一行你都理解
  - LangGraph 100 行 = 但获得了状态持久化 + 可视化 + 分支能力

选择的标准不是"哪个最简单"，而是"哪个提供的额外能力刚好是你需要的"。
"""


# ═══════════════════════════════════════════════════════════════════
# 5. 面试高频问题 + 回答要点
# ═══════════════════════════════════════════════════════════════════


@dataclass
class InterviewQA:
    """一个面试问题及其回答要点。"""

    question: str
    why_asked: str  # 面试官为什么问这个（考察什么能力）
    key_points: list[str]  # 回答要点
    red_flags: list[str]  # 不要说的（会扣分的回答）


INTERVIEW_QUESTIONS: list[InterviewQA] = [
    InterviewQA(
        question="LangChain 和 LlamaIndex 有什么区别？什么时候用哪个？",
        why_asked="考察你是否理解框架的定位，而不是只会调 API。",
        key_points=[
            "LangChain 是通用 LLM 应用框架，核心抽象是 Chain 和 Agent",
            "LlamaIndex 专注数据索引和检索，核心抽象是 Index 和 QueryEngine",
            "LangChain 适合: 需要多种 LLM 交互模式的项目（对话/Agent/多步推理）",
            "LlamaIndex 适合: 大量文档的检索增强生成（RAG）",
            "两者不互斥 —— LangChain 的 Agent 可以调用 LlamaIndex 的 QueryEngine",
            "我们的项目同时用了两个: ChromaDB + LlamaIndex 做索引，LangChain 做 Agent 编排",
        ],
        red_flags=[
            "不要说 'LangChain 比 LlamaIndex 好' —— 它们解决不同问题",
            "不要说 '我都用' —— 没有场景的选型等于没选",
            "不要只比较 API 好不好用 —— 面试官想听的是架构层面的理解",
        ],
    ),
    InterviewQA(
        question="什么场景下你会选择自己写而不是用框架？",
        why_asked="考察工程判断力 —— 知道什么时候该用框架，什么时候不该用。",
        key_points=[
            "三个判断标准: 复杂度、可维护性、团队熟悉度",
            "应该自研: (1) 简单任务（单轮工具调用）(2) 对依赖敏感 (3) 框架功能远超需求",
            "应该用框架: (1) 框架覆盖了你 80% 的需求 (2) 团队对框架熟悉 (3) 长期维护的项目",
            "我们 Week 5 Day 2/3 手写了 ReAct 和 Plan-Execute，理解了底层机制后才知道框架在做什么",
            "折中方案: 用 LangGraph 做编排层（避免 Lock-in），Agent 内部可以替换",
            "关键不是'用不用框架'，而是'能否在需要时离开框架'",
        ],
        red_flags=[
            "不要说 '框架都是垃圾，我自己写的更好' —— 显得不成熟",
            "不要说 '永远用框架' —— 显得没有判断力",
            "不要只说代码量 —— 要谈可维护性、团队协作、长期成本",
        ],
    ),
    InterviewQA(
        question="ReAct Agent 的循环什么时候会出问题？怎么解决？",
        why_asked="考察你是否真正用过 Agent，遇到问题能否诊断和修复。",
        key_points=[
            "三个常见问题: (1) 无限循环 (2) 工具选择错误 (3) 输出解析失败",
            "无限循环: 设置 max_steps 硬限制 + 连续重复检测（连续 3 次同样 action 就停）",
            "工具选择错误: 在 system prompt 中明确工具适用场景 + 加入 few-shot 示例",
            "输出解析失败: 要求 LLM 用固定格式输出（JSON/XML），解析失败时重试一次",
            "更深层的问题: ReAct 没有全局规划——LLM 只看到下一步，看不到三步之后",
            "我们的实践: Day 2 遇到 LLM 输出 Thought 和 Action 混在一起的问题，做了正则提取 + 失败重试",
            "根本解决方案: 对于需要规划的任务，用 Plan-Execute 替代 ReAct",
        ],
        red_flags=[
            "不要说 '没遇到过问题' —— 这等于说没真正用过",
            "不要只说加 max_steps —— 这是治标不治本",
            "不要说改 prompt 就行 —— 展示系统性的诊断思维",
        ],
    ),
    InterviewQA(
        question="CrewAI 和 AutoGen 的核心区别是什么？怎么选？",
        why_asked="考察你对多 Agent 系统的理解和框架选型能力。",
        key_points=[
            "核心区别: CrewAI = 角色扮演+固定流程，AutoGen = 对话驱动+自由协商",
            "类比: CrewAI 像手术室（每人有明确角色），AutoGen 像会议室（大家讨论达成共识）",
            "选 CrewAI: 流程固定（研究->写作->审核），角色明确，需要低代码界面",
            "选 AutoGen: 需要 Agent 间协商（追问、澄清、迭代），需要嵌套团队",
            "终止条件: CrewAI 的 Task 有天然终点，AutoGen 需要精心设计 TerminationCondition",
            "我们的实践: Day 5 分别用两者实现了研究员+写手任务，AutoGen 实际跑通了",
            "两者和 LangGraph 的关系: LangGraph 是底层编排引擎，CrewAI/AutoGen 是高层封装",
            "不是替代关系，可以在 CrewAI Agent 内部用 LangGraph 做复杂推理",
        ],
        red_flags=[
            "不要说 'X 比 Y 好' —— 要讲场景",
            "不要只比较 API 难易 —— 要谈设计哲学差异",
            "不要说 '微信用哪个我就用哪个' —— 缺乏独立判断",
        ],
    ),
    InterviewQA(
        question="如果让你从零设计一个 Agent 框架，你会怎么设计？",
        why_asked="考察系统设计能力和对现有框架的批判性思考。",
        key_points=[
            "核心模块: (1) Agent 定义层 (2) 工具管理层 (3) 编排层 (4) 可观测性层",
            "Agent 定义层: 借鉴 CrewAI 的角色定义 + AutoGen 的系统消息，两者兼有",
            "工具管理层: 借鉴 LangChain 的工具生态 + 自研的统一 Tool 接口",
            "编排层: 借鉴 LangGraph 的图编排，但提供更简单的 DSL",
            "可观测性层: 借鉴 AutoGen 的 streaming + 自研的 trace 记录（Day 6 的内容）",
            "设计原则: (1) 渐进披露（简单场景简单 API，复杂场景深入）(2) 可替换（每层可独立替换）",
            "最关键的 insight: 框架不是越强大越好，而是越'可理解'越好",
            "好的框架让用户 80% 的时间不用看源码，但要看源码时能找到明确的边界",
        ],
        red_flags=[
            "不要说 '把 X 框架的功能都做进去' —— 那是大杂烩，不是设计",
            "不要只说技术细节 —— 要先讲设计原则和 trade-off",
            "不要忽略可观测性 —— 这是生产环境最关键但最被忽视的部分",
        ],
    ),
    InterviewQA(
        question="Agent 框架的版本迭代很快，你怎么跟上变化？",
        why_asked="考察学习能力和对技术演进的判断力。",
        key_points=[
            "策略: 关注设计模式而非 API —— API 会变，模式不变",
            "例如: LangChain 的 API 从 0.1 到 0.3 大改，但 AgentExecutor 的设计思想没变",
            "ReAct 论文是 2022 年的，仍然是 Agent 设计的核心范式",
            "具体做法: (1) 锁定生产环境的框架版本 (2) 定期阅读 Changelog (3) 实验新版本但不盲追",
            "最好的学习方式: 手写一遍核心逻辑（就像 Week 5 Day 2/3 做的那样）",
            "手写过 ReAct 后，换到 LangChain/CrewAI/AutoGen 都能快速上手——因为底层逻辑一样",
            "这个 16 周计划的课程设计就体现了这个思想——先学原理再学框架",
        ],
        red_flags=[
            "不要说 '我不追版本，稳定就好' —— 显得没有成长意识",
            "不要说 '每个新版本都升级' —— 显得没有风险评估能力",
            "不要说 '看文档就行了' —— 文档常常落后于代码",
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════════
# 6. 框架选型速查表（一页纸）
# ═══════════════════════════════════════════════════════════════════

QUICK_REFERENCE = """
                    Agent 框架选型速查表
                    ====================

场景                                        推荐方案
────────────────────────────────────────  ─────────────────
"我刚学 Agent，想理解原理"                 手写 ReAct（Day 2）
"我要在一天内做个 Demo 给老板看"           LangChain AgentExecutor
"我要上线一个 FAQ 机器人"                  自研 + LangGraph 编排
"我要做代码审查（固定流程）"               Plan-Execute + LangGraph
"我要做一个能搜索+总结的研究助手"          ReAct + 工具集合
"我要让 3 个 Agent 协作写报告"             CrewAI（固定流程）/ AutoGen（灵活协商）
"我要做一个内部工具平台的 Agent 引擎"      LangGraph（编排层）+ 自研（节点实现）
"我对依赖零容忍（银行/政府项目）"          自研（只用 SDK）
"我要评估不同 Agent 方案的效果"            统一的评测框架（Day 6 内容）
"""


# ═══════════════════════════════════════════════════════════════════
# 7. 输出接口
# ═══════════════════════════════════════════════════════════════════


def print_full_guide() -> None:
    """输出完整的框架选型指南。"""
    _print_header("Agent 框架选型指南")
    _print_framework_cards()
    _print_comparison_matrix()
    _print_decision_tree()
    print(SAME_TASK_COMPARISON)
    _print_interview_qa()
    print(QUICK_REFERENCE)


def _print_header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def _print_framework_cards() -> None:
    _print_header("Part 1: 框架/范式信息卡片")
    for i, fw in enumerate(FRAMEWORKS, 1):
        print(f"[{i}] {fw.name}")
        print(f"    定位: {fw.tagline}")
        print(f"    层级: {fw.level.value}")
        print(f"    哲学: {fw.design_philosophy}")
        print(f"    擅长: {', '.join(fw.best_for[:2])}...")
        print(f"    不擅: {', '.join(fw.not_for[:2])}...")
        print(f"    曲线: {fw.learning_curve}")
        print()


def _print_comparison_matrix() -> None:
    _print_header("Part 2: 多维度对比矩阵")

    frameworks_short = ["自研", "LangChain", "ReAct", "Plan-Ex", "LangGraph", "CrewAI", "AutoGen"]

    # 表头
    header = f"{'维度':12s}"
    for fw in frameworks_short:
        header += f" | {fw:8s}"
    print(header)
    print("-" * len(header))

    for comp in COMPARISONS:
        row = f"{comp.dimension:12s}"
        for fw_key in [
            "自研",
            "LangChain",
            "ReAct",
            "Plan-Execute",
            "LangGraph",
            "CrewAI",
            "AutoGen",
        ]:
            score = comp.scores.get(fw_key, 0)
            bar = "*" * score + " " * (5 - score)
            row += f" | {bar}"
        print(row)

    print()
    print("评分: 1=最差, 5=最好, * 越多越好")
    print()

    # 详细解释
    for comp in COMPARISONS:
        print(f"--- {comp.dimension} ---")
        for name, expl in comp.explanation.items():
            print(f"  {name}: {expl}")
        print()


def _print_decision_tree() -> None:
    _print_header("Part 3: 选型决策树")

    print("你的任务是: 需要多个 Agent 协作吗？")
    print()
    print("  [是] -> 协作流程固定还是需要协商？")
    print("           [固定] -> CrewAI")
    print("           [协商] -> AutoGen")
    print()
    print("  [否] -> 流程有分支吗？")
    print("           [有] -> 需要状态持久化吗？")
    print("                    [需要] -> LangGraph")
    print("                    [不需要] -> ReAct / Plan-Execute")
    print("           [没有] -> 快速原型还是精细控制？")
    print("                      [快速原型] -> LangChain AgentExecutor")
    print("                      [精细控制] -> 自研 (SDK)")
    print()


def _print_interview_qa() -> None:
    _print_header("Part 4: 面试高频问题")
    for i, qa in enumerate(INTERVIEW_QUESTIONS, 1):
        print(f"Q{i}: {qa.question}")
        print(f"  考察: {qa.why_asked}")
        print("  要点:")
        for pt in qa.key_points:
            print(f"    - {pt}")
        print("  避雷:")
        for rf in qa.red_flags:
            print(f"    [x] {rf}")
        print()


# ═══════════════════════════════════════════════════════════════════
# 8. 命令行入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_full_guide()
