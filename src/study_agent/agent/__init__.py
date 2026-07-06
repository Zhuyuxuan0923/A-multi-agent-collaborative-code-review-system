"""Agent 模块 -- AI Agent 核心实现。

包含多种 Agent 范式：
  - PersonalQAAgent: RAG + 对话历史 + 多知识库管理
  - ReactAgent: ReAct (Reasoning + Acting) 范式
  - TracedReactAgent: 带 Trace 追踪的 ReAct Agent
  - PlanExecuteAgent: Plan-Execute (先计划后执行) 范式
  - LangGraphRouterAgent: LangGraph 条件分支工作流
  - AutogenResearchTeam: AutoGen 多 Agent 协作 (研究员+写手)
  - CrewAIResearchCrew: CrewAI 多 Agent 协作 (研究员+写手, API展示)

评测框架：
  - AgentEvaluator: Agent 评测引擎 (准确性/效率/工具正确率)
  - AgentTestCase: 测试用例数据模型
  - AgentEvalResult / AgentEvalScore: 评测结果数据模型

状态与会话管理：
  - AgentState: 会话状态 Pydantic 模型
  - SessionManager: 并发会话管理器
  - StepRecord / ToolCallRecord: 执行记录

Trace 可观测性：
  - AgentTrace: 一次完整 Agent 执行的记录
  - TraceSpan: 一个执行阶段（如一轮 ReAct 循环）
  - TraceEvent: 一个原子事件
  - TraceCollector: Trace 采集器
  - TracedReactAgent: 带 Trace 的 ReAct Agent
  - export_json / export_markdown / export_console_summary: Trace 导出函数

对抗测试 (Red Teaming)：
  - AdversarialTestCase: 对抗测试用例 (8个类别, 40+条内置用例)
  - RedTeamBot: 红队测试引擎 (自动攻击 + 5类安全检测 + 报告生成)
  - SafetyDetector: 安全检测器 (5类: system_prompt_leak / harmful_output /
    tool_abuse / loop_detected / instruction_violation)

安全守卫：
  - InputGuard: 输入安全检测 (空输入/超长/重复/越狱/注入)
  - ToolGuard: 工具调用安全检测 (参数长度/内容/重复调用)
  - LoopGuard: 循环保护 (max_rounds/prompt长度/无进展检测)
  - GuardedAgent: 安全 Agent 包装器 (组合三层守卫)
"""

from study_agent.agent.adversarial_cases import (
    CASES_BY_CATEGORY,
    CATEGORY_NAMES_ZH,
    AdversarialTestCase,
    get_all_cases,
    get_cases_by_category,
)
from study_agent.agent.adversarial_tester import (
    AdversarialTestResult,
    RedTeamBot,
    SafetyCheckResult,
    SafetyDetector,
)
from study_agent.agent.agent_eval_cases import BUILTIN_TEST_CASES
from study_agent.agent.agent_evaluator import (
    AgentEvalResult,
    AgentEvalScore,
    AgentEvaluator,
    AgentTestCase,
)
from study_agent.agent.agent_guard import (
    GuardedAgent,
    InputGuard,
    LoopGuard,
    ToolGuard,
)
from study_agent.agent.bus_orchestrator import (
    BusAwareAgent,
    BusOrchestrator,
)
from study_agent.agent.code_review_agents import (
    BaseReviewAgent,
    CodeReviewOrchestrator,
    ReporterAgent,
    ResearcherAgent,
    ResearchResult,
    ReviewerAgent,
    ReviewResult,
)
from study_agent.agent.conflict_resolver import (
    ArbiterAgent,
    Conflict,
    ConflictDetector,
    ConflictReport,
    ConflictResolver,
    ExternalVerifier,
    VotingStrategy,
)
from study_agent.agent.conversation import ConversationManager
from study_agent.agent.integration_test import (
    TEST_CASES,
    IntegrationTestReport,
    IntegrationTestRunner,
    PipelineResult,
    StageResult,
    TestCase,
)
from study_agent.agent.kb_agent import PersonalQAAgent
from study_agent.agent.knowledge_base import KnowledgeBaseManager
from study_agent.agent.langgraph_router import (
    LangGraphRouterAgent,
    RouterState,
    build_router_graph,
)
from study_agent.agent.message_bus import (
    AgentNotRegisteredError,
    MessageBus,
    MessageTimeoutError,
)
from study_agent.agent.message_protocol import (
    AgentCapability,
    AgentInfo,
    AgentMessage,
    DeliveryResult,
    MessageType,
    Priority,
    RouteType,
    RoutingRule,
    export_json_schema,
)
from study_agent.agent.plan_execute_agent import PlanExecuteAgent
from study_agent.agent.react_agent import ReactAgent
from study_agent.agent.state import (
    AgentState,
    AgentStatus,
    SessionManager,
    StepRecord,
    ToolCallRecord,
)
from study_agent.agent.trace import (
    AgentTrace,
    TraceCollector,
    TraceEvent,
    TraceSpan,
)
from study_agent.agent.trace_exporter import (
    export_console_summary,
    export_json,
    export_markdown,
)
from study_agent.agent.traced_agent import TracedReactAgent

__all__ = [
    "PersonalQAAgent",
    "ConversationManager",
    "KnowledgeBaseManager",
    "ReactAgent",
    "PlanExecuteAgent",
    "LangGraphRouterAgent",
    "RouterState",
    "build_router_graph",
    "AgentEvaluator",
    "AgentTestCase",
    "AgentEvalResult",
    "AgentEvalScore",
    "BUILTIN_TEST_CASES",
    "AgentState",
    "AgentStatus",
    "SessionManager",
    "StepRecord",
    "ToolCallRecord",
    "AgentTrace",
    "TraceSpan",
    "TraceEvent",
    "TraceCollector",
    "TracedReactAgent",
    "export_json",
    "export_markdown",
    "export_console_summary",
    "AdversarialTestCase",
    "RedTeamBot",
    "SafetyCheckResult",
    "SafetyDetector",
    "AdversarialTestResult",
    "CASES_BY_CATEGORY",
    "CATEGORY_NAMES_ZH",
    "get_all_cases",
    "get_cases_by_category",
    "InputGuard",
    "ToolGuard",
    "LoopGuard",
    "GuardedAgent",
    "AgentMessage",
    "AgentInfo",
    "AgentCapability",
    "MessageType",
    "Priority",
    "RouteType",
    "RoutingRule",
    "DeliveryResult",
    "export_json_schema",
    "MessageBus",
    "MessageTimeoutError",
    "AgentNotRegisteredError",
    "CodeReviewOrchestrator",
    "ReviewerAgent",
    "ResearcherAgent",
    "ReporterAgent",
    "ReviewResult",
    "ResearchResult",
    "BaseReviewAgent",
    "BusOrchestrator",
    "BusAwareAgent",
    "ConflictResolver",
    "ConflictDetector",
    "Conflict",
    "ConflictReport",
    "VotingStrategy",
    "ArbiterAgent",
    "ExternalVerifier",
    "IntegrationTestRunner",
    "IntegrationTestReport",
    "PipelineResult",
    "StageResult",
    "TestCase",
    "TEST_CASES",
]
