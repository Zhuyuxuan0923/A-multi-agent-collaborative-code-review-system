"""Agent 评测用例集 —— 12 条测试用例，覆盖 5 个类别 x 3 个难度级别。

测试用例设计原则：
  1. 可验证：每条用例都有明确的 expected_keywords，可以自动打分
  2. 可复现：使用 MOCK_KNOWLEDGE 中的模拟数据，不需要外部 API
  3. 有区分度：不同难度/类别的用例能区分出 Agent 能力的差异
  4. 覆盖全面：事实查询、计算、多步骤、对比、时间查询五类都覆盖

类别说明：
  factual    — 事实查询：搜一次就能回答（考察搜索+提取关键信息的能力）
  calculation — 数学计算：纯计算，不需要搜索（考察工具选择能力）
  multi_step — 多步骤：搜索 -> 提取 -> 计算/推理（考察规划与执行能力）
  comparison — 对比分析：搜索多次 -> 综合比较（考察信息整合能力）
  time       — 时间查询：获取当前时间（考察简单工具调用）
"""

from study_agent.agent.agent_evaluator import AgentTestCase

# ========================================================================
# 12 条内置测试用例
# ========================================================================
# 每条用例都基于 MOCK_KNOWLEDGE 中的数据进行验证。
# 如果你要添加新用例，确保 expected_keywords 能在 mock 搜索结果中找到。
# ========================================================================

BUILTIN_TEST_CASES: list[AgentTestCase] = [
    # ── Easy (4 条)：单一工具，明确答案 ────────────────────
    AgentTestCase(
        id="E01",
        question="React 19 是什么时候发布的？",
        difficulty="easy",
        category="factual",
        expected_keywords=["2024", "12月", "React 19"],
        expected_tools=["search"],
        min_steps=1,
        note="考察最基本的搜索+信息提取能力：从搜索结果中找到发布日期",
    ),
    AgentTestCase(
        id="E02",
        question="FastAPI 有哪些核心特性？请列举。",
        difficulty="easy",
        category="factual",
        expected_keywords=["类型提示", "自动生成", "异步", "依赖注入"],
        expected_tools=["search"],
        min_steps=1,
        note="考察从搜索结果中提取多条关键信息的能力",
    ),
    AgentTestCase(
        id="E03",
        question="计算 (15 + 27) * 3 - 50 的结果。",
        difficulty="easy",
        category="calculation",
        expected_keywords=["76"],
        expected_tools=["calculator"],
        min_steps=1,
        note="考察是否能识别出这是纯计算任务，直接调用 calculator 而不是搜索",
    ),
    AgentTestCase(
        id="E04",
        question="现在是几点？今天的日期是什么？",
        difficulty="easy",
        category="time",
        expected_keywords=["202"],  # 年份至少包含 "202"
        expected_tools=["current_time"],
        min_steps=1,
        note="考察简单工具调用：直接调用 current_time，不应该用搜索",
    ),
    # ── Medium (4 条)：两步操作或需要推理 ──────────────────
    AgentTestCase(
        id="E05",
        question="React 19 的客户端 bundle 比 React 18 减小了多少？",
        difficulty="medium",
        category="factual",
        expected_keywords=["20%", "减小"],
        expected_tools=["search"],
        min_steps=1,
        note="考察从搜索结果中定位具体数值的能力",
    ),
    AgentTestCase(
        id="E06",
        question="LangChain 包含哪些核心模块？每个模块的作用是什么？",
        difficulty="medium",
        category="factual",
        expected_keywords=["Model", "Retrieval", "Agents", "Chains"],
        expected_tools=["search"],
        min_steps=1,
        note="考察对搜索结果的结构化整理能力",
    ),
    AgentTestCase(
        id="E07",
        question="搜索 LangChain 相关信息，然后用一段话总结 LangChain 是什么。",
        difficulty="medium",
        category="multi_step",
        expected_keywords=["LangChain", "LLM", "框架"],
        expected_tools=["search", "summarize"],
        min_steps=2,
        note="考察两步操作：搜索后总结（信息提取+压缩）",
    ),
    AgentTestCase(
        id="E08",
        question="请计算 sqrt(144) + pow(3, 4) 的值。",
        difficulty="medium",
        category="calculation",
        expected_keywords=["93"],  # sqrt(144)=12, 3^4=81, 12+81=93
        expected_tools=["calculator"],
        min_steps=1,
        note="考察是否能正确传递复杂数学表达式给 calculator",
    ),
    # ── Hard (4 条)：多工具协作或需要深层推理 ──────────────
    AgentTestCase(
        id="E09",
        question="AI Agent 由哪些核心组件构成？请详细说明每个组件的作用。",
        difficulty="hard",
        category="multi_step",
        expected_keywords=["LLM", "工具", "记忆", "编排"],
        expected_tools=["search"],
        min_steps=1,
        note="考察从搜索结果中提取和组织复杂信息的能力",
    ),
    AgentTestCase(
        id="E10",
        question="比较 LangChain 和 FastAPI：它们分别适合构建什么类型的应用？有什么共同点？",
        difficulty="hard",
        category="comparison",
        expected_keywords=["LangChain", "FastAPI", "LLM", "Web", "框架"],
        expected_tools=["search"],
        min_steps=2,
        note="考察对比分析能力：需要理解两个不同工具的定位",
    ),
    AgentTestCase(
        id="E11",
        question="搜索 React 19 的新特性，重点关注 Server Components 和 Actions 机制，然后总结它们解决了什么问题。",
        difficulty="hard",
        category="multi_step",
        expected_keywords=["Server Component", "Action", "React 19"],
        expected_tools=["search", "summarize"],
        min_steps=2,
        note="考察搜索+提取+总结的完整链路",
    ),
    AgentTestCase(
        id="E12",
        question="在 AI Agent 开发中，如果要构建一个前端界面，React 19 的哪些特性最有帮助？请结合 AI Agent 的架构来说明。",
        difficulty="hard",
        category="comparison",
        expected_keywords=["React 19", "Agent", "前端"],
        expected_tools=["search"],
        min_steps=2,
        note="考察跨领域知识整合：需要同时理解 AI Agent 架构和 React 19 特性",
    ),
]


def get_cases_by_difficulty(difficulty: str) -> list[AgentTestCase]:
    """按难度筛选测试用例。"""
    return [c for c in BUILTIN_TEST_CASES if c.difficulty == difficulty]


def get_cases_by_category(category: str) -> list[AgentTestCase]:
    """按类别筛选测试用例。"""
    return [c for c in BUILTIN_TEST_CASES if c.category == category]


def get_case_summary() -> str:
    """生成测试用例集的简要概览。"""
    by_diff: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for c in BUILTIN_TEST_CASES:
        by_diff[c.difficulty] = by_diff.get(c.difficulty, 0) + 1
        by_cat[c.category] = by_cat.get(c.category, 0) + 1

    lines = [
        f"测试用例总数: {len(BUILTIN_TEST_CASES)}",
        "",
        "按难度分布:",
    ]
    for diff in ["easy", "medium", "hard"]:
        count = by_diff.get(diff, 0)
        bar = "#" * count
        lines.append(f"  {diff:6s}: {bar} ({count})")

    lines += ["", "按类别分布:"]
    for cat, count in sorted(by_cat.items()):
        bar = "#" * count
        lines.append(f"  {cat:12s}: {bar} ({count})")

    return "\n".join(lines)
