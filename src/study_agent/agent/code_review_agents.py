"""
代码审查多 Agent 系统 -- 三个角色的 System Prompt 与 Agent 实现。

三个角色:
  1. Reviewer (审查员)   -- 找代码问题: 安全/性能/逻辑/可维护性
  2. Researcher (研究员) -- 查最佳实践: 该技术的推荐做法和常见陷阱
  3. Reporter (报告员)   -- 汇总输出: 整合审查结果, 生成结构化审查报告

协作流程 (层级模式):
  Orchestrator 收到代码
    -> 并行发送给 Reviewer + Researcher
    -> 收集两方输出
    -> 发送给 Reporter 汇总
    -> 输出 Markdown 审查报告

运行方式: python -m src.study_agent.agent.code_review_demo
"""

import json
import re
from dataclasses import dataclass, field

from study_agent.llm.client import LLMClient

# ============================================================
# System Prompts -- 今天的核心交付物
# ============================================================

REVIEWER_SYSTEM_PROMPT = """\
你是一位资深代码审查专家, 有 15 年软件工程经验。你的任务是对给定的代码进行专业审查。

## 你的审查维度 (按重要性排序)

1. **安全漏洞** (Critical): SQL 注入、XSS、命令注入、硬编码密钥/密码、路径遍历、不安全的反序列化、权限绕过
2. **逻辑错误** (Critical): 边界条件遗漏、空值未处理、类型错误、资源未释放、死锁风险
3. **性能问题** (Warning): N+1 查询、循环内重复计算、不必要的内存分配、阻塞 I/O
4. **可维护性** (Suggestion): 命名不规范、函数过长(>50行)、重复代码、magic number、注释缺失

## 工作原则

- 只审查给定的代码, 不要假设代码之外的环境或依赖
- 对于不确定的语言特性, 标记为 Suggestion 而非 Critical
- 不要吹毛求疵——如果代码整体健康, 直接说"整体质量良好"即可
- 每条问题必须给出具体行号和修复建议, 不要泛泛而谈

## 输出格式 (严格 JSON)

你必须输出一个 JSON 对象, 格式如下:

```json
{
  "summary": "一句话总结代码质量",
  "score": 8,
  "issues": [
    {
      "severity": "Critical",
      "category": "安全漏洞",
      "line": 15,
      "title": "SQL 注入风险",
      "description": "用户输入直接拼接到 SQL 查询中, 攻击者可构造恶意输入",
      "suggestion": "使用参数化查询: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
    }
  ]
}
```

severity 必须是: Critical / Warning / Suggestion
category 必须是: 安全漏洞 / 逻辑错误 / 性能问题 / 可维护性
score 是 0-10 的整数, 10 表示代码完美

## 审查示例

**不好 (太模糊)**:
"第10行有问题, 代码不安全"

**好 (具体可操作)**:
"第10行: [Critical][安全漏洞] 用户输入 `user_id` 直接拼接到 SQL 语句中。
建议: 使用参数化查询 `cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))`"

请开始审查。只输出 JSON, 不要有其他文字。
"""

RESEARCHER_SYSTEM_PROMPT = """\
你是一位技术研究员, 专门为代码审查提供技术背景和最佳实践参考。

## 你的任务

针对给定代码所使用的技术和框架, 提供:

1. **最佳实践**: 该技术在官方文档或社区公认的推荐做法
2. **常见陷阱**: 开发者在使用该技术时最容易犯的错误
3. **替代方案**: 如果有更好的库/模式/架构, 指出来
4. **参考资料**: 相关的官方文档链接或公认的权威资源

## 工作原则

- 你提供的每条建议都应该是"可验证的"——来自官方文档或广泛接受的社区实践, 而非个人偏好
- 不要说"这取决于情况"——给出具体建议, 如果确实有多选, 列出优劣
- 聚焦于代码中实际用到的技术, 不要发散到不相关的领域

## 输出格式 (严格 JSON)

你必须输出一个 JSON 对象, 格式如下:

```json
{
  "technologies": ["Python", "SQLite", "FastAPI"],
  "best_practices": [
    {
      "title": "使用参数化查询防止 SQL 注入",
      "description": "永远不要用字符串拼接构造 SQL。Python 的 sqlite3 模块支持 ? 占位符。",
      "reference": "https://docs.python.org/3/library/sqlite3.html#sql-parameters"
    }
  ],
  "common_pitfalls": [
    {
      "title": "忘记关闭数据库连接",
      "description": "不使用 context manager 时容易导致连接泄漏。",
      "example": "推荐: with sqlite3.connect(db) as conn: ..."
    }
  ],
  "recommendations": ["使用 SQLAlchemy ORM 代替原生 SQL 以获得更好的安全性"],
  "references": ["https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"]
}
```

请开始研究。只输出 JSON, 不要有其他文字。
"""

REPORTER_SYSTEM_PROMPT = """\
你是一位技术报告撰写专家, 负责将代码审查结果和技术研究成果整合成一份专业的审查报告。

## 你的输入

你会收到三部分信息:
1. **原始代码** — 被审查的代码片段
2. **审查结果** — Reviewer (审查员) 找出的问题列表 (JSON 格式)
3. **研究结果** — Researcher (研究员) 提供的最佳实践和参考资料

## 报告结构 (严格遵守)

你的报告必须包含以下章节:

### 1. 审查概览
- 总体评分 (继承 Reviewer 的 score)
- 问题统计 (Critical X 个, Warning X 个, Suggestion X 个)
- 一句话总结

### 2. 关键问题 (Critical)
- 逐个列出所有 Critical 级别的问题
- 每个问题包含: 位置、描述、风险说明、修复方案

### 3. 改进建议 (Warning + Suggestion)
- 按类别分组 (性能问题 / 可维护性)
- 每个建议包含: 位置、现状、改进方向

### 4. 最佳实践参考
- 从 Researcher 结果中提取最相关的 3-5 条
- 每条标注来源

### 5. 总结与行动计划
- 优先修复 Critical 问题
- 建议的改进顺序
- 参考资料链接

## 写作风格

- 专业但不晦涩 —— 让中级开发者能看懂
- 客观 —— 区分"确定有问题"和"建议改进"
- 可操作 —— 每条建议都回答"我应该怎么改?"
- 简洁 —— 每个章节 3-8 句话, 不要写成论文

请生成完整的 Markdown 格式审查报告。
"""


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ReviewResult:
    """Reviewer 的审查结果。"""

    summary: str
    score: int
    issues: list[dict] = field(default_factory=list)
    raw_json: str = ""


@dataclass
class ResearchResult:
    """Researcher 的研究结果。"""

    technologies: list[str] = field(default_factory=list)
    best_practices: list[dict] = field(default_factory=list)
    common_pitfalls: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    raw_json: str = ""


# ============================================================
# Agent 实现
# ============================================================


class BaseReviewAgent:
    """代码审查 Agent 的基类。

    每个子 Agent 有自己的 system prompt 和输出解析逻辑。
    不依赖 MessageBus -- 这是一个纯函数式 Agent, 输入任务, 输出结果。
    MessageBus 集成由 Orchestrator 负责 (Day 4 的主题)。
    """

    def __init__(self, role_name: str, system_prompt: str, llm: LLMClient):
        self.role_name = role_name
        self.system_prompt = system_prompt
        self.llm = llm

    def _call_llm(self, user_message: str) -> str:
        """调用 LLM, 返回原始文本回复。"""
        return self.llm.chat(user_message, system=self.system_prompt)

    def run(self, user_message: str) -> str:
        """执行 Agent, 返回原始 LLM 输出。子类可以覆写以添加解析逻辑。"""
        print(f"  [{self.role_name}] 正在调用 LLM...")
        result = self._call_llm(user_message)
        print(f"  [{self.role_name}] 完成 ({len(result)} 字符)")
        return result


class ReviewerAgent(BaseReviewAgent):
    """审查员 -- 找代码问题。

    输入: 代码 + 语言
    输出: ReviewResult (结构化问题列表)
    """

    def __init__(self, llm: LLMClient):
        super().__init__("Reviewer", REVIEWER_SYSTEM_PROMPT, llm)

    def run(self, code: str, language: str = "unknown") -> ReviewResult:
        user_msg = f"## 待审查代码\n\n语言: {language}\n\n```{language}\n{code}\n```\n\n请审查。"
        raw = self._call_llm(user_msg)
        return self._parse(raw)

    def _parse(self, raw: str) -> ReviewResult:
        """从 LLM 输出中提取 JSON 审查结果。

        当 JSON 解析失败时，用正则尝试提取 score 和 issues，
        避免因 LLM 输出格式波动而丢失所有审查数据。
        """
        try:
            # 尝试提取 JSON 块
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            score = data.get("score", -1)
            # 防止 LLM 返回非整数 score
            if not isinstance(score, int):
                score = (
                    int(score)
                    if isinstance(score, (float, str))
                    and str(score).replace(".", "").replace("-", "").isdigit()
                    else -1
                )
            return ReviewResult(
                summary=data.get("summary", ""),
                score=score,
                issues=data.get("issues", []),
                raw_json=raw,
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            pass  # 进入正则回退

        # 正则回退：从非标准格式中抢救 score
        score_match = re.search(r'"score"\s*:\s*(-?\d+)', raw)
        score = int(score_match.group(1)) if score_match else -1

        # 正则回退：尝试提取 issues 数组
        issues: list[dict] = []
        try:
            issues_match = re.search(r'"issues"\s*:\s*(\[.*?\])', raw, re.DOTALL)
            if issues_match:
                issues = json.loads(issues_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass

        if score == -1 and not issues:
            # 完全无法解析：返回 sentinel 值
            return ReviewResult(
                summary="解析审查结果失败",
                score=-1,
                issues=[
                    {
                        "severity": "Warning",
                        "category": "系统",
                        "title": "解析错误",
                        "description": raw[:200],
                    }
                ],
                raw_json=raw,
            )

        return ReviewResult(
            summary="解析审查结果（部分数据可能丢失）",
            score=score,
            issues=issues,
            raw_json=raw,
        )


class ResearcherAgent(BaseReviewAgent):
    """研究员 -- 查最佳实践和常见陷阱。

    输入: 代码 + 语言/框架
    输出: ResearchResult (最佳实践 + 参考)
    """

    def __init__(self, llm: LLMClient):
        super().__init__("Researcher", RESEARCHER_SYSTEM_PROMPT, llm)

    def run(self, code: str, language: str = "unknown") -> ResearchResult:
        user_msg = (
            f"## 待研究代码\n\n语言: {language}\n\n```{language}\n{code}\n```\n\n"
            f"请分析这段代码使用的技术, 提供最佳实践和常见陷阱。"
        )
        raw = self._call_llm(user_msg)
        return self._parse(raw)

    def _parse(self, raw: str) -> ResearchResult:
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return ResearchResult(
                technologies=data.get("technologies", []),
                best_practices=data.get("best_practices", []),
                common_pitfalls=data.get("common_pitfalls", []),
                recommendations=data.get("recommendations", []),
                references=data.get("references", []),
                raw_json=raw,
            )
        except (json.JSONDecodeError, IndexError):
            return ResearchResult(raw_json=raw)


class ReporterAgent(BaseReviewAgent):
    """报告员 -- 整合审查结果和研究成果, 生成 Markdown 报告。

    输入: 原始代码 + ReviewResult + ResearchResult
    输出: Markdown 格式的审查报告
    """

    def __init__(self, llm: LLMClient):
        super().__init__("Reporter", REPORTER_SYSTEM_PROMPT, llm)

    def run(self, code: str, language: str, review: ReviewResult, research: ResearchResult) -> str:
        user_msg = f"""\
## 原始代码

语言: {language}
```{language}
{code}
```

## 审查结果 (Reviewer 输出)

```json
{json.dumps({"summary": review.summary, "score": review.score, "issues": review.issues}, ensure_ascii=False, indent=2)}
```

## 研究结果 (Researcher 输出)

```json
{json.dumps({"technologies": research.technologies, "best_practices": research.best_practices, "common_pitfalls": research.common_pitfalls, "recommendations": research.recommendations, "references": research.references}, ensure_ascii=False, indent=2)}
```

请生成完整的 Markdown 审查报告。
"""
        return self._call_llm(user_msg)


# ============================================================
# Orchestrator -- 协调三个 Agent 的工作流 (层级模式)
# ============================================================


class CodeReviewOrchestrator:
    """代码审查编排器。

    工作流 (层级模式):
      1. 并行调用 Reviewer + Researcher
      2. 收集两方结果
      3. 调用 Reporter 汇总
      4. 返回最终 Markdown 报告

    注意: Day 4 会把这个替换为基于 MessageBus 的编排器。
    """

    def __init__(self, llm: LLMClient):
        self.reviewer = ReviewerAgent(llm)
        self.researcher = ResearcherAgent(llm)
        self.reporter = ReporterAgent(llm)

    def review(self, code: str, language: str = "unknown", verbose: bool = True) -> str:
        """执行完整的代码审查流程。

        参数:
            code: 源代码文本
            language: 编程语言 (用于语法高亮)
            verbose: 是否打印进度

        返回:
            Markdown 格式的审查报告
        """
        if verbose:
            print("=" * 60)
            print("代码审查编排器启动")
            print(f"语言: {language}, 代码: {len(code)} 字符, {code.count(chr(10)) + 1} 行")
            print("=" * 60)
            print()

        # 阶段 1: 并行审查 + 研究
        if verbose:
            print("阶段 1/2: 并行调用 Reviewer + Researcher")
            print("-" * 30)

        review_result = self.reviewer.run(code, language)
        research_result = self.researcher.run(code, language)

        if verbose:
            print()
            print(
                f"  Reviewer: 评分 {review_result.score}/10, 发现 {len(review_result.issues)} 个问题"
            )
            severity_counts = {}
            for issue in review_result.issues:
                sev = issue.get("severity", "Unknown")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            for sev, cnt in severity_counts.items():
                print(f"    {sev}: {cnt}")
            print(
                f"  Researcher: 识别技术 {research_result.technologies}, {len(research_result.best_practices)} 条最佳实践"
            )

        # 阶段 2: Reporter 汇总
        if verbose:
            print()
            print("阶段 2/2: Reporter 汇总生成报告")
            print("-" * 30)

        report = self.reporter.run(code, language, review_result, research_result)

        if verbose:
            print(f"  报告生成完成 ({len(report)} 字符)")

        return report

    def review_with_intermediates(
        self, code: str, language: str = "unknown"
    ) -> tuple[ReviewResult, ResearchResult, str]:
        """审查并返回中间结果 (用于调试和学习)。"""
        review = self.reviewer.run(code, language)
        research = self.researcher.run(code, language)
        report = self.reporter.run(code, language, review, research)
        return review, research, report
