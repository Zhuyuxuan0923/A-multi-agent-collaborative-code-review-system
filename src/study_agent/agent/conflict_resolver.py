"""
冲突检测与解决系统 -- Week 7 Day 5 核心交付物。

当多个 Agent 对同一段代码产生不同意见时，如何发现冲突并解决？

背景: 在 Day 3/4 的代码审查系统中，Reviewer 找代码问题，Researcher 查最佳实践。
两个 Agent 视角不同，自然会产生分歧:
  - Reviewer 说"第10行有 SQL 注入" -> 标为 Critical
  - Researcher 说"该 ORM 框架已自动转义参数" -> 认为这里安全
  谁对？这就是冲突解决要回答的问题。

三种策略:
  1. 投票 (Voting): 运行多个 Reviewer 实例，对每个问题投票，少数服从多数
  2. 层级裁决 (Hierarchy): 引入 Arbiter (仲裁员)，职位高于 Reviewer/Researcher，做最终决定
  3. 外部验证 (External): 用 LLM 的知识库做"事实核查"，验证争议中的技术主张

冲突类型:
  - factual: 两个 Agent 对同一事实有相反的判断
  - severity: 对问题严重程度的判断不同
  - omission: 一个 Agent 发现了问题，另一个完全没提

运行方式: python -m src.study_agent.agent.conflict_resolver_demo
"""

import json
import uuid
from dataclasses import dataclass, field

from .code_review_agents import (
    LLMClient,
    ResearcherAgent,
    ResearchResult,
    ReviewerAgent,
    ReviewResult,
)

# ============================================================
# 1. 数据结构 -- 冲突长什么样？
# ============================================================


@dataclass
class Conflict:
    """表示两个 Agent 之间的一个分歧。

    生活类比: 两个医生看了同一张 X 光片，一个说"骨折了"，一个说"只是骨裂"。
    这个数据结构记录的就是这种分歧——谁的判断？基于什么依据？如何解决？
    """

    conflict_id: str
    topic: str  # 争议主题，如 "SQL 注入风险评估"
    position_a: str  # Agent A (Reviewer) 的立场
    position_b: str  # Agent B (Researcher) 的立场
    evidence_a: str  # Agent A 的依据
    evidence_b: str  # Agent B 的依据
    conflict_type: str = "factual"  # factual | severity | omission
    resolved: bool = False
    resolution: str = ""  # 解决结论
    resolution_strategy: str = ""  # voting | hierarchy | external
    winner: str = ""  # a | b | compromise


@dataclass
class ConflictReport:
    """冲突检测与解决的完整报告。"""

    has_conflicts: bool
    conflicts: list[Conflict] = field(default_factory=list)
    agreements: list[dict] = field(default_factory=list)
    resolved_count: int = 0
    unresolved_count: int = 0


# ============================================================
# 2. 冲突检测器 -- 怎么发现 Agent 之间在"吵架"？
# ============================================================

CONFLICT_DETECTOR_PROMPT = """\
你是一个冲突检测专家。你的任务是对比两个 Agent 的输出，找出它们之间的分歧。

你会收到:
1. **原始代码** — 被审查的代码
2. **Reviewer 的输出** — 代码审查员找出的问题 (JSON)
3. **Researcher 的输出** — 技术研究员提供的最佳实践 (JSON)

## 你需要找出的冲突类型

### 1. 事实冲突 (factual)
两个 Agent 对同一个事实有相反的判断。
- 例: Reviewer 说"第10行有 SQL 注入风险", 而 Researcher 列出的最佳实践中暗示该框架会自动转义
- 例: Reviewer 说"这个函数有性能问题", 但 Researcher 的最佳实践认为这是标准做法

### 2. 严重度冲突 (severity)
两个 Agent 对问题的严重程度判断不同。
- 例: Reviewer 标为 Critical, 但从上下文和 Researcher 的分析看实际上只是 Warning 级别

### 3. 遗漏 (omission)
一个 Agent 发现了重要问题, 另一个完全没提到。
- 注意: 这不一定是"冲突", 但要标记出来供人工判断

## 输出格式

你必须输出一个 JSON 对象:
```json
{
  "has_conflicts": true,
  "conflicts": [
    {
      "topic": "SQL 注入风险评估分歧",
      "position_a": "Reviewer 认为第10行存在 SQL 注入风险，标记为 Critical",
      "position_b": "Researcher 的最佳实践中未特别提及此具体风险，而是笼统建议使用 ORM",
      "evidence_a": "Reviewer: 第10行使用字符串拼接构造 SQL 查询",
      "evidence_b": "Researcher: 建议使用参数化查询，但未直接评论第10行",
      "conflict_type": "omission"
    }
  ],
  "agreements": [
    {"topic": "错误处理", "both_say": "双方都认为缺少异常处理"}
  ]
}
```

只输出 JSON, 不要有其他文字。
"""


class ConflictDetector:
    """用 LLM 检测 Reviewer 和 Researcher 输出之间的冲突。

    为什么不直接写规则来检测？
    - Reviewer 输出的是"在第10行有 SQL 注入"
    - Researcher 输出的是"使用参数化查询防止 SQL 注入"
    - 这两句话在说同一件事，但措辞完全不同
    - 硬编码的字符串匹配认不出它们是相关的
    - 只有 LLM 能判断它们是在"同意"还是在"各说各话"

    工作流程:
      输入 -> 构造 prompt (代码 + 两个 Agent 的输出) -> LLM 分析 -> 解析冲突列表
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def detect(
        self,
        code: str,
        language: str,
        review: ReviewResult,
        research: ResearchResult,
    ) -> ConflictReport:
        """检测 Reviewer 和 Researcher 之间的冲突。

        参数:
            code: 原始代码
            language: 编程语言
            review: Reviewer 的审查结果
            research: Researcher 的研究结果

        返回:
            ConflictReport 包含所有检测到的冲突和一致点
        """
        review_json = json.dumps(
            {"summary": review.summary, "score": review.score, "issues": review.issues},
            ensure_ascii=False,
            indent=2,
        )
        research_json = json.dumps(
            {
                "technologies": research.technologies,
                "best_practices": research.best_practices,
                "common_pitfalls": research.common_pitfalls,
                "recommendations": research.recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )

        user_msg = (
            f"## 原始代码\n\n```{language}\n{code}\n```\n\n"
            f"## Reviewer 输出\n\n```json\n{review_json}\n```\n\n"
            f"## Researcher 输出\n\n```json\n{research_json}\n```"
        )
        raw = self.llm.chat(user_msg, system=CONFLICT_DETECTOR_PROMPT)
        return self._parse(raw)

    def _parse(self, raw: str) -> ConflictReport:
        """从 LLM 的 JSON 输出中提取 ConflictReport。"""
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())

            conflicts = []
            for c in data.get("conflicts", []):
                conflicts.append(
                    Conflict(
                        conflict_id=uuid.uuid4().hex[:8],
                        topic=c.get("topic", ""),
                        position_a=c.get("position_a", ""),
                        position_b=c.get("position_b", ""),
                        evidence_a=c.get("evidence_a", ""),
                        evidence_b=c.get("evidence_b", ""),
                        conflict_type=c.get("conflict_type", "factual"),
                    )
                )

            return ConflictReport(
                has_conflicts=data.get("has_conflicts", False),
                conflicts=conflicts,
                agreements=data.get("agreements", []),
            )
        except (json.JSONDecodeError, IndexError):
            return ConflictReport(has_conflicts=False)


# ============================================================
# 3. 投票策略 -- 多个 Reviewer，少数服从多数
# ============================================================


class VotingStrategy:
    """运行多个 Reviewer 实例，对争议问题投票。

    核心思想:
      如果一个 Reviewer 可能看走眼，那就找三个。
      对每个发现的问题，统计有多少个 Reviewer 同意其存在。
      超过半数就算"通过"。

    为什么需要投票？
      LLM 的输出有随机性（temperature > 0），同一个 prompt 可能得到略有不同的结果。
      某个 Reviewer 可能"漏看"一个问题，或者"过度敏感"误报一个不是问题的问题。
      多个 Reviewer 投票可以平滑这种随机性——就像"三个臭皮匠顶个诸葛亮"。

    生活类比: 歌唱比赛的评委打分，去掉一个最高分一个最低分，取平均。
    这里的"投票"是: 对每个问题，有几个评委认为它真的存在？
    """

    def __init__(self, llm: LLMClient, num_reviewers: int = 3):
        self.llm = llm
        self.num_reviewers = num_reviewers
        # 同一个 system prompt，但 LLM 每次的输出有自然随机性
        self.reviewers = [ReviewerAgent(llm) for _ in range(num_reviewers)]

    def vote(self, code: str, language: str, verbose: bool = True) -> dict:
        """运行多个 Reviewer，对每个发现的问题进行投票。

        返回结构:
            {
                "results": [ReviewResult, ...],    # 所有原始结果
                "voting_result": {
                    "issue_key": {
                        "topic": str,
                        "severity": str,
                        "votes_for": int,
                        "total": int,
                        "passed": bool,
                        "reviewers_agreeing": [...],
                    }
                },
                "final_issues": [...],              # 投票通过的 issue 列表
                "final_score": float,               # 平均评分
            }
        """
        if verbose:
            print(f"  [投票] 启动 {self.num_reviewers} 个 Reviewer 并行审查...")

        # 运行所有 Reviewer
        results: list[ReviewResult] = []
        for i, reviewer in enumerate(self.reviewers, 1):
            result = reviewer.run(code, language)
            results.append(result)
            if verbose:
                print(f"  [投票] Reviewer {i}: 评分 {result.score}/10, {len(result.issues)} 个问题")

        # 合并所有 issues，按标题+类别去重
        all_issues = self._merge_issues(results)

        # 对每个 issue 投票
        threshold = self.num_reviewers / 2 + 0.5  # 过半数的阈值
        voting_result = {}
        final_issues = []

        for issue_key, votes in all_issues.items():
            vote_count = len(votes)
            passed = vote_count >= threshold

            representative = votes[0]  # 取第一个作为代表
            voting_result[issue_key] = {
                "topic": representative.get("title", issue_key),
                "severity": representative.get("severity", ""),
                "votes_for": vote_count,
                "total": self.num_reviewers,
                "passed": passed,
                "reviewers_agreeing": [f"Reviewer-{v.get('_reviewer_id', '?')}" for v in votes],
            }

            if passed:
                final_issues.append(representative)

        avg_score = sum(r.score for r in results) / len(results)

        summary = (
            f"投票策略: {self.num_reviewers} 个 Reviewer, "
            f"{len(final_issues)}/{len(all_issues)} 个问题通过 (过半数={threshold:.0f}票)"
        )
        if verbose:
            print(f"  [投票] {summary}")
            print(f"  [投票] 平均评分: {avg_score:.1f}/10")

        return {
            "results": results,
            "voting_result": voting_result,
            "final_issues": final_issues,
            "final_score": round(avg_score, 1),
        }

    def _merge_issues(self, results: list[ReviewResult]) -> dict:
        """合并多个 Reviewer 的 issues，按标题去重。

        两个 issue 算"同一个"的标准: 类别 + 标题的前 80 个字符相同。
        这是近似匹配——用 LLM 判断会更准但太慢，实际项目中可以先粗筛再用 LLM 精排。

        返回: {issue_key: [issue_dict_with_reviewer_id, ...]}
        """
        merged: dict = {}

        for i, result in enumerate(results, 1):
            for issue in result.issues:
                title = issue.get("title", "")
                category = issue.get("category", "")
                key = f"{category}|{title}"[:80]

                issue_copy = dict(issue)
                issue_copy["_reviewer_id"] = i

                if key not in merged:
                    merged[key] = []
                merged[key].append(issue_copy)

        return merged


# ============================================================
# 4. 层级裁决 -- Arbiter (仲裁员) 做最终决定
# ============================================================

ARBITER_SYSTEM_PROMPT = """\
你是一个高级技术仲裁员 (Senior Technical Arbiter)，在代码审查团队中职位最高。

## 你的职责

当 Reviewer (代码审查员) 和 Researcher (技术研究员) 对代码问题有分歧时，你来做最终裁决。

你的判断就是终局的——就像法官在法庭上做出的判决。

## 裁决原则 (按优先级排序)

1. **安全性优先**: 当安全和其他维度冲突时，安全优先。宁可误报 (false positive)，不可漏报 (false negative)。
2. **可验证性**: 如果某个 Agent 的论据有具体代码行号/文档引用支撑，比泛泛而谈更可信。
3. **实际影响**: 考虑问题的实际影响范围——理论上的风险如果触发条件极苛刻，可以降级处理。
4. **业界共识**: 如果某个做法在业界被广泛接受（如 OWASP 标准、语言官方风格指南），优先采信。

## 输出格式

对于每个争议，输出你的裁决:

```json
{
  "verdicts": [
    {
      "conflict_topic": "SQL 注入风险争议",
      "decision": "uphold_reviewer",
      "reasoning": "Reviewer 的担心是合理的。虽然参数化查询是最佳实践，但这段代码在第10行使用了字符串拼接构造 SQL，不经过任何 ORM 或转义。Researcher 的最佳实践建议是正确的，但不适用于此代码的实际情况。",
      "final_severity": "Critical",
      "action": "将第10行的字符串拼接替换为参数化查询: cursor.execute('SELECT ... WHERE id = ?', (user_id,))"
    }
  ]
}
```

decision 必须是:
- "uphold_reviewer" -- 支持 Reviewer 的判断
- "uphold_researcher" -- 支持 Researcher 的判断
- "compromise" -- 双方各有道理, 取折中方案

请开始仲裁。只输出 JSON。
"""


class ArbiterAgent:
    """仲裁员 -- 在 Reviewer 和 Researcher 有分歧时做最终裁决。

    为什么它是"层级"的？
    - Reviewer 和 Researcher 是平级的 (peer)
    - Arbiter 是它们的"上级"，有最终决定权
    - 就像公司里技术总监拍板——两个工程师争论不休时，总监一锤定音

    优缺点:
      优点: 决策快，责任明确，不会陷入无限讨论
      缺点: Arbiter 也可能犯错，单点决策有风险——如果 Arbiter 判断失误，整个审查结果就歪了
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def arbitrate(
        self,
        code: str,
        language: str,
        conflicts: list[Conflict],
        review: ReviewResult,
        research: ResearchResult,
        verbose: bool = True,
    ) -> list[dict]:
        """对每个冲突做出裁决。

        参数:
            code: 原始代码
            language: 编程语言
            conflicts: ConflictDetector 检测到的冲突列表
            review: Reviewer 的审查结果
            research: Researcher 的研究结果
            verbose: 是否打印过程

        返回:
            verdicts 列表，每个包含 decision, reasoning, action
        """
        if not conflicts:
            if verbose:
                print("  [仲裁] 没有冲突需要裁决")
            return []

        if verbose:
            print(f"  [仲裁] Arbiter 开始裁决 {len(conflicts)} 个冲突...")

        # 构造冲突描述文本
        parts = []
        for i, c in enumerate(conflicts, 1):
            parts.append(
                f"### 冲突 {i}: {c.topic}\n"
                f"- 类型: {c.conflict_type}\n"
                f"- Reviewer 立场: {c.position_a}\n"
                f"- Researcher 立场: {c.position_b}\n"
                f"- Reviewer 依据: {c.evidence_a}\n"
                f"- Researcher 依据: {c.evidence_b}\n"
            )

        review_json = json.dumps(
            {"summary": review.summary, "score": review.score, "issues": review.issues},
            ensure_ascii=False,
            indent=2,
        )
        research_json = json.dumps(
            {
                "best_practices": research.best_practices,
                "common_pitfalls": research.common_pitfalls,
                "recommendations": research.recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )

        user_msg = (
            f"## 原始代码\n\n```{language}\n{code}\n```\n\n"
            f"## 争议列表\n\n{chr(10).join(parts)}\n\n"
            f"## Reviewer 完整输出\n\n```json\n{review_json}\n```\n\n"
            f"## Researcher 完整输出\n\n```json\n{research_json}\n```"
        )
        raw = self.llm.chat(user_msg, system=ARBITER_SYSTEM_PROMPT)
        verdicts = self._parse(raw)

        if verbose:
            for v in verdicts:
                decision_label = {
                    "uphold_reviewer": "支持 Reviewer",
                    "uphold_researcher": "支持 Researcher",
                    "compromise": "折中",
                }.get(v.get("decision", ""), v.get("decision", "?"))
                print(f"  [仲裁] {v.get('conflict_topic', '?')}: {decision_label}")

        return verdicts

    def _parse(self, raw: str) -> list[dict]:
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return data.get("verdicts", [])
        except (json.JSONDecodeError, IndexError):
            return []

    def resolve_conflicts(
        self,
        review_data: dict,
        conflicts: list[Conflict],
        verdicts: list[dict],
    ) -> dict:
        """根据仲裁结果，修正 Reviewer 的输出。

        具体操作:
        - "uphold_reviewer" -> issue 保持不变，标记"仲裁确认"
        - "uphold_researcher" -> issue 降级或移除，标记"仲裁推翻"
        - "compromise" -> issue 保留但降级严重度，标记"仲裁调整"
        """
        resolved = dict(review_data)
        issues = list(resolved.get("issues", []))

        for i, conflict in enumerate(conflicts):
            if i >= len(verdicts):
                break
            verdict = verdicts[i]
            decision = verdict.get("decision", "")
            reason = verdict.get("reasoning", "")

            # 在 issues 列表中找对应的 issue
            matched = False
            for j, issue in enumerate(issues):
                title = issue.get("title", "")
                desc = issue.get("description", "")
                if conflict.topic in title or conflict.topic in desc:
                    matched = True
                    if decision == "uphold_researcher":
                        # Researcher 赢了 -> 降级或移除
                        new_sev = verdict.get("final_severity")
                        if new_sev:
                            issues[j]["severity"] = new_sev
                            issues[j]["arbiter_note"] = reason
                        else:
                            issues[j]["removed_by_arbiter"] = True
                            issues[j]["arbiter_note"] = reason
                    elif decision == "compromise":
                        new_sev = verdict.get("final_severity")
                        if new_sev:
                            issues[j]["severity"] = new_sev
                        issues[j]["arbiter_note"] = reason
                    else:  # uphold_reviewer
                        issues[j]["arbiter_confirmed"] = True
                        issues[j]["arbiter_note"] = reason
                    break

            # 遗漏类冲突: 没有对应的 issue, 但 Researcher 发现了 Reviewer 遗漏的点
            if (
                not matched
                and decision == "uphold_researcher"
                and conflict.conflict_type == "omission"
            ):
                issues.append(
                    {
                        "severity": verdict.get("final_severity", "Warning"),
                        "category": "遗漏补充 (仲裁员发现)",
                        "title": conflict.topic,
                        "description": reason,
                        "from_arbiter": True,
                    }
                )

        # 过滤被仲裁移除的 issues
        resolved["issues"] = [i for i in issues if not i.get("removed_by_arbiter")]
        resolved["arbitration_applied"] = True
        return resolved


# ============================================================
# 5. 外部验证策略 -- 用 LLM 的知识库做"事实核查"
# ============================================================

EXTERNAL_VERIFIER_PROMPT = """\
你是一个技术事实核查员 (Technical Fact Checker)。你的任务是验证技术争议中的事实主张。

## 你的验证方式

利用你的训练数据中的知识 (官方文档、OWASP 标准、CVE 数据库、语言规范、著名博客) 来判断每个争议中哪个主张更符合事实。

## 验证标准

- **Confirmed**: 该主张与业界共识/官方文档/语言规范完全一致
- **Partially Correct**: 主张大体正确但有细微偏差或上下文限制
- **Unverified**: 无法从已知知识中确认 (需要查阅具体文档)
- **Incorrect**: 与已知事实或共识矛盾

## 输出格式

```json
{
  "verifications": [
    {
      "conflict_topic": "SQL 注入风险争议",
      "claim_a_verification": "Confirmed",
      "claim_b_verification": "Partially Correct",
      "explanation": "根据 OWASP Top 10 (2021版), 使用字符串拼接构造 SQL 查询确实是注入漏洞 (A03:2021-Injection)。Reviewer 的判断符合安全最佳实践。Researcher 关于'应使用 ORM'的建议本身是正确的，但 ORM 也不能完全防止注入——原生 SQL 仍需要参数化。",
      "winner": "a",
      "confidence": "high"
    }
  ]
}
```

winner 必须是: "a" (支持 Reviewer) / "b" (支持 Researcher) / "unclear" (双方都有道理, 无法判断)
confidence 必须是: "high" / "medium" / "low"

请开始核查。只输出 JSON。
"""


class ExternalVerifier:
    """用 LLM 作为外部知识库来验证争议事实。

    工作原理:
      1. 提取每个争议中的"事实主张"
      2. 让 LLM 凭训练数据判断哪个主张更符合事实
      3. 返回验证结果 + 置信度

    为什么不接真正的搜索引擎 (Google/Bing API)?
      - 对学习项目来说，LLM 训练数据已经是"外部知识"的代理
      - 真正的搜索 API 需要额外的付费和配置
      - 这里的重点是"验证"这个架构模式——以后可以换成真的搜索

    生活类比: 两个同事争论一个技术问题，第三个人打开 Google 查证。
    这里 LLM 扮演的就是"查 Google 的第三人"。
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(
        self,
        conflicts: list[Conflict],
        verbose: bool = True,
    ) -> list[dict]:
        """验证冲突中的事实主张。

        参数:
            conflicts: 冲突列表

        返回:
            verifications 列表，每个包含 claim verification + winner + confidence
        """
        if not conflicts:
            if verbose:
                print("  [外部验证] 没有争议需要验证")
            return []

        if verbose:
            print(f"  [外部验证] 开始核查 {len(conflicts)} 个技术争议...")

        # 构造验证请求
        parts = []
        for i, c in enumerate(conflicts, 1):
            parts.append(
                f"### 争议 {i}: {c.topic}\n"
                f"- Reviewer 主张: {c.position_a}\n"
                f"  依据: {c.evidence_a}\n"
                f"- Researcher 主张: {c.position_b}\n"
                f"  依据: {c.evidence_b}\n"
            )

        user_msg = f"以下是代码审查中发现的技术争议，请核查每个主张。\n\n{chr(10).join(parts)}"
        raw = self.llm.chat(user_msg, system=EXTERNAL_VERIFIER_PROMPT)
        verifications = self._parse(raw)

        if verbose:
            for v in verifications:
                topic = v.get("conflict_topic", "?")
                winner = v.get("winner", "?")
                confidence = v.get("confidence", "?")
                print(f"  [外部验证] {topic}: winner={winner}, confidence={confidence}")

        return verifications

    def _parse(self, raw: str) -> list[dict]:
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return data.get("verifications", [])
        except (json.JSONDecodeError, IndexError):
            return []

    def apply_verification(
        self,
        review_data: dict,
        conflicts: list[Conflict],
        verifications: list[dict],
    ) -> dict:
        """将外部验证结果应用到审查结果中。

        - winner=a (Reviewer 正确): issue 添加 "external_verification: confirmed" 标记
        - winner=b (Researcher 正确): issue 降级为 Suggestion 或移除
        - winner=unclear: issue 保持原样，添加 unresolved 标记
        """
        resolved = dict(review_data)
        issues = list(resolved.get("issues", []))

        for i, conflict in enumerate(conflicts):
            if i >= len(verifications):
                break
            verification = verifications[i]
            winner = verification.get("winner", "")
            explanation = verification.get("explanation", "")

            for j, issue in enumerate(issues):
                title = issue.get("title", "")
                desc = issue.get("description", "")
                if conflict.topic in title or conflict.topic in desc:
                    if winner == "b":
                        issues[j]["severity"] = "Suggestion"
                        issues[j]["external_note"] = f"[外部验证推翻] {explanation}"
                    elif winner == "a":
                        issues[j]["external_note"] = f"[外部验证确认] {explanation}"
                    else:
                        issues[j]["external_note"] = f"[外部验证无结论] {explanation}"
                    break

        resolved["issues"] = issues
        resolved["external_verification_applied"] = True
        return resolved


# ============================================================
# 6. 冲突解决编排器 -- 整合三种策略
# ============================================================


class ConflictResolver:
    """冲突感知的代码审查编排器。

    与 Day 3/4 的 Orchestrator 的区别:
      Day 3 (CodeReviewOrchestrator): 直接运行 Reviewer + Researcher -> 返回报告
      Day 4 (BusOrchestrator): 通过 MessageBus 消息驱动 -> 返回报告
      Day 5 (ConflictResolver): 运行审查 + 冲突检测 + 三种策略解决 -> 返回解决后的结果

    工作流:
      1. 初始审查: Reviewer + Researcher 各自工作
      2. 冲突检测: ConflictDetector 对比输出，找分歧
      3. 策略组合:
         a. 投票: 多个 Reviewer 实例投票，减少随机偏差
         b. 层级裁决: Arbiter 仲裁所有严重度/遗漏类冲突
         c. 外部验证: LLM 核查所有事实类冲突
      4. 输出: 最终 resolve 后的审查结果 + 完整的冲突解决记录

    三种策略的选择逻辑:
      - 投票: "这个代码段是否有问题？" (存在问题)
      - 层级裁决: "这个问题应该标什么严重度？" (判断问题)
      - 外部验证: "这个技术主张是否符合业界共识？" (事实问题)
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.reviewer = ReviewerAgent(llm)
        self.researcher = ResearcherAgent(llm)
        self.detector = ConflictDetector(llm)
        self.voting = VotingStrategy(llm, num_reviewers=3)
        self.arbiter = ArbiterAgent(llm)
        self.verifier = ExternalVerifier(llm)

    def review_with_conflict_resolution(
        self,
        code: str,
        language: str = "unknown",
        strategies: list[str] | None = None,
        verbose: bool = True,
    ) -> dict:
        """完整的审查 + 冲突检测 + 解决流程。

        参数:
            code: 源代码
            language: 编程语言
            strategies: 要启用的策略，默认全部。可选: "voting", "hierarchy", "external"
            verbose: 是否打印详细过程

        返回:
            {
                "review": ReviewResult,       # 初始审查结果
                "research": ResearchResult,   # 研究结果
                "conflict_report": ConflictReport,  # 冲突检测报告
                "resolution_results": {       # 各策略的解决结果
                    "voting": {...},
                    "hierarchy": {"verdicts": [...], "resolved_review": {...}},
                    "external": {"verifications": [...], "resolved_review": {...}},
                },
                "final_review": {...},        # 最终审查结果 (dict 形式)
            }
        """
        if strategies is None:
            strategies = ["voting", "hierarchy", "external"]

        if verbose:
            print("=" * 60)
            print("ConflictResolver: 冲突感知代码审查")
            print(f"语言: {language}, 代码: {len(code)} 字符 ({code.count(chr(10)) + 1} 行)")
            print(f"启用的策略: {', '.join(strategies)}")
            print("=" * 60)
            print()

        # ---- 阶段 0: 初始审查 ----
        if verbose:
            print("[阶段 0] 初始审查: Reviewer + Researcher")
            print("-" * 40)

        review = self.reviewer.run(code, language)
        research = self.researcher.run(code, language)

        if verbose:
            print(f"  Reviewer: 评分 {review.score}/10, {len(review.issues)} 个问题")
            print(
                f"  Researcher: {len(research.best_practices)} 条最佳实践, "
                f"{len(research.common_pitfalls)} 条常见陷阱"
            )
            print()

        # ---- 阶段 1: 冲突检测 ----
        if verbose:
            print("[阶段 1] 冲突检测: 对比 Reviewer vs Researcher 输出")
            print("-" * 40)

        conflict_report = self.detector.detect(code, language, review, research)

        if verbose:
            if conflict_report.has_conflicts:
                print(f"  [发现] {len(conflict_report.conflicts)} 个冲突:")
                for c in conflict_report.conflicts:
                    label = {"factual": "事实", "severity": "严重度", "omission": "遗漏"}.get(
                        c.conflict_type, c.conflict_type
                    )
                    print(f"    [{label}] {c.topic}")
            else:
                print("  [OK] 未检测到冲突")
            print(f"  [一致] {len(conflict_report.agreements)} 个一致点")
            print()

        # ---- 阶段 2: 应用解决策略 ----
        resolution_results = {}

        # 初始的 final_review 从 Reviewer 的原始输出开始
        final_review = {
            "summary": review.summary,
            "score": review.score,
            "issues": [dict(issue) for issue in review.issues],
        }

        # 策略 A: 投票 (总是先跑，因为它在"发现问题"层面工作)
        if "voting" in strategies:
            if verbose:
                print("[阶段 2a] 投票策略: 多 Reviewer 实例并行 + 少数服从多数")
                print("-" * 40)

            voting_result = self.voting.vote(code, language, verbose=verbose)
            resolution_results["voting"] = voting_result

            # 用投票通过的 issues 替换原始 issues
            final_review["issues"] = voting_result["final_issues"]
            final_review["score"] = voting_result["final_score"]

            if verbose:
                print()

        # 策略 B: 层级裁决 (处理严重度和遗漏类冲突)
        if "hierarchy" in strategies:
            if verbose:
                print("[阶段 2b] 层级裁决: Arbiter 高级仲裁员")
                print("-" * 40)

            hierarchy_conflicts = [
                c for c in conflict_report.conflicts if c.conflict_type in ("severity", "omission")
            ]
            if hierarchy_conflicts:
                if verbose:
                    print(f"  [仲裁] 待裁决冲突: {len(hierarchy_conflicts)} 个 (严重度 + 遗漏)")
                verdicts = self.arbiter.arbitrate(
                    code, language, hierarchy_conflicts, review, research, verbose=verbose
                )
                resolved = self.arbiter.resolve_conflicts(
                    final_review, hierarchy_conflicts, verdicts
                )
                resolution_results["hierarchy"] = {
                    "verdicts": verdicts,
                    "resolved_review": resolved,
                }
                final_review = resolved
            else:
                if verbose:
                    print("  [仲裁] 没有严重度/遗漏类冲突，跳过")
                resolution_results["hierarchy"] = {"verdicts": [], "resolved_review": final_review}

            if verbose:
                print()

        # 策略 C: 外部验证 (处理事实类冲突)
        if "external" in strategies:
            if verbose:
                print("[阶段 2c] 外部验证: LLM 事实核查")
                print("-" * 40)

            factual_conflicts = [
                c for c in conflict_report.conflicts if c.conflict_type == "factual"
            ]
            if factual_conflicts:
                if verbose:
                    print(f"  [外部验证] 待核查争议: {len(factual_conflicts)} 个 (事实类)")
                verifications = self.verifier.verify(factual_conflicts, verbose=verbose)
                resolved = self.verifier.apply_verification(
                    final_review, factual_conflicts, verifications
                )
                resolution_results["external"] = {
                    "verifications": verifications,
                    "resolved_review": resolved,
                }
                final_review = resolved
            else:
                if verbose:
                    print("  [外部验证] 没有事实类冲突，跳过")
                resolution_results["external"] = {
                    "verifications": [],
                    "resolved_review": final_review,
                }

            if verbose:
                print()

        # ---- 阶段 3: 总结 ----
        if verbose:
            print("[阶段 3] 冲突解决完成")
            print("-" * 40)
            print(f"  检测到冲突: {len(conflict_report.conflicts)} 个")
            print(f"  初始 issues: {len(review.issues)} 个")

            # 统计各策略解决的问题
            for strategy_name, result_key in [
                ("投票", "voting"),
                ("层级裁决", "hierarchy"),
                ("外部验证", "external"),
            ]:
                if strategy_name in strategies:
                    result = resolution_results.get(result_key, {})
                    if result_key == "voting":
                        v_result = result.get("voting_result", {})
                        passed = sum(1 for v in v_result.values() if v.get("passed"))
                        rejected = sum(1 for v in v_result.values() if not v.get("passed"))
                        print(f"  {strategy_name}: {passed} 个 issue 通过, {rejected} 个被否决")
                    elif result_key == "hierarchy":
                        v_count = len(result.get("verdicts", []))
                        if v_count:
                            print(f"  {strategy_name}: {v_count} 个冲突已裁决")
                    elif result_key == "external":
                        v_count = len(result.get("verifications", []))
                        if v_count:
                            print(f"  {strategy_name}: {v_count} 个主张已核查")

            print(f"  最终 issues: {len(final_review.get('issues', []))} 个")
            print()

        return {
            "review": review,
            "research": research,
            "conflict_report": conflict_report,
            "resolution_results": resolution_results,
            "final_review": final_review,
        }
