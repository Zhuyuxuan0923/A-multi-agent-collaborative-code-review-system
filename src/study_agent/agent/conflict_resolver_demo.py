"""
冲突解决系统演示 -- Week 7 Day 5。

运行方式: python -m src.study_agent.agent.conflict_resolver_demo

这个演示分五个步骤:
  1. 展示会引发 Agent 冲突的代码样本
  2. 运行初始审查 (Reviewer + Researcher)
  3. 冲突检测: 对比两个 Agent 的输出，找出分歧
  4. 应用三种策略逐一解决冲突
  5. 输出对比: 解决前 vs 解决后的审查结果

三种策略的演示顺序:
  - 投票: 3 个 Reviewer 实例对每个问题投票 (解决"发现偏差")
  - 层级裁决: Arbiter 仲裁争议 (解决"判断偏差")
  - 外部验证: LLM 事实核查 (解决"知识偏差")
"""

import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from study_agent.agent.code_review_agents import ResearcherAgent, ReviewerAgent
from study_agent.agent.conflict_resolver import ConflictResolver
from study_agent.llm.client import LLMClient

# ================================================================
# 测试代码样本 -- 精心设计的"冲突磁铁"
# ================================================================

SAMPLE_CODE = '''
import pickle
import sqlite3
import os

ADMIN_PASSWORD = "super_secret_123"

def load_user_data(filename):
    """Load user data from a file using pickle."""
    data = open(filename, 'rb').read()
    return pickle.loads(data)

def get_user(db_path, user_id):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return conn.execute(query).fetchone()

def process_template(template_str, context):
    """Process a template string with variable substitution."""
    for key, value in context.items():
        template_str = template_str.replace("{{" + key + "}}", str(value))
    return eval(template_str)  # Evaluate the final expression

def save_user_data(filename, data):
    f = open("/tmp/" + filename, "w")
    f.write(str(data))
    # Note: file handle not closed
'''

SAMPLE_LANGUAGE = "python"


def print_section(title: str):
    """打印一个分隔的章节标题。"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subsection(title: str):
    """打印子标题。"""
    print()
    print("-" * 40)
    print(title)
    print("-" * 40)


def show_code(code: str, language: str):
    """打印代码样本，带行号。"""
    print(f"```{language}")
    for i, line in enumerate(code.strip().split("\n"), 1):
        print(f"{i:2d} | {line}")
    print("```")


def show_review_result(review):
    """格式化打印 Reviewer 的输出。"""
    print(f"  评分: {review.score}/10")
    print(f"  摘要: {review.summary}")
    print(f"  问题数: {len(review.issues)}")
    for i, issue in enumerate(review.issues, 1):
        sev = issue.get("severity", "?")
        cat = issue.get("category", "?")
        title = issue.get("title", "?")
        print(f"    [{i}] [{sev}] [{cat}] {title}")


def show_research_result(research):
    """格式化打印 Researcher 的输出。"""
    print(f"  技术识别: {research.technologies}")
    print(f"  最佳实践: {len(research.best_practices)} 条")
    for bp in research.best_practices[:3]:
        print(f"    - {bp.get('title', '?')}")
    print(f"  常见陷阱: {len(research.common_pitfalls)} 条")
    for cp in research.common_pitfalls[:3]:
        print(f"    - {cp.get('title', '?')}")
    if research.recommendations:
        print(f"  改进建议: {len(research.recommendations)} 条")


def show_conflict_report(conflict_report):
    """格式化打印冲突检测结果。"""
    print(f"  有冲突: {conflict_report.has_conflicts}")
    print(f"  冲突数: {len(conflict_report.conflicts)}")
    type_labels = {"factual": "事实冲突", "severity": "严重度冲突", "omission": "遗漏"}
    for i, c in enumerate(conflict_report.conflicts, 1):
        label = type_labels.get(c.conflict_type, c.conflict_type)
        print(f"  --- 冲突 {i}: [{label}] {c.topic} ---")
        print(f"    Reviewer 立场: {c.position_a[:80]}...")
        print(f"    Researcher 立场: {c.position_b[:80]}...")
    print(f"  一致点: {len(conflict_report.agreements)} 个")
    for a in conflict_report.agreements:
        print(f"    - {a.get('topic', '?')}")


def compare_before_after(before: dict, after: dict):
    """对比解决前后的审查结果。"""
    print_subsection("解决前 vs 解决后 对比")

    before_issues = before.get("issues", [])
    after_issues = after.get("issues", [])

    print(f"  Issue 数量: {len(before_issues)} -> {len(after_issues)}")
    print(f"  评分: {before.get('score', '?')} -> {after.get('score', '?')}")
    print()

    # 列出被移除的 issues
    before_titles = {i.get("title", "") for i in before_issues}
    after_titles = {i.get("title", "") for i in after_issues}
    removed = before_titles - after_titles
    added = after_titles - before_titles

    if removed:
        print(f"  被解决策略移除或合并的问题 ({len(removed)} 个):")
        for title in removed:
            print(f"    [移除] {title}")
    if added:
        print(f"  解决过程中新增的问题 ({len(added)} 个):")
        for title in added:
            print(f"    [新增] {title}")

    # 列出严重度变更
    print()
    print("  严重度变更:")
    before_sev = {i.get("title", ""): i.get("severity", "") for i in before_issues}
    after_sev = {i.get("title", ""): i.get("severity", "") for i in after_issues}
    changed = 0
    for title, old_sev in before_sev.items():
        new_sev = after_sev.get(title, "")
        if new_sev and old_sev != new_sev:
            print(f"    {title}: {old_sev} -> {new_sev}")
            changed += 1
    if changed == 0:
        print("    (无变更)")

    print()
    print("  最终 Issue 列表:")
    for i, issue in enumerate(after_issues, 1):
        sev = issue.get("severity", "?")
        cat = issue.get("category", "?")
        title = issue.get("title", "?")
        extra = ""
        if issue.get("arbiter_note"):
            extra += " [经仲裁]"
        if issue.get("external_note"):
            extra += " [经外部验证]"
        if issue.get("from_arbiter"):
            extra += " [仲裁补充]"
        print(f"    [{i}] [{sev}] [{cat}] {title}{extra}")


def save_report(output: dict, code: str, language: str, provider: str, elapsed_ms: float):
    """保存完整的冲突解决报告到 data/ 目录。"""
    output_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "data",
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"conflict_resolution_report_{timestamp}.md")

    conflict_report = output["conflict_report"]
    resolution_results = output["resolution_results"]
    final_review = output["final_review"]

    # 构建 Markdown 报告
    lines = [
        "# 冲突解决审查报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> LLM Provider: {provider}",
        f"> 审查语言: {language}",
        f"> 总耗时: {elapsed_ms:.0f}ms",
        "",
        "---",
        "",
        "## 1. 冲突检测结果",
        "",
        f"检测到 **{len(conflict_report.conflicts)}** 个冲突, **{len(conflict_report.agreements)}** 个一致点。",
        "",
    ]

    if conflict_report.conflicts:
        lines.append("| # | 类型 | 主题 |")
        lines.append("|---|------|------|")
        for i, c in enumerate(conflict_report.conflicts, 1):
            type_label = {"factual": "事实", "severity": "严重度", "omission": "遗漏"}.get(
                c.conflict_type, c.conflict_type
            )
            lines.append(f"| {i} | {type_label} | {c.topic} |")
        lines.append("")

    lines.extend(
        [
            "## 2. 解决策略结果",
            "",
        ]
    )

    # 投票结果
    voting = resolution_results.get("voting", {})
    if voting:
        v_result = voting.get("voting_result", {})
        passed = sum(1 for v in v_result.values() if v.get("passed"))
        rejected = sum(1 for v in v_result.values() if not v.get("passed"))
        lines.append(f"### 投票策略: {passed} 个通过, {rejected} 个被否决")
        lines.append("")
        if v_result:
            lines.append("| Issue | 投票 | 结果 |")
            lines.append("|-------|------|------|")
            for key, v in v_result.items():
                result = "[OK] 通过" if v.get("passed") else "[X] 否决"
                lines.append(
                    f"| {v.get('topic', key)[:50]} | {v.get('votes_for', 0)}/{v.get('total', 0)} | {result} |"
                )
            lines.append("")

    # 仲裁结果
    hierarchy = resolution_results.get("hierarchy", {})
    verdicts = hierarchy.get("verdicts", [])
    if verdicts:
        lines.append(f"### 层级裁决: {len(verdicts)} 个冲突已裁决")
        lines.append("")
        for v in verdicts:
            decision_map = {
                "uphold_reviewer": "支持 Reviewer",
                "uphold_researcher": "支持 Researcher",
                "compromise": "折中",
            }
            decision = decision_map.get(v.get("decision", ""), v.get("decision", "?"))
            lines.append(f"- **{v.get('conflict_topic', '?')}**: {decision}")
            lines.append(f"  > {v.get('reasoning', '')[:200]}")
        lines.append("")

    # 外部验证结果
    external = resolution_results.get("external", {})
    verifications = external.get("verifications", [])
    if verifications:
        lines.append(f"### 外部验证: {len(verifications)} 个主张已核查")
        lines.append("")
        for v in verifications:
            lines.append(
                f"- **{v.get('conflict_topic', '?')}**: 判定支持 {v.get('winner', '?')} (置信度: {v.get('confidence', '?')})"
            )
            lines.append(f"  > {v.get('explanation', '')[:200]}")
        lines.append("")

    lines.extend(
        [
            "## 3. 最终审查结果",
            "",
            f"- 最终评分: {final_review.get('score', '?')}/10",
            f"- 最终 issues: {len(final_review.get('issues', []))} 个",
            "",
        ]
    )

    if final_review.get("issues"):
        lines.append("| # | 严重度 | 类别 | 问题 |")
        lines.append("|---|--------|------|------|")
        for i, issue in enumerate(final_review["issues"], 1):
            sev = issue.get("severity", "?")
            cat = issue.get("category", "?")
            title = issue.get("title", "?")
            lines.append(f"| {i} | {sev} | {cat} | {title[:60]} |")
        lines.append("")

    lines.extend(
        [
            "## 4. 原始代码",
            "",
            f"```{language}",
            code,
            "```",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


async def main():
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"LLM Provider: {provider}")

    llm = LLMClient(provider=provider)

    code = SAMPLE_CODE.strip()
    language = SAMPLE_LANGUAGE

    # ================================================================
    # 第 1 步: 展示代码 -- 为什么这段代码容易引发冲突？
    # ================================================================
    print_section("冲突解决系统演示")

    print("待审查代码:")
    show_code(code, language)

    print()
    print("这段代码为什么容易引发 Agent 冲突？")
    print()
    print("  1. pickle.loads() 接收外部数据 -- Reviewer 会标记为 Critical RCE,")
    print("     Researcher 可能会说'pickle 是 Python 标准库, 内部使用是安全的'")
    print()
    print("  2. SQL 字符串拼接 -- Reviewer 标记为 Critical SQL 注入,")
    print("     Researcher 可能会建议'使用 ORM 可以自动转义'(偏离了实际代码)")
    print()
    print("  3. eval() 执行动态表达式 -- Reviewer 标记为 Critical,")
    print("     Researcher 可能会讨论'模板引擎是常见模式'(模糊了 eval 的危险)")
    print()
    print("  4. 硬编码密码 + 文件句柄泄漏 -- 这些是明确的问题,")
    print("     但两个 Agent 对其严重度的判断可能不同")
    print()

    # ================================================================
    # 第 2 步: 初始审查 -- 看看 Reviewer 和 Researcher 各自怎么说
    # ================================================================
    print_section("第 2 步: 初始审查 (Reviewer + Researcher)")

    reviewer = ReviewerAgent(llm)
    researcher = ResearcherAgent(llm)

    print("运行 Reviewer (审查员)...")
    review = reviewer.run(code, language)
    show_review_result(review)

    print_subsection("")
    print("运行 Researcher (研究员)...")
    research = researcher.run(code, language)
    show_research_result(research)

    print()
    print("[观察] 注意 Reviewer 和 Researcher 的视角差异:")
    print("  Reviewer 关注'这段代码有什么问题?'")
    print("  Researcher 关注'这段代码用到的技术有什么最佳实践?'")
    print("  两个视角可能对同一段代码有不同的判断 -- 这就是冲突的来源。")

    # ================================================================
    # 第 3 步: 冲突检测
    # ================================================================
    print_section("第 3 步: 冲突检测 -- 找出两个 Agent 的分歧")

    resolver = ConflictResolver(llm)
    conflict_report = resolver.detector.detect(code, language, review, research)
    show_conflict_report(conflict_report)

    if not conflict_report.has_conflicts:
        print()
        print("[注意] 本次审查未检测到明显冲突。这是正常的 -- 不是每次审查都会有冲突。")
        print("冲突解决系统的价值恰恰在于: 当冲突发生时, 有明确的处理机制。")
        print("为了演示三种策略, 接下来的步骤仍会运行完整流程。")

    # ================================================================
    # 第 4 步: 三种策略逐一演示
    # ================================================================
    print_section("第 4 步: 三种策略解决冲突")

    start_time = time.time()

    # 使用完整的 ConflictResolver 运行
    result = resolver.review_with_conflict_resolution(
        code,
        language,
        strategies=["voting", "hierarchy", "external"],
        verbose=True,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    # ================================================================
    # 第 5 步: 对比 -- 解决前 vs 解决后
    # ================================================================
    print_section("第 5 步: 冲突解决前后对比")

    # 构造"解决前"的状态 (投票后的 raw issues)
    before = {
        "score": review.score,
        "issues": [dict(i) for i in review.issues],
    }
    after = result["final_review"]

    compare_before_after(before, after)

    # ================================================================
    # 总结
    # ================================================================
    print_section("三种策略总结")

    print(
        """
  +-- 投票策略 --+
  |
  | 原理: 多个 Reviewer 实例并行审查, 对每个发现的问题投票
  |       过半数同意 -> 通过; 不过半数 -> 否决
  |
  | 适用: "这段代码有没有问题?" (存在性问题)
  | 优势: 平滑 LLM 随机性, 减少误报/漏报
  | 劣势: 需要多次 LLM 调用, 成本是原来的 N 倍
  |
  +-- 层级裁决 --+
  |
  | 原理: 引入 Arbiter (仲裁员), 职位高于 Reviewer/Researcher
  |       对争议做最终裁定, 一锤定音
  |
  | 适用: "这个问题应该标什么严重度?" (判断问题)
  | 优势: 决策快, 责任明确
  | 劣势: 单点决策风险, Arbiter 判断失误则全盘皆输
  |
  +-- 外部验证 --+
  |
  | 原理: 用 LLM 的知识库做"事实核查"
  |       对争议中的技术主张, 查证业界共识/文档/标准
  |
  | 适用: "这个技术主张是否符合业界共识?" (事实问题)
  | 优势: 有据可查, 不是"我觉得"而是"标准说"
  | 劣势: 依赖 LLM 训练数据的时效性, 无法验证最新技术
  |
  +-- 组合使用 --+
  |
  | 工作流: 投票 (发现问题) -> 层级裁决 (判定严重度) -> 外部验证 (核实事实)
  | 三种策略各司其职, 覆盖不同类型的冲突
  """
    )

    # ================================================================
    # 保存报告
    # ================================================================
    report_path = save_report(result, code, language, provider, elapsed_ms)
    print(f"冲突解决报告已保存到: {report_path}")
    print(f"总耗时: {elapsed_ms:.0f}ms")

    print()
    print("Day 3 (CodeReviewOrchestrator): 直接运行 Agent, 不处理冲突")
    print("Day 4 (BusOrchestrator): 消息驱动, 有重试无冲突处理")
    print("Day 5 (ConflictResolver): 冲突检测 + 三种策略 + 完整解决记录")


if __name__ == "__main__":
    asyncio.run(main())
