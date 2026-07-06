"""
多 Agent 代码审查系统集成测试 -- Week 7 Day 6。

端到端测试: 用真实场景的代码片段跑通整个多 Agent 审查流程。

测试三个 Pipeline:
  Pipeline A (Day 3): CodeReviewOrchestrator — 直接函数调用
  Pipeline B (Day 4): BusOrchestrator — 消息驱动编排
  Pipeline C (Day 5): ConflictResolver — 带冲突解决的审查

记录维度:
  - 每个阶段的耗时 (ms)
  - 每个 Agent 的决策 (发现了什么/建议了什么)
  - 消息流转记录 (Pipeline B 专属)
  - 冲突检测与解决记录 (Pipeline C 专属)
  - 最终审查报告

运行方式: python -m src.study_agent.agent.integration_test
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from study_agent.agent.bus_orchestrator import BusOrchestrator
from study_agent.agent.code_review_agents import (
    CodeReviewOrchestrator,
)
from study_agent.agent.conflict_resolver import ConflictResolver
from study_agent.llm.client import LLMClient

# ================================================================
# 测试用例定义
# ================================================================


@dataclass
class TestCase:
    """一个集成测试用例。"""

    name: str
    description: str
    code: str
    language: str
    expected_issues_min: int = 1  # 预期最少发现的问题数
    expected_issues_max: int = 20  # 预期最多发现的问题数


# 测试用例 1: Python 用户注册 + 登录接口 (真实 Web 后端场景)
TEST_CASE_1_CODE = '''
import hashlib
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)
SECRET_KEY = "my-secret-key-123"

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.md5(password.encode()).hexdigest()

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    email = data.get("email", "")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Check if user exists
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    existing = cursor.execute(query).fetchone()
    if existing:
        return jsonify({"error": "User already exists"}), 400

    # Insert new user
    hashed = hash_password(password)
    insert_query = "INSERT INTO users (username, password, email) VALUES ('" + username + "', '" + hashed + "', '" + email + "')"
    cursor.execute(insert_query)
    conn.commit()

    user_id = cursor.lastrowid
    return jsonify({"id": user_id, "username": username}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + hash_password(password) + "'"
    user = cursor.execute(query).fetchone()

    if user:
        token = SECRET_KEY + str(user[0])
        return jsonify({"token": token, "user": {"id": user[0], "username": user[1]}})
    return jsonify({"error": "Invalid credentials"}), 401

if __name__ == "__main__":
    app.run(debug=True)
'''

# 测试用例 2: Node.js 任务管理 API (跨语言 + 异步问题)
TEST_CASE_2_CODE = """
const express = require('express');
const mysql = require('mysql');
const app = express();
app.use(express.json());

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'admin123',
    database: 'tasks'
});

app.post('/api/tasks', (req, res) => {
    const { title, assignee, dueDate } = req.body;

    const query = `INSERT INTO tasks (title, assignee, due_date) VALUES ('${title}', '${assignee}', '${dueDate}')`;
    db.query(query, (err, result) => {
        if (err) {
            console.log(err);
            res.status(500).send('Error');
            return;
        }
        res.json({ id: result.insertId, title: title });
    });
});

app.get('/api/tasks/search', (req, res) => {
    const keyword = req.query.q;
    const sql = `SELECT * FROM tasks WHERE title LIKE '%${keyword}%'`;

    db.query(sql, (err, rows) => {
        if (err) return res.status(500).send('Database error');
        const tasks = rows.map(row => {
            return { ...row, dueDate: eval(row.due_date) };
        });
        res.json(tasks);
    });
});

app.get('/api/admin/users', (req, res) => {
    const users = db.query('SELECT id, username, password FROM users');
    res.json(users);
});

app.listen(3000, () => console.log('Server running on port 3000'));
"""

# 测试用例 3: Java 电商订单服务 (类型安全 + 并发问题)
TEST_CASE_3_CODE = """
import java.sql.*;
import java.util.*;
import java.util.concurrent.*;

public class OrderService {
    private static final String DB_URL = "jdbc:mysql://localhost:3306/shop";
    private static final String DB_USER = "admin";
    private static final String DB_PASS = "P@ssword123";
    private static Map<Integer, Order> orderCache = new HashMap<>();

    public Order createOrder(int userId, List<Item> items) throws SQLException {
        Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASS);

        double total = 0;
        for (Item item : items) {
            total += item.price * item.quantity;
        }

        String sql = "INSERT INTO orders (user_id, total) VALUES (" + userId + ", " + total + ")";
        Statement stmt = conn.createStatement();
        stmt.executeUpdate(sql);

        ResultSet rs = stmt.executeQuery("SELECT LAST_INSERT_ID()");
        rs.next();
        int orderId = rs.getInt(1);

        for (Item item : items) {
            String itemSql = "INSERT INTO order_items (order_id, product_id, qty) VALUES (" + orderId + ", " + item.productId + ", " + item.quantity + ")";
            stmt.executeUpdate(itemSql);
        }

        Order order = new Order(orderId, userId, items, total);
        orderCache.put(orderId, order);
        return order;
    }

    public Order getOrder(int orderId) {
        if (orderCache.containsKey(orderId)) {
            return orderCache.get(orderId);
        }
        return null;
    }

    public void processRefund(int orderId) throws Exception {
        Order order = getOrder(orderId);
        if (order == null) throw new Exception("Order not found");

        Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASS);
        String sql = "UPDATE orders SET status = 'REFUNDED' WHERE id = " + orderId;
        conn.createStatement().executeUpdate(sql);

        // Send refund notification
        ExecutorService executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                sendRefundEmail(order.userId, orderId);
            } catch (Exception e) {
                // silently ignore
            }
        });
    }

    private void sendRefundEmail(int userId, int orderId) {
        // stub
    }

    static class Order {
        int id, userId;
        List<Item> items;
        double total;
        Order(int id, int userId, List<Item> items, double total) {
            this.id = id; this.userId = userId; this.items = items; this.total = total;
        }
    }

    static class Item {
        int productId, quantity;
        double price;
    }
}
"""

TEST_CASES = [
    TestCase(
        name="Python Flask 用户注册登录 API",
        description="包含 SQL 注入、MD5 弱哈希、硬编码密钥、无输入校验 等典型安全问题",
        code=TEST_CASE_1_CODE.strip(),
        language="python",
    ),
    TestCase(
        name="Node.js 任务管理 API",
        description="跨语言测试: SQL 注入、eval 代码执行、明文密码、无鉴权的管理接口",
        code=TEST_CASE_2_CODE.strip(),
        language="javascript",
    ),
    TestCase(
        name="Java 电商订单服务",
        description="类型安全语言测试: SQL 注入、HashMap 并发问题、连接泄漏、硬编码凭据、静默吞异常",
        code=TEST_CASE_3_CODE.strip(),
        language="java",
    ),
]


# ================================================================
# 计时工具
# ================================================================


class Timer:
    """记录一个阶段的开始时间、结束时间和耗时。"""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self


# ================================================================
# 测试结果数据结构
# ================================================================


@dataclass
class StageResult:
    """一个测试阶段的记录。"""

    stage_name: str
    elapsed_ms: float
    details: dict = field(default_factory=dict)
    status: str = "ok"  # ok | warning | error


@dataclass
class PipelineResult:
    """一个完整 Pipeline 的测试结果。"""

    pipeline_name: str
    total_elapsed_ms: float = 0.0
    stages: list[StageResult] = field(default_factory=list)
    agent_decisions: dict = field(default_factory=dict)
    report: str = ""
    status: str = "ok"


@dataclass
class IntegrationTestReport:
    """完整的集成测试报告。"""

    test_case_name: str
    provider: str
    timestamp: str = ""
    pipelines: list[PipelineResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ================================================================
# 集成测试运行器
# ================================================================


class IntegrationTestRunner:
    """多 Agent 代码审查系统的集成测试运行器。

    测试三个 Pipeline:
      Pipeline A (Day 3): 直接调用编排 — 基准性能
      Pipeline B (Day 4): 消息驱动编排 — 通信开销
      Pipeline C (Day 5): 冲突感知审查 — 准确性提升

    每个 Pipeline 记录:
      - 各阶段耗时
      - Agent 决策摘要
      - 最终报告
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ----------------------------------------------------------------
    # Pipeline A: Day 3 — 直接函数调用编排
    # ----------------------------------------------------------------

    def run_pipeline_a(self, test_case: TestCase, verbose: bool = True) -> PipelineResult:
        """Pipeline A: CodeReviewOrchestrator (Day 3)。

        这是最简的路径: Orchestrator 直接调用 agent.run()。
        没有消息总线, 没有冲突解决。作为性能基准。
        """
        if verbose:
            print()
            print("=" * 60)
            print("Pipeline A (Day 3): 直接函数调用编排")
            print(f"测试用例: {test_case.name}")
            print("=" * 60)

        result = PipelineResult(pipeline_name="A-DirectCall (Day 3)")
        orchestrator = CodeReviewOrchestrator(self.llm)
        total_timer = Timer("total").start()

        try:
            # 阶段 1: Reviewer
            t1 = Timer("reviewer").start()
            review = orchestrator.reviewer.run(test_case.code, test_case.language)
            t1.stop()
            result.stages.append(
                StageResult(
                    stage_name="Reviewer 审查",
                    elapsed_ms=t1.elapsed_ms,
                    details={
                        "score": review.score,
                        "issues_count": len(review.issues),
                        "issues": [i.get("title", "") for i in review.issues[:5]],
                    },
                )
            )
            if verbose:
                print(
                    f"  [Reviewer] {t1.elapsed_ms:.0f}ms | 评分 {review.score}/10, {len(review.issues)} 个问题"
                )

            # 阶段 2: Researcher
            t2 = Timer("researcher").start()
            research = orchestrator.researcher.run(test_case.code, test_case.language)
            t2.stop()
            result.stages.append(
                StageResult(
                    stage_name="Researcher 研究",
                    elapsed_ms=t2.elapsed_ms,
                    details={
                        "technologies": research.technologies,
                        "practices_count": len(research.best_practices),
                        "pitfalls_count": len(research.common_pitfalls),
                    },
                )
            )
            if verbose:
                print(
                    f"  [Researcher] {t2.elapsed_ms:.0f}ms | {len(research.technologies)} 项技术, {len(research.best_practices)} 条实践"
                )

            # 阶段 3: Reporter
            t3 = Timer("reporter").start()
            report = orchestrator.reporter.run(test_case.code, test_case.language, review, research)
            t3.stop()
            result.stages.append(
                StageResult(
                    stage_name="Reporter 汇总",
                    elapsed_ms=t3.elapsed_ms,
                    details={"report_length": len(report)},
                )
            )
            if verbose:
                print(f"  [Reporter] {t3.elapsed_ms:.0f}ms | 报告 {len(report)} 字符")

            result.report = report
            result.total_elapsed_ms = t1.elapsed_ms + t2.elapsed_ms + t3.elapsed_ms

        except Exception as e:
            result.status = "error"
            result.stages.append(
                StageResult(
                    stage_name="Pipeline A 异常",
                    elapsed_ms=0,
                    details={"error": str(e)},
                    status="error",
                )
            )
            if verbose:
                print(f"  [ERR] Pipeline A 异常: {e}")

        total_timer.stop()
        result.agent_decisions = {
            "reviewer": {
                "score": review.score if "review" in dir() else 0,
                "issues": len(review.issues) if "review" in dir() else 0,
            },
            "researcher": {
                "practices": len(research.best_practices) if "research" in dir() else 0,
            },
        }

        if verbose:
            print(f"  [总耗时] {result.total_elapsed_ms:.0f}ms")

        return result

    # ----------------------------------------------------------------
    # Pipeline B: Day 4 — 消息驱动编排 (BusOrchestrator)
    # ----------------------------------------------------------------

    async def run_pipeline_b(self, test_case: TestCase, verbose: bool = True) -> PipelineResult:
        """Pipeline B: BusOrchestrator (Day 4)。

        与 Pipeline A 相同的 Agent, 但通过 MessageBus 通信。
        记录消息流转、超时处理、重试决策。
        """
        if verbose:
            print()
            print("=" * 60)
            print("Pipeline B (Day 4): 消息驱动编排")
            print(f"测试用例: {test_case.name}")
            print("=" * 60)

        result = PipelineResult(pipeline_name="B-MessageBus (Day 4)")
        orchestrator = BusOrchestrator(self.llm)

        # 记录消息流转
        messages_logged = []

        def message_logger(msg):
            from study_agent.agent.message_protocol import MessageType

            if msg.message_type != MessageType.HEARTBEAT:
                messages_logged.append(
                    {
                        "sender": msg.sender_id,
                        "receiver": msg.receiver_id or f"topic:{msg.topic}",
                        "type": msg.message_type.value,
                        "payload_type": msg.payload.get("task_type", ""),
                    }
                )
            return msg

        orchestrator.bus.use(message_logger)

        total_timer = Timer("total").start()

        try:
            # 因为 BusOrchestrator.review() 内部已经包含了所有阶段,
            # 我们用 verbose=False 来减少输出, 只记录我们关心的数据
            report = await orchestrator.review(
                test_case.code,
                test_case.language,
                max_retries=1,
                request_timeout=90.0,
                verbose=False,  # 关闭内部 verbose, 由测试框架控制输出
            )

            total_timer.stop()
            result.report = report
            result.total_elapsed_ms = total_timer.elapsed_ms

            # 从统计数据推断阶段耗时
            result.stages.append(
                StageResult(
                    stage_name="Orchestrator 完整流程",
                    elapsed_ms=total_timer.elapsed_ms,
                    details={
                        "report_length": len(report),
                        "messages_exchanged": len(messages_logged),
                    },
                )
            )

            # 记录每个 Agent 的统计
            for agent in [
                orchestrator.reviewer_bus,
                orchestrator.researcher_bus,
                orchestrator.reporter_bus,
            ]:
                s = agent.stats
                result.stages.append(
                    StageResult(
                        stage_name=f"{s['role']} 统计",
                        elapsed_ms=s["avg_latency_ms"],
                        details={
                            "messages_processed": s["messages_processed"],
                            "avg_latency_ms": s["avg_latency_ms"],
                        },
                    )
                )
                if verbose:
                    print(
                        f"  [{s['role']}] {s['messages_processed']} 条消息, 平均 {s['avg_latency_ms']:.0f}ms"
                    )

            result.agent_decisions["messages_logged"] = messages_logged

            if verbose:
                print(f"  [消息流转] {len(messages_logged)} 条消息")
                print(f"  [总耗时] {total_timer.elapsed_ms:.0f}ms")

        except Exception as e:
            result.status = "error"
            result.stages.append(
                StageResult(
                    stage_name="Pipeline B 异常",
                    elapsed_ms=total_timer.elapsed_ms if total_timer.start_time else 0,
                    details={"error": str(e)},
                    status="error",
                )
            )
            if verbose:
                print(f"  [ERR] Pipeline B 异常: {e}")

        return result

    # ----------------------------------------------------------------
    # Pipeline C: Day 5 — 冲突感知审查 (ConflictResolver)
    # ----------------------------------------------------------------

    def run_pipeline_c(self, test_case: TestCase, verbose: bool = True) -> PipelineResult:
        """Pipeline C: ConflictResolver (Day 5)。

        在 Pipeline A 基础上增加冲突检测和三种解决策略。
        记录冲突数、仲裁决策、外部验证结论。
        """
        if verbose:
            print()
            print("=" * 60)
            print("Pipeline C (Day 5): 冲突感知审查")
            print(f"测试用例: {test_case.name}")
            print("=" * 60)

        result = PipelineResult(pipeline_name="C-ConflictResolver (Day 5)")
        resolver = ConflictResolver(self.llm)

        total_timer = Timer("total").start()

        try:
            output = resolver.review_with_conflict_resolution(
                test_case.code,
                test_case.language,
                strategies=["voting", "hierarchy", "external"],
                verbose=False,
            )

            total_timer.stop()
            result.total_elapsed_ms = total_timer.elapsed_ms

            # 提取各阶段信息
            conflict_report = output["conflict_report"]
            resolution_results = output["resolution_results"]
            final_review = output["final_review"]

            # 阶段 0: 初始审查
            t0_ms = 0  # 没有单独计时, 从 total 中估算
            result.stages.append(
                StageResult(
                    stage_name="初始审查 (Reviewer + Researcher)",
                    elapsed_ms=t0_ms,
                    details={
                        "reviewer_score": output["review"].score,
                        "reviewer_issues": len(output["review"].issues),
                        "researcher_practices": len(output["research"].best_practices),
                    },
                )
            )

            # 阶段 1: 冲突检测
            result.stages.append(
                StageResult(
                    stage_name="冲突检测",
                    elapsed_ms=0,  # 包含在 total 中
                    details={
                        "has_conflicts": conflict_report.has_conflicts,
                        "conflict_count": len(conflict_report.conflicts),
                        "agreement_count": len(conflict_report.agreements),
                        "conflicts": [
                            {"type": c.conflict_type, "topic": c.topic[:60]}
                            for c in conflict_report.conflicts
                        ],
                    },
                )
            )

            # 阶段 2a: 投票
            voting = resolution_results.get("voting", {})
            v_result = voting.get("voting_result", {})
            passed = sum(1 for v in v_result.values() if v.get("passed"))
            rejected = sum(1 for v in v_result.values() if not v.get("passed"))
            result.stages.append(
                StageResult(
                    stage_name="投票策略 (3 个 Reviewer)",
                    elapsed_ms=0,
                    details={
                        "passed": passed,
                        "rejected": rejected,
                        "final_score": voting.get("final_score", 0),
                    },
                )
            )

            # 阶段 2b: 层级裁决
            hierarchy = resolution_results.get("hierarchy", {})
            verdicts = hierarchy.get("verdicts", [])
            result.stages.append(
                StageResult(
                    stage_name="层级裁决 (Arbiter 仲裁)",
                    elapsed_ms=0,
                    details={
                        "verdict_count": len(verdicts),
                        "verdicts": [
                            {
                                "topic": v.get("conflict_topic", "")[:60],
                                "decision": v.get("decision", ""),
                            }
                            for v in verdicts
                        ],
                    },
                )
            )

            # 阶段 2c: 外部验证
            external = resolution_results.get("external", {})
            verifications = external.get("verifications", [])
            result.stages.append(
                StageResult(
                    stage_name="外部验证 (LLM 事实核查)",
                    elapsed_ms=0,
                    details={
                        "verification_count": len(verifications),
                        "verifications": [
                            {
                                "topic": v.get("conflict_topic", "")[:60],
                                "winner": v.get("winner", ""),
                                "confidence": v.get("confidence", ""),
                            }
                            for v in verifications
                        ],
                    },
                )
            )

            # 最终结果
            result.stages.append(
                StageResult(
                    stage_name="最终审查结果",
                    elapsed_ms=0,
                    details={
                        "final_score": final_review.get("score", 0),
                        "final_issues": len(final_review.get("issues", [])),
                        "arbitration_applied": final_review.get("arbitration_applied", False),
                        "external_verification_applied": final_review.get(
                            "external_verification_applied", False
                        ),
                    },
                )
            )

            result.agent_decisions = {
                "reviewer_initial_score": output["review"].score,
                "reviewer_initial_issues": len(output["review"].issues),
                "final_score": final_review.get("score", 0),
                "final_issues": len(final_review.get("issues", [])),
                "conflicts_detected": len(conflict_report.conflicts),
            }

            # 用 final_review 生成一个简单的文本报告
            lines = [
                f"## 冲突感知审查 — {test_case.name}",
                "",
                f"初始评分: {output['review'].score}/10 (解决后: {final_review.get('score', '?')})",
                f"初始问题: {len(output['review'].issues)} (解决后: {len(final_review.get('issues', []))})",
                f"检测到冲突: {len(conflict_report.conflicts)} 个",
                f"外部验证: {len(verifications)} 个主张已核查",
                "",
                "### 最终 Issues",
            ]
            for i, issue in enumerate(final_review.get("issues", []), 1):
                lines.append(f"{i}. [{issue.get('severity', '?')}] {issue.get('title', '?')}")
            result.report = "\n".join(lines)

            if verbose:
                print(f"  [冲突] 检测到 {len(conflict_report.conflicts)} 个冲突")
                print(f"  [投票] {passed} 个通过, {rejected} 个被否决")
                print(f"  [仲裁] {len(verdicts)} 个裁决")
                print(f"  [验证] {len(verifications)} 个核查")
                print(f"  [总耗时] {total_timer.elapsed_ms:.0f}ms")

        except Exception as e:
            result.status = "error"
            result.stages.append(
                StageResult(
                    stage_name="Pipeline C 异常",
                    elapsed_ms=total_timer.elapsed_ms if total_timer.start_time else 0,
                    details={"error": str(e)},
                    status="error",
                )
            )
            if verbose:
                print(f"  [ERR] Pipeline C 异常: {e}")

        return result

    # ----------------------------------------------------------------
    # 运行集成测试
    # ----------------------------------------------------------------

    async def run_all(
        self,
        test_cases: list[TestCase] | None = None,
        pipelines: list[str] | None = None,
        verbose: bool = True,
    ) -> list[IntegrationTestReport]:
        """运行所有测试用例 x 所有 Pipeline 的组合。

        参数:
            test_cases: 测试用例列表，默认用 TEST_CASES 全部
            pipelines: 要运行的 Pipeline，可选 "A", "B", "C"，默认全部
            verbose: 是否打印详细过程

        返回:
            每个测试用例一个 IntegrationTestReport
        """
        if test_cases is None:
            test_cases = TEST_CASES
        if pipelines is None:
            pipelines = ["A", "B", "C"]

        reports: list[IntegrationTestReport] = []

        for tc in test_cases:
            if verbose:
                print()
                print("*" * 60)
                print(f"测试用例: {tc.name}")
                print(f"描述: {tc.description}")
                print(f"代码: {len(tc.code)} 字符, {tc.code.count(chr(10)) + 1} 行")
                print(f"Pipeline: {', '.join(pipelines)}")
                print("*" * 60)

            report = IntegrationTestReport(
                test_case_name=tc.name,
                provider=self.llm.provider,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            if "A" in pipelines:
                pipeline_a = self.run_pipeline_a(tc, verbose=verbose)
                report.pipelines.append(pipeline_a)

            if "B" in pipelines:
                pipeline_b = await self.run_pipeline_b(tc, verbose=verbose)
                report.pipelines.append(pipeline_b)

            if "C" in pipelines:
                pipeline_c = self.run_pipeline_c(tc, verbose=verbose)
                report.pipelines.append(pipeline_c)

            # 生成摘要
            report.summary = self._generate_summary(report)
            reports.append(report)

        return reports

    def _generate_summary(self, report: IntegrationTestReport) -> dict:
        """生成单个测试用例的摘要。"""
        summary = {
            "pipelines_ok": sum(1 for p in report.pipelines if p.status == "ok"),
            "pipelines_error": sum(1 for p in report.pipelines if p.status == "error"),
            "timing_comparison": {},
            "key_findings": [],
        }

        for p in report.pipelines:
            summary["timing_comparison"][p.pipeline_name] = f"{p.total_elapsed_ms:.0f}ms"

            # 收集关键发现
            for stage in p.stages:
                if "issues" in stage.details or "issues_count" in stage.details:
                    count = stage.details.get("issues_count", stage.details.get("final_issues", 0))
                    if isinstance(count, int) and count > 0:
                        summary["key_findings"].append(
                            f"[{p.pipeline_name}] {stage.stage_name}: {count} 个问题"
                        )
                if "conflict_count" in stage.details:
                    summary["key_findings"].append(
                        f"[{p.pipeline_name}] {stage.stage_name}: {stage.details['conflict_count']} 个冲突"
                    )

        return summary


# ================================================================
# 报告生成
# ================================================================


def generate_markdown_report(
    reports: list[IntegrationTestReport],
    total_elapsed_ms: float,
) -> str:
    """生成完整的集成测试 Markdown 报告。"""
    lines = [
        "# 多 Agent 代码审查系统 — 集成测试报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> LLM Provider: {reports[0].provider if reports else 'N/A'}",
        f"> 测试用例数: {len(reports)}",
        f"> 总耗时: {total_elapsed_ms:.0f}ms",
        "",
        "---",
        "",
        "## 测试概览",
        "",
    ]

    for i, report in enumerate(reports, 1):
        s = report.summary
        lines.append(f"### 测试 {i}: {report.test_case_name}")
        lines.append("")
        lines.append(
            f"Pipeline 通过: {s['pipelines_ok']}/{s['pipelines_ok'] + s['pipelines_error']}"
        )
        lines.append("")

        if s["timing_comparison"]:
            lines.append("| Pipeline | 总耗时 |")
            lines.append("|----------|--------|")
            for pipeline_name, time_str in s["timing_comparison"].items():
                lines.append(f"| {pipeline_name} | {time_str} |")
            lines.append("")

        if s["key_findings"]:
            lines.append("**关键发现:**")
            for finding in s["key_findings"]:
                lines.append(f"- {finding}")
            lines.append("")

    # 详细报告 — 每个 Pipeline
    lines.append("---")
    lines.append("")
    lines.append("## 详细报告")
    lines.append("")

    for report in reports:
        lines.append(f"### {report.test_case_name}")
        lines.append("")

        for p in report.pipelines:
            status_label = "[OK]" if p.status == "ok" else "[ERR]"
            lines.append(f"#### {status_label} {p.pipeline_name}")
            lines.append("")
            lines.append(f"总耗时: {p.total_elapsed_ms:.0f}ms")
            lines.append("")

            # 阶段详情
            if p.stages:
                lines.append("| 阶段 | 耗时 | 详情 |")
                lines.append("|------|------|------|")
                for stage in p.stages:
                    detail_str = ", ".join(
                        f"{k}={v}"
                        for k, v in stage.details.items()
                        if not isinstance(v, (list, dict))
                    )
                    lines.append(
                        f"| {stage.stage_name} | {stage.elapsed_ms:.0f}ms | {detail_str[:100]} |"
                    )
                lines.append("")

            # Agent 决策 (如果有)
            if p.agent_decisions:
                lines.append("**Agent 决策摘要:**")
                lines.append("```json")
                lines.append(
                    json.dumps(
                        {k: v for k, v in p.agent_decisions.items() if k != "messages_logged"},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                lines.append("```")
                lines.append("")

            # 报告片段
            if p.report:
                lines.append("<details>")
                lines.append("<summary>审查报告</summary>")
                lines.append("")
                lines.append(p.report[:1500])
                if len(p.report) > 1500:
                    lines.append(f"... (截断, 完整 {len(p.report)} 字符)")
                lines.append("")
                lines.append("</details>")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 测试结论")
    lines.append("")
    lines.append("### 通信开销对比")
    lines.append("")
    lines.append("Day 3 直接调用: 零通信开销, 函数调用即完成")
    lines.append("Day 4 消息驱动: 增加了消息序列化/反序列化、总线路由、超时管理的开销")
    lines.append("Day 5 冲突感知: 在 Day 3 基础上增加了 3 个额外 LLM 调用 (检测/仲裁/验证)")
    lines.append("")
    lines.append("### 系统可靠性")
    lines.append("")
    lines.append("- Day 4 的消息总线提供了重试和超时机制，Agent 单点故障不阻塞整体流程")
    lines.append("- Day 5 的冲突解决通过多重机制降低了单 Agent 误判的风险")
    lines.append("- 投票策略的代价是 N 倍 LLM 调用，适合对准确性要求高的场景")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# 主入口
# ================================================================


async def main():
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"LLM Provider: {provider}")
    print()

    # 选择测试用例
    # 默认跑全部 3 个, 如果只想快速验证可以只跑 test_cases[:1]
    test_cases = TEST_CASES  # 改这里控制范围

    # 选择 Pipeline
    # 全部跑: ["A", "B", "C"]
    # 快速模式: ["A"] (只跑 Day 3 直接调用)
    # 完整模式: ["A", "B", "C"]
    pipelines = ["A", "B", "C"]

    print("测试配置:")
    print(f"  测试用例: {len(test_cases)} 个")
    for tc in test_cases:
        print(f"    - {tc.name} ({tc.language}, {tc.code.count(chr(10)) + 1} 行)")
    print(f"  Pipeline: {pipelines}")
    print()

    # 初始化
    llm = LLMClient(provider=provider)
    runner = IntegrationTestRunner(llm)

    # 运行
    start_all = time.time()
    reports = await runner.run_all(
        test_cases=test_cases,
        pipelines=pipelines,
        verbose=True,
    )
    total_elapsed = (time.time() - start_all) * 1000

    # 生成报告
    report_md = generate_markdown_report(reports, total_elapsed)

    # 保存报告
    output_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "data",
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"integration_test_report_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # 打印摘要
    print()
    print("=" * 60)
    print("集成测试完成")
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.0f}ms ({total_elapsed/1000:.1f}s)")
    print(f"测试用例: {len(reports)} 个")
    print(f"报告已保存: {output_path}")
    print()

    # 打印对比表
    print("Pipeline 耗时对比:")
    print("-" * 60)
    for report in reports:
        print(f"  {report.test_case_name[:50]}:")
        for p in report.pipelines:
            status = "[OK]" if p.status == "ok" else "[ERR]"
            print(f"    {status} {p.pipeline_name}: {p.total_elapsed_ms:.0f}ms")
    print()


if __name__ == "__main__":
    asyncio.run(main())
