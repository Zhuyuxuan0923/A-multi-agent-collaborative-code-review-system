"""
MessageBus 驱动的编排器 -- Day 4 核心交付物。

Day 3 的编排器直接调用 Agent 方法:
  result = reviewer.run(code)        # 函数调用, 紧耦合

Day 4 的编排器通过 MessageBus 发消息:
  await bus.request(task_message)     # 消息传递, 松耦合

松耦合的好处:
  1. Agent 可以独立部署 (不同进程/不同机器)
  2. 总线可以做日志/监控/限流
  3. 新增 Agent 不影响现有代码
  4. 超时和重试由总线统一处理

架构:
  BusOrchestrator
    |
    +-- 注册 3 个 BusAwareAgent 到 MessageBus
    +-- 分解任务 -> 并行发送 -> 收集结果 -> 验证 -> 重试? -> 汇总
    |
    +-- BusAwareAgent(Reviewer)   监听 "review.request" 主题
    +-- BusAwareAgent(Researcher) 监听 "research.request" 主题
    +-- BusAwareAgent(Reporter)   监听 "report.request" 主题
"""

import asyncio
import time

from .code_review_agents import (
    LLMClient,
    ReporterAgent,
    ResearcherAgent,
    ResearchResult,
    ReviewerAgent,
    ReviewResult,
)
from .message_bus import MessageBus, MessageTimeoutError
from .message_protocol import (
    AgentCapability,
    AgentInfo,
    AgentMessage,
    MessageType,
    Priority,
)

# ============================================================
# BusAwareAgent -- 把 Day 3 的 Agent 包装成消息驱动的 Agent
# ============================================================


class BusAwareAgent:
    """将 Day 3 的"纯函数式 Agent"包装为"消息驱动 Agent"。

    Day 3 的 Agent: agent.run(input) -> output  (函数调用)
    Day 4 的 Agent: 监听总线消息 -> 调用 agent.run() -> 通过总线回复

    这层包装器就是 Agent 的"网络层"——让它能通过消息总线收发任务。
    """

    def __init__(
        self,
        agent_id: str,
        role_name: str,
        base_agent,  # Day 3 的 ReviewerAgent / ResearcherAgent / ReporterAgent
        bus: MessageBus,
        listen_topic: str,  # 监听的请求主题
    ):
        self.agent_id = agent_id
        self.role_name = role_name
        self.base_agent = base_agent
        self.bus = bus
        self.listen_topic = listen_topic
        self.running = False
        self._task: asyncio.Task | None = None

        # 统计
        self.messages_processed = 0
        self.total_latency_ms = 0.0

    async def start(self):
        """启动 Agent: 注册到总线并开始监听消息。"""
        # 注册
        self.bus.register_agent(
            AgentInfo(
                agent_id=self.agent_id,
                agent_type="worker",
                capabilities=[
                    AgentCapability(
                        name=self.role_name,
                        topics=[self.listen_topic],
                    )
                ],
            )
        )
        self.bus.subscribe(self.agent_id, self.listen_topic)

        # 启动后台消息处理循环
        self.running = True
        self._task = asyncio.create_task(self._message_loop())

    async def stop(self):
        """停止 Agent。"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _message_loop(self):
        """后台消息处理循环 -- Agent 的"主循环"。

        持续从收件箱取消息, 调用底层 Agent 处理, 然后通过总线回复。

        关键: LLM 调用是同步阻塞的 (llm.chat()),
        必须用 asyncio.to_thread() 放到线程池执行,
        否则会阻塞整个事件循环, 导致其他 Agent 无法处理消息。
        """
        while self.running:
            try:
                msg = await self.bus.receive(self.agent_id, timeout=1.0)
            except TimeoutError:
                continue  # 没消息, 继续等

            start = time.time()
            self.messages_processed += 1

            try:
                msg_type = msg.message_type
                payload = msg.payload

                if msg_type in (MessageType.TASK_ASSIGNMENT, MessageType.QUERY):
                    # 将阻塞的 LLM 调用放到线程池, 不阻塞事件循环
                    result = await asyncio.to_thread(self._handle_task, payload)
                    self.bus.reply(msg, {"status": "ok", "result": result})
                else:
                    self.bus.reply(msg, {"status": "error", "error": f"未知消息类型: {msg_type}"})

            except Exception as e:
                self.bus.reply(msg, {"status": "error", "error": str(e)})

            elapsed = (time.time() - start) * 1000
            self.total_latency_ms += elapsed

    def _handle_task(self, payload: dict) -> dict:
        """根据 Agent 角色处理任务。子类可以覆写这个方法。"""
        # 默认: 调用底层 Agent 的 run 方法
        code = payload.get("code", "")
        language = payload.get("language", "unknown")

        if self.role_name == "reviewer":
            result: ReviewResult = self.base_agent.run(code, language)
            return {
                "summary": result.summary,
                "score": result.score,
                "issues": result.issues,
                "raw_json": result.raw_json,
            }
        elif self.role_name == "researcher":
            result: ResearchResult = self.base_agent.run(code, language)
            return {
                "technologies": result.technologies,
                "best_practices": result.best_practices,
                "common_pitfalls": result.common_pitfalls,
                "recommendations": result.recommendations,
                "references": result.references,
                "raw_json": result.raw_json,
            }
        elif self.role_name == "reporter":
            review_data = payload.get("review_result", {})
            research_data = payload.get("research_result", {})
            review = (
                ReviewResult(**review_data) if review_data else ReviewResult(summary="", score=0)
            )
            research = ResearchResult(**research_data) if research_data else ResearchResult()
            report_md = self.base_agent.run(code, language, review, research)
            return {"report": report_md}
        else:
            return {"error": f"未知角色: {self.role_name}"}

    @property
    def stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role_name,
            "messages_processed": self.messages_processed,
            "avg_latency_ms": (
                self.total_latency_ms / self.messages_processed
                if self.messages_processed > 0
                else 0
            ),
        }


# ============================================================
# BusOrchestrator -- 消息驱动的编排器 (Day 4 核心)
# ============================================================


class BusOrchestrator:
    """基于 MessageBus 的代码审查编排器。

    与 Day 3 的 CodeReviewOrchestrator 的区别:
      Day 3: 直接调用 agent.run() -> 同步, 紧耦合
      Day 4: 通过 bus.request() 发消息 -> 异步, 松耦合

    编排流程 (层级模式):
      1. 启动所有 Agent (注册 + 开始监听)
      2. 并行发送审查任务给 Reviewer + Researcher
      3. 收集结果, 验证, 必要时重试
      4. 发送汇总任务给 Reporter
      5. 返回最终 Markdown 报告
    """

    # 消息主题常量
    TOPIC_REVIEW = "code_review.review"
    TOPIC_RESEARCH = "code_review.research"
    TOPIC_REPORT = "code_review.report"

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.bus = MessageBus()

        # 创建底层 Agent (Day 3 的纯函数 Agent)
        raw_reviewer = ReviewerAgent(llm)
        raw_researcher = ResearcherAgent(llm)
        raw_reporter = ReporterAgent(llm)

        # 包装为总线感知 Agent
        self.reviewer_bus = BusAwareAgent(
            "reviewer_1", "reviewer", raw_reviewer, self.bus, self.TOPIC_REVIEW
        )
        self.researcher_bus = BusAwareAgent(
            "researcher_1", "researcher", raw_researcher, self.bus, self.TOPIC_RESEARCH
        )
        self.reporter_bus = BusAwareAgent(
            "reporter_1", "reporter", raw_reporter, self.bus, self.TOPIC_REPORT
        )

    # ================================================================
    # 编排逻辑 (Day 4 重点)
    # ================================================================

    async def review(
        self,
        code: str,
        language: str = "unknown",
        max_retries: int = 1,
        request_timeout: float = 60.0,
        verbose: bool = True,
    ) -> str:
        """执行完整的代码审查编排流程。

        这是 Day 4 的核心 -- 所有 Agent 协调都通过 MessageBus 消息完成。

        参数:
            code: 源代码
            language: 编程语言
            max_retries: 单个 Agent 失败后的最大重试次数
            request_timeout: 单次 request 超时 (秒)
            verbose: 是否打印编排过程

        返回:
            Markdown 审查报告
        """
        # ---- 第 0 步: 启动所有 Agent ----
        if verbose:
            print("=" * 60)
            print("BusOrchestrator 启动")
            print(f"语言: {language}, 代码: {len(code)} 字符")
            print("=" * 60)
            print()
            print("[编排] 启动 3 个 Agent...")

        await self.reviewer_bus.start()
        await self.researcher_bus.start()
        await self.reporter_bus.start()

        if verbose:
            print(f"  [OK] Reviewer 已启动, 监听 '{self.TOPIC_REVIEW}'")
            print(f"  [OK] Researcher 已启动, 监听 '{self.TOPIC_RESEARCH}'")
            print(f"  [OK] Reporter 已启动, 监听 '{self.TOPIC_REPORT}'")
            print(f"  [总线状态] {self.bus.stats()['agents_registered']} 个 Agent 在线")
            print()

        try:
            # ---- 第 1 步: 并行分发 Reviewer + Researcher (广播模式) ----
            review_result, research_result = await self._dispatch_parallel(
                code, language, max_retries, request_timeout, verbose
            )

            # ---- 第 2 步: 决策 -- 检查结果质量 ----
            review_result, research_result = await self._validate_and_retry(
                code,
                language,
                review_result,
                research_result,
                max_retries,
                request_timeout,
                verbose,
            )

            # ---- 第 3 步: Reporter 汇总 (顺序模式) ----
            report = await self._dispatch_reporter(
                code, language, review_result, research_result, request_timeout, verbose
            )

        finally:
            # ---- 第 4 步: 停止所有 Agent ----
            if verbose:
                print()
                print("[编排] 停止所有 Agent...")
            await self.reviewer_bus.stop()
            await self.researcher_bus.stop()
            await self.reporter_bus.stop()
            if verbose:
                print("  [OK] 所有 Agent 已停止")

        # ---- 打印统计 ----
        if verbose:
            print()
            print("[编排] Agent 统计:")
            for agent in [self.reviewer_bus, self.researcher_bus, self.reporter_bus]:
                s = agent.stats
                print(
                    f"  {s['role']}: {s['messages_processed']} 条消息, 平均 {s['avg_latency_ms']:.0f}ms"
                )

        return report

    # ----------------------------------------------------------------
    # 编排子步骤
    # ----------------------------------------------------------------

    async def _dispatch_parallel(
        self, code: str, language: str, max_retries: int, timeout: float, verbose: bool
    ) -> tuple[dict, dict]:
        """[编排逻辑] 并行分发任务给 Reviewer 和 Researcher。

        这是层级模式的核心: Orchestrator -> [Reviewer, Researcher] 并行。
        """

        if verbose:
            print("[编排] 阶段 1/3: 并行分发 -> Reviewer + Researcher")
            print("-" * 40)

        # 构造审查请求消息
        review_msg = AgentMessage(
            sender_id="orchestrator",
            receiver_id="reviewer_1",
            message_type=MessageType.TASK_ASSIGNMENT,
            topic=self.TOPIC_REVIEW,
            priority=Priority.HIGH,
            payload={"code": code, "language": language, "task_type": "review"},
            ttl=timeout * 2,
        )

        # 构造研究请求消息
        research_msg = AgentMessage(
            sender_id="orchestrator",
            receiver_id="researcher_1",
            message_type=MessageType.TASK_ASSIGNMENT,
            topic=self.TOPIC_RESEARCH,
            priority=Priority.NORMAL,
            payload={"code": code, "language": language, "task_type": "research"},
            ttl=timeout * 2,
        )

        if verbose:
            print(f"  [发送] 审查请求 -> reviewer_1 (topic: {self.TOPIC_REVIEW})")
            print(f"  [发送] 研究请求 -> researcher_1 (topic: {self.TOPIC_RESEARCH})")

        # 并行发送, 等待回复
        start = time.time()
        review_task = self.bus.request(review_msg, timeout=timeout, max_retries=0)
        research_task = self.bus.request(research_msg, timeout=timeout, max_retries=0)

        review_reply, research_reply = await asyncio.gather(
            review_task, research_task, return_exceptions=True
        )
        elapsed = (time.time() - start) * 1000

        # 处理 Reviewer 结果
        if isinstance(review_reply, Exception):
            if verbose:
                print(f"  [ERR] Reviewer 失败: {review_reply}")
            review_result = {"error": str(review_reply), "issues": []}
        else:
            review_result = review_reply.payload.get("result", {})
            if verbose:
                score = review_result.get("score", "?")
                issues = len(review_result.get("issues", []))
                print(f"  [OK] Reviewer 回复: 评分 {score}/10, {issues} 个问题")

        # 处理 Researcher 结果
        if isinstance(research_reply, Exception):
            if verbose:
                print(f"  [ERR] Researcher 失败: {research_reply}")
            research_result = {"error": str(research_reply), "best_practices": []}
        else:
            research_result = research_reply.payload.get("result", {})
            if verbose:
                techs = research_result.get("technologies", [])
                practices = len(research_result.get("best_practices", []))
                print(f"  [OK] Researcher 回复: 技术 {techs}, {practices} 条最佳实践")

        if verbose:
            print(f"  [耗时] 并行阶段: {elapsed:.0f}ms")
            print()

        return review_result, research_result

    async def _validate_and_retry(
        self,
        code: str,
        language: str,
        review_result: dict,
        research_result: dict,
        max_retries: int,
        timeout: float,
        verbose: bool,
    ) -> tuple[dict, dict]:
        """[编排逻辑] 验证结果质量, 必要时重试。

        决策点:
          1. Reviewer 返回了空 issues 列表? -> 可能是没认真审查, 重试
          2. Reviewer 返回了 error? -> 重试
          3. Researcher 返回了空 best_practices? -> 重试并给更具体的指令
        """

        # --- 检查 Reviewer 结果 ---
        should_retry_review = False
        retry_reason = ""

        if "error" in review_result:
            should_retry_review = True
            retry_reason = f"Reviewer 报错: {review_result['error']}"
        elif len(review_result.get("issues", [])) == 0:
            should_retry_review = True
            retry_reason = "Reviewer 未发现任何问题 (可能是审查不充分)"

        if should_retry_review and max_retries > 0:
            if verbose:
                print("[编排] 阶段 2a/3: 重试 Reviewer")
                print(f"  原因: {retry_reason}")
                print(f"  剩余重试次数: {max_retries}")

            retry_msg = AgentMessage(
                sender_id="orchestrator",
                receiver_id="reviewer_1",
                message_type=MessageType.TASK_ASSIGNMENT,
                topic=self.TOPIC_REVIEW,
                priority=Priority.CRITICAL,
                payload={
                    "code": code,
                    "language": language,
                    "task_type": "review_retry",
                    "hint": "请仔细逐行审查。之前的审查可能不充分, 请重新检查。",
                },
            )

            try:
                retry_reply = await self.bus.request(retry_msg, timeout=timeout, max_retries=0)
                review_result = retry_reply.payload.get("result", {})
                if verbose:
                    print(
                        f"  [OK] 重试成功: 评分 {review_result.get('score', '?')}/10, {len(review_result.get('issues', []))} 个问题"
                    )
            except MessageTimeoutError:
                if verbose:
                    print("  [ERR] 重试超时, 使用原始结果")

        # --- 检查 Researcher 结果 ---
        should_retry_research = False
        retry_reason_research = ""

        if "error" in research_result:
            should_retry_research = True
            retry_reason_research = f"Researcher 报错: {research_result['error']}"
        elif len(research_result.get("best_practices", [])) == 0:
            should_retry_research = True
            retry_reason_research = "Researcher 未提供最佳实践"

        if should_retry_research and max_retries > 0:
            if verbose:
                print("[编排] 阶段 2b/3: 重试 Researcher")
                print(f"  原因: {retry_reason_research}")

            retry_msg = AgentMessage(
                sender_id="orchestrator",
                receiver_id="researcher_1",
                message_type=MessageType.TASK_ASSIGNMENT,
                topic=self.TOPIC_RESEARCH,
                priority=Priority.HIGH,
                payload={
                    "code": code,
                    "language": language,
                    "task_type": "research_retry",
                    "hint": f"请针对 {language} 代码提供至少 3 条最佳实践和常见陷阱。之前的回应可能不充分。",
                },
            )

            try:
                retry_reply = await self.bus.request(retry_msg, timeout=timeout, max_retries=0)
                research_result = retry_reply.payload.get("result", {})
                if verbose:
                    print(
                        f"  [OK] 重试成功: {len(research_result.get('best_practices', []))} 条实践"
                    )
            except MessageTimeoutError:
                if verbose:
                    print("  [ERR] 重试超时, 使用原始结果")

        if not should_retry_review and not should_retry_research and verbose:
            print("[编排] 阶段 2/3: 结果验证通过, 无需重试")
            print()

        return review_result, research_result

    async def _dispatch_reporter(
        self,
        code: str,
        language: str,
        review_result: dict,
        research_result: dict,
        timeout: float,
        verbose: bool,
    ) -> str:
        """[编排逻辑] 将审查和研究结果发送给 Reporter 汇总。"""

        if verbose:
            print("[编排] 阶段 3/3: Reporter 汇总生成报告")
            print("-" * 40)

        report_msg = AgentMessage(
            sender_id="orchestrator",
            receiver_id="reporter_1",
            message_type=MessageType.TASK_ASSIGNMENT,
            topic=self.TOPIC_REPORT,
            priority=Priority.HIGH,
            payload={
                "code": code,
                "language": language,
                "task_type": "report",
                "review_result": review_result,
                "research_result": research_result,
            },
        )

        if verbose:
            print("  [发送] 汇总请求 -> reporter_1 (含 Reviewer + Researcher 结果)")

        start = time.time()
        try:
            reply = await self.bus.request(report_msg, timeout=timeout, max_retries=1)
            report_data = reply.payload.get("result", {})
            report_md = report_data.get("report", "")
            elapsed = (time.time() - start) * 1000
            if verbose:
                print(f"  [OK] Reporter 完成 ({len(report_md)} 字符, {elapsed:.0f}ms)")
            return report_md
        except MessageTimeoutError as e:
            elapsed = (time.time() - start) * 1000
            if verbose:
                print(f"  [ERR] Reporter 超时 ({elapsed:.0f}ms): {e}")
            # 降级: 返回原始结果, 不做格式化
            return self._fallback_report(code, language, review_result, research_result)

    # ----------------------------------------------------------------
    # 降级处理
    # ----------------------------------------------------------------

    def _fallback_report(self, code: str, language: str, review: dict, research: dict) -> str:
        """Reporter 失败时的降级报告 -- 不做格式化, 直接输出原始结果。"""
        lines = [
            "# 代码审查报告 (降级版本)",
            "",
            "> Reporter 超时, 以下为 Reviewer 和 Researcher 的原始输出。",
            "",
            "## Reviewer 原始结果",
            "",
            f"评分: {review.get('score', 'N/A')}/10",
            f"摘要: {review.get('summary', 'N/A')}",
            f"问题数: {len(review.get('issues', []))}",
            "",
            "## Researcher 原始结果",
            "",
            f"技术栈: {research.get('technologies', [])}",
            f"最佳实践: {len(research.get('best_practices', []))} 条",
            f"常见陷阱: {len(research.get('common_pitfalls', []))} 条",
            f"参考资料: {research.get('references', [])}",
            "",
            "## 原始代码",
            f"```{language}",
            code,
            "```",
        ]
        return "\n".join(lines)


# ============================================================
# 对比: Day 3 vs Day 4 的调用方式
# ============================================================


def compare_day3_vs_day4():
    """打印 Day 3 和 Day 4 编排方式的对比说明。"""
    print(
        """
+-- Day 3 编排方式 (直接调用) --+
|
|  orchestrator = CodeReviewOrchestrator(llm)
|  report = orchestrator.review(code, language)
|
|  内部:
|    review = self.reviewer.run(code, language)       # 同步函数调用
|    research = self.researcher.run(code, language)   # 同步函数调用
|    report = self.reporter.run(code, language, review, research)
|
|  特点: 简单直接, 但紧耦合
|
+-- Day 4 编排方式 (消息驱动) --+
|
|  orchestrator = BusOrchestrator(llm)
|  report = await orchestrator.review(code, language)
|
|  内部:
|    await reviewer_bus.start()    # Agent 注册 + 启动消息循环
|    msg = AgentMessage(...)       # 构造消息
|    reply = await bus.request(msg, timeout=60)  # 通过总线发送, 等待回复
|
|  特点:
|    1. Agent 通过消息通信 (松耦合)
|    2. 总线统一管理超时/重试/日志
|    3. Agent 可以独立部署
|    4. 编排逻辑集中在 BusOrchestrator.review() 方法中
"""
    )
