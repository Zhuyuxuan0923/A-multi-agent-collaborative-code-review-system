"""
Agent 通信协议演示 -- 用 MessageBus 实现 Day 1 的 4 种协作模式。

运行方式: python -m src.study_agent.agent.communication_demo

这个演示不调用 LLM(不消耗 token), 所有 Agent 返回预定义的回复,
重点展示消息在 Agent 之间如何流转。
"""

import asyncio
import time

from .message_bus import MessageBus, MessageTimeoutError
from .message_protocol import (
    AgentCapability,
    AgentInfo,
    AgentMessage,
    MessageType,
    Priority,
)

# ============================================================
# 工具: 启动 Agent 消息处理循环
# ============================================================


async def run_agent_loop(
    agent_id: str,
    bus: MessageBus,
    responses: dict[str, str],
    max_messages: int = 10,
) -> list[AgentMessage]:
    """
    启动一个 Agent 的消息处理循环。

    收到消息后, 从 responses 字典匹配关键词, 返回对应的回复。
    这个循环模拟了 Agent 的核心行为: 收消息 -> 处理 -> 回复。

    返回: 处理过的所有消息列表
    """
    processed = []
    for _ in range(max_messages):
        try:
            msg = await bus.receive(agent_id, timeout=3.0)
        except TimeoutError:
            break

        task_text = str(msg.payload.get("task", ""))
        answer = f"[{agent_id}] 收到但无匹配回复"

        for keyword, canned in responses.items():
            if keyword in task_text:
                answer = canned
                break

        bus.reply(msg, {"result": answer, "agent": agent_id})
        processed.append(msg)

    return processed


# ============================================================
# 演示 1: 顺序模式 -- A -> B -> C 管道
# ============================================================


async def demo_sequential():
    """顺序模式: 需求分析 -> 方案设计 -> 代码实现"""
    print("=" * 60)
    print("演示 1: 顺序模式 (Sequential)")
    print("=" * 60)
    print("管道: user -> analyst -> designer -> coder")
    print()

    bus = MessageBus()
    bus.register_agent(AgentInfo(agent_id="user_input", agent_type="orchestrator"))

    for aid, topic in [
        ("analyst", "pipeline.step1"),
        ("designer", "pipeline.step2"),
        ("coder", "pipeline.step3"),
    ]:
        bus.register_agent(
            AgentInfo(
                agent_id=aid,
                agent_type="worker",
                capabilities=[AgentCapability(name="step", topics=[topic])],
            )
        )

    responses = {
        "analyst": {"登录功能": "[analyst] 需求分析完成: 用户名+密码登录, 支持记住我"},
        "designer": {
            "需求分析完成": "[designer] 方案设计完成: JWT token + refresh token, session 存 Redis"
        },
        "coder": {"方案设计完成": "[coder] 代码已生成: auth.py, jwt_handler.py, session_store.py"},
    }

    # 启动 3 个 Agent 的处理循环 (后台任务)
    tasks = [
        asyncio.create_task(run_agent_loop(aid, bus, resp, max_messages=1))
        for aid, resp in responses.items()
    ]

    # 第一步: user -> analyst
    print("  [1] user -> analyst: 发送需求")
    req1 = AgentMessage(
        sender_id="user_input",
        receiver_id="analyst",
        message_type=MessageType.TASK_ASSIGNMENT,
        topic="pipeline.step1",
        payload={"task": "实现用户登录功能", "step": 1},
    )
    reply1 = await bus.request(req1, timeout=5.0)
    result1 = reply1.payload.get("result", "")
    print(f"  [1] analyst -> user: {result1}")

    # 第二步: analyst 结果 -> designer
    print("  [2] user -> designer: 传递需求分析结果")
    req2 = AgentMessage(
        sender_id="user_input",
        receiver_id="designer",
        message_type=MessageType.TASK_ASSIGNMENT,
        topic="pipeline.step2",
        payload={"task": result1, "step": 2},
    )
    reply2 = await bus.request(req2, timeout=5.0)
    result2 = reply2.payload.get("result", "")
    print(f"  [2] designer -> user: {result2}")

    # 第三步: designer 结果 -> coder
    print("  [3] user -> coder: 传递设计方案")
    req3 = AgentMessage(
        sender_id="user_input",
        receiver_id="coder",
        message_type=MessageType.TASK_ASSIGNMENT,
        topic="pipeline.step3",
        payload={"task": result2, "step": 3},
    )
    reply3 = await bus.request(req3, timeout=5.0)
    result3 = reply3.payload.get("result", "")
    print(f"  [3] coder -> user: {result3}")

    await asyncio.gather(*tasks)
    print("\n  管道执行完成: user -> analyst -> designer -> coder")
    print("  每步的输出 = 下一步的输入, 单向不可逆")
    print(f"  消息历史: {len(bus._history)} 条")


# ============================================================
# 演示 2: 广播模式 -- 多角度并行审查
# ============================================================


async def demo_broadcast():
    """广播模式: 代码同时发给 3 个审查专家, 并行审查"""
    print("\n" + "=" * 60)
    print("演示 2: 广播模式 (Broadcast)")
    print("=" * 60)
    print("场景: 代码同时发给安全/性能/风格三位专家, 并行审查")
    print()

    bus = MessageBus()
    bus.register_agent(AgentInfo(agent_id="orchestrator", agent_type="orchestrator"))

    experts = {
        "security_expert": {
            "审查": "[security_expert] 发现 SQL 注入风险在第 3 行, 建议使用参数化查询",
        },
        "performance_expert": {
            "审查": "[performance_expert] 第 4-5 行循环内查询数据库, 建议批量查询",
        },
        "style_expert": {
            "审查": "[style_expert] 函数名 getUserData 应改为 get_user_data 遵循 PEP 8",
        },
    }

    for aid, resp in experts.items():
        bus.register_agent(
            AgentInfo(
                agent_id=aid,
                agent_type="worker",
                capabilities=[AgentCapability(name="code_review", topics=["code_review"])],
            )
        )
        bus.subscribe(aid, "code_review")

    # 启动 3 个 Agent 的后台处理循环
    agent_tasks = [
        asyncio.create_task(run_agent_loop(aid, bus, resp, max_messages=1))
        for aid, resp in experts.items()
    ]

    code_snippet = """
def getUserData(id):
    query = "SELECT * FROM users WHERE id=" + id
    for user in getUsers():
        db.execute("UPDATE users SET last_login=NOW() WHERE id=" + user.id)
    return db.fetch()
"""

    # 使用 broadcast_and_collect 并行发送+收集
    # 内部会对每个订阅者发送消息并等待回复
    broadcast_msg = AgentMessage(
        sender_id="orchestrator",
        topic="code_review",
        message_type=MessageType.BROADCAST,
        payload={"task": "审查以下代码", "code": code_snippet},
        priority=Priority.HIGH,
    )

    print("  [发送] 向主题 'code_review' 广播审查请求")
    print(f"  [订阅者] {bus.get_subscribers('code_review')}")

    start = time.time()
    results = await bus.broadcast_and_collect(broadcast_msg, timeout=5.0)
    elapsed = (time.time() - start) * 1000

    # 等待 agent 任务完成
    await asyncio.gather(*agent_tasks)

    print(f"\n  [结果] 收到 {len(results)} 位专家的回复 (耗时 {elapsed:.0f}ms):")
    for agent_id, reply in results.items():
        if reply:
            print(f"    {reply.payload.get('result', '无结果')}")
        else:
            print(f"    [{agent_id}] 超时, 无回复")

    print("\n  [关键] 3 个专家并行执行, 总耗时 ~ 最慢那个 (不是三者之和)")
    print("  [关键] 广播模式适合: 多角度独立审查、并行信息收集")


# ============================================================
# 演示 3: 辩论模式 -- 多轮互驳
# ============================================================


async def demo_debate():
    """辩论模式: 2 个架构师就 SQL vs NoSQL 进行 3 轮辩论"""
    print("\n" + "=" * 60)
    print("演示 3: 辩论模式 (Debate)")
    print("=" * 60)
    print("场景: 架构师 A 和 B 就 'SQL vs NoSQL' 进行 3 轮辩论")
    print()

    bus = MessageBus()
    bus.register_agent(AgentInfo(agent_id="moderator", agent_type="orchestrator"))

    stances = {
        "architect_a": [
            "[A 第1轮] SQL 优势: ACID 事务保证数据一致性, 适合金融场景",
            "[A 第2轮] 反驳: NoSQL 的最终一致性在支付场景不可接受",
            "[A 第3轮] 最终立场: 核心交易数据必须用 SQL, 日志/分析用 NoSQL",
        ],
        "architect_b": [
            "[B 第1轮] NoSQL 优势: 水平扩展简单, Schema 灵活, 快速迭代",
            "[B 第2轮] 反驳: 现代 SQL 也支持 JSON 列和水平分片",
            "[B 第3轮] 最终立场: 同意混合方案, 先 NoSQL 验证, 后期迁 SQL",
        ],
    }

    for aid in stances:
        bus.register_agent(
            AgentInfo(
                agent_id=aid,
                agent_type="worker",
                capabilities=[AgentCapability(name="debate", topics=["debate.sql_vs_nosql"])],
            )
        )
        bus.subscribe(aid, "debate.sql_vs_nosql")

    topic = "debate.sql_vs_nosql"
    question = "电商系统的订单数据应该用 SQL 还是 NoSQL?"

    for round_num, round_name in enumerate(["独立观点", "反驳对方", "最终立场"], 1):
        print(f"--- 第 {round_num} 轮: {round_name} ---")

        # 启动本轮的两个 Agent 处理循环
        agent_tasks = [
            asyncio.create_task(
                run_agent_loop(aid, bus, {question: stances[aid][round_num - 1]}, max_messages=1)
            )
            for aid in stances
        ]

        task_text = question if round_num == 1 else f"请给出第{round_num}轮观点"
        round_msg = AgentMessage(
            sender_id="moderator",
            topic=topic,
            message_type=MessageType.BROADCAST,
            payload={"task": question, "round": round_num},
        )

        bus.publish(round_msg)
        await asyncio.gather(*agent_tasks)

        # 从 moderator 的视角看结果
        for aid in stances:
            reply = bus.receive_nowait("moderator")
            if reply:
                print(f"  {reply.payload.get('result', '')}")

    print("\n  [关键] 每轮所有 Agent 看到上一轮所有人的输出 (全连接通信)")
    print("  [关键] 3 轮后达成共识: SQL + NoSQL 混合方案")
    print("  [关键] Token 消耗: 2 Agent x 3 轮 = 6 次 LLM 调用 (辩论最吃 Token)")


# ============================================================
# 演示 4: 层级模式 -- Orchestrator 分配+检查+汇总
# ============================================================


async def demo_hierarchical():
    """层级模式: Orchestrator 分解任务 -> 并行分配给 Worker -> 汇总"""
    print("\n" + "=" * 60)
    print("演示 4: 层级模式 (Hierarchical)")
    print("=" * 60)
    print("场景: Orchestrator 把'实现用户注册'分解为 3 个子任务")
    print()

    bus = MessageBus()
    bus.register_agent(AgentInfo(agent_id="orchestrator", agent_type="orchestrator"))

    worker_tasks_map = {
        "frontend_dev": {"注册页面": "[frontend_dev] 完成: 注册页面 UI (表单+验证)"},
        "backend_dev": {"注册 API": "[backend_dev] 完成: POST /register + bcrypt 加密"},
        "test_dev": {"测试": "[test_dev] 完成: 3 个测试用例 (正常/重名/弱密码)"},
    }

    subtasks = [
        ("frontend_dev", "注册页面 UI: 表单 + 验证"),
        ("backend_dev", "注册 API: POST /register + 密码加密"),
        ("test_dev", "测试: 正常注册 + 重复用户名 + 弱密码"),
    ]

    for aid, resp in worker_tasks_map.items():
        bus.register_agent(
            AgentInfo(
                agent_id=aid,
                agent_type="worker",
                capabilities=[
                    AgentCapability(name=resp.get("cap", "worker"), topics=[f"work.{aid}"])
                ],
            )
        )

    print("[Orchestrator] 收到任务: '实现用户注册功能'")
    print("[Orchestrator] 分解为 3 个子任务:")

    # 启动所有 Worker 的后台处理循环
    agent_tasks = []
    for aid, task_desc in subtasks:
        print(f"  -> 分配给 {aid}: {task_desc}")
        agent_tasks.append(
            asyncio.create_task(run_agent_loop(aid, bus, worker_tasks_map[aid], max_messages=1))
        )

    # 并行发送 request (等待每个 Worker 回复)
    requests = []
    for aid, task_desc in subtasks:
        msg = AgentMessage(
            sender_id="orchestrator",
            receiver_id=aid,
            message_type=MessageType.TASK_ASSIGNMENT,
            topic=f"work.{aid}",
            payload={"task": task_desc},
        )
        requests.append(bus.request(msg, timeout=5.0))

    print("\n[Orchestrator] 等待 Worker 结果...")
    start = time.time()
    replies = await asyncio.gather(*requests, return_exceptions=True)
    elapsed = (time.time() - start) * 1000

    for reply in replies:
        if isinstance(reply, Exception):
            print(f"  [超时/错误] {reply}")
        else:
            print(f"  [收到] {reply.payload.get('result', '无结果')}")

    await asyncio.gather(*agent_tasks)

    print(f"\n[Orchestrator] 汇总 (耗时 {elapsed:.0f}ms): 所有子任务完成, 注册功能就绪")
    print("  [关键] 子任务并行执行, Worker 之间互不知晓")
    print("  [关键] Orchestrator 是唯一决策者 (星型拓扑)")


# ============================================================
# 演示 5: 超时与重试
# ============================================================


async def demo_timeout_retry():
    """超时与重试机制演示"""
    print("\n" + "=" * 60)
    print("演示 5: 超时与重试 (Timeout & Retry)")
    print("=" * 60)

    bus = MessageBus()
    bus.register_agent(AgentInfo(agent_id="sender", agent_type="worker"))

    # 场景 1: 目标不存在
    print("\n[场景 1] 发给不存在的 Agent...")
    msg = AgentMessage(
        sender_id="sender",
        receiver_id="ghost",
        message_type=MessageType.QUERY,
        payload={"task": "ping"},
    )
    try:
        bus.send(msg)
        print("  [异常] 不应该成功")
    except Exception as e:
        print(f"  [OK] 预期错误: {type(e).__name__}")

    # 场景 2: 目标存在但不回复 (模拟超时)
    print("\n[场景 2] 目标存在但不回复 (超时 + 重试)...")
    bus.register_agent(AgentInfo(agent_id="slow_agent", agent_type="worker"))

    msg2 = AgentMessage(
        sender_id="sender",
        receiver_id="slow_agent",
        message_type=MessageType.QUERY,
        payload={"task": "ping"},
    )

    start = time.time()
    try:
        await bus.request(msg2, timeout=1.0, max_retries=2, retry_backoff=0.5)
        print("  [异常] 不应该收到回复")
    except MessageTimeoutError:
        elapsed = (time.time() - start) * 1000
        print(f"  [OK] 预期超时, 耗时 {elapsed:.0f}ms")
        print("  [OK] 内部: 3 次尝试 (1 次原始 + 2 次重试)")
        print("  [OK] 退避: 第1次重试等 0.5*2^0=0.5s, 第2次等 0.5*2^1=1.0s")

    # 场景 3: 正常请求-回复 (快速路径)
    print("\n[场景 3] 正常请求-回复...")
    bus.register_agent(AgentInfo(agent_id="fast_agent", agent_type="worker"))
    fast_task = asyncio.create_task(
        run_agent_loop("fast_agent", bus, {"ping": "[fast_agent] pong!"}, max_messages=1)
    )

    msg3 = AgentMessage(
        sender_id="sender",
        receiver_id="fast_agent",
        message_type=MessageType.QUERY,
        payload={"task": "ping"},
    )

    start = time.time()
    reply = await bus.request(msg3, timeout=5.0)
    elapsed = (time.time() - start) * 1000
    print(f"  [OK] 收到回复: {reply.payload.get('result')} (耗时 {elapsed:.0f}ms)")

    await fast_task

    print("\n  [关键] request() 的超时+重试防止无限等待")
    print("  [关键] 指数退避: wait = backoff * 2^attempt + 随机抖动")
    print("  [关键] 随机抖动防止多个 Agent 同时重试 (雪崩效应)")


# ============================================================
# 主入口
# ============================================================


async def main():
    print("Agent 通信协议演示 (MessageBus)")
    print("Day 1 的 4 种协作模式 x MessageBus 实现")
    print()
    print("消息流程图例:")
    print("  --task_assignment-->  分配任务")
    print("  --broadcast-->        广播")
    print("  --response-->         回复")
    print("  --query-->            查询")
    print()

    await demo_sequential()
    await demo_broadcast()
    await demo_debate()
    await demo_hierarchical()
    await demo_timeout_retry()

    print("\n" + "=" * 60)
    print("全部演示完成! 5 个场景覆盖了 Day 1 的 4 种模式 + 可靠性机制")
    print("=" * 60)
    print()
    print("通信模式速查:")
    print("  顺序模式:  send() 点对点, A->B->C 链式传递")
    print("  广播模式:  publish() 一发多收, broadcast_and_collect() 并行收集")
    print("  辩论模式:  publish() 多轮, 每轮所有人看到上轮输出")
    print("  层级模式:  request() 星型, Orchestrator 分配+汇总+把关")
    print()
    print("可靠性机制:")
    print("  超时控制:  每条 request 可设 timeout, 避免无限等待")
    print("  重试机制:  max_retries + 指数退避 + 随机抖动")
    print("  消息 TTL:  消息自带过期时间, 过期自动丢弃")
    print("  中间件链:  可插拔的日志/认证/限流拦截层")


if __name__ == "__main__":
    asyncio.run(main())
