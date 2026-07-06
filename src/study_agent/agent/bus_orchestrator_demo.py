"""
MessageBus 驱动的代码审查编排器演示 -- Day 4。

运行方式: python -m src.study_agent.agent.bus_orchestrator_demo

与 Day 3 的对比:
  Day 3: 编排器直接调用 agent.run() (函数调用)
  Day 4: 编排器通过 MessageBus 发消息驱动 Agent (消息传递)

这个演示展示:
  1. Agent 注册到 MessageBus 并启动后台消息循环
  2. Orchestrator 通过 bus.request() 并行分发任务
  3. 结果验证与重试决策
  4. Reporter 汇总
  5. 消息流转日志
"""

import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from study_agent.agent.bus_orchestrator import BusOrchestrator, compare_day3_vs_day4
from study_agent.agent.message_protocol import MessageType
from study_agent.llm.client import LLMClient

# 与 Day 3 相同的测试代码 (用于对比)
SAMPLE_CODE = """
import sqlite3
import os

ADMIN_PASSWORD = "admin123"

def getUserData(userId):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE id = " + userId
    result = conn.execute(query).fetchall()
    return result

def getUsersWithOrders():
    users = getUserData("1")
    result = []
    for user in users:
        orders = db.execute("SELECT * FROM orders WHERE user_id = " + str(user[0]))
        result.append({"user": user, "orders": orders})
    return result

def saveFile(filename, content):
    f = open("/tmp/" + filename, "w")
    f.write(content)

def processRequest(req):
    data = req["data"]
    value = data.upper()
    return {"status": "ok", "value": value}
"""


async def main():
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"LLM Provider: {provider}")
    print()

    # 打印 Day 3 vs Day 4 对比
    compare_day3_vs_day4()

    # 初始化
    llm = LLMClient(provider=provider)
    orchestrator = BusOrchestrator(llm)

    # 注册总线消息日志中间件 -- 可视化消息流转
    def log_msg(msg):
        if msg.message_type != MessageType.HEARTBEAT:
            receiver = msg.receiver_id or f"topic:{msg.topic}"
            payload_preview = str(msg.payload.get("task_type", msg.payload.get("task", "")))[:40]
            print(
                f"  [BUS] {msg.sender_id} --{msg.message_type.value}--> {receiver} | {payload_preview}"
            )
        return msg

    orchestrator.bus.use(log_msg)

    code = SAMPLE_CODE.strip()
    language = "python"

    print("=" * 60)
    print("待审查代码 (与 Day 3 相同):")
    print("-" * 40)
    for i, line in enumerate(code.split("\n"), 1):
        print(f"  {i:2d} | {line}")
    print("-" * 40)
    print()

    # 执行消息驱动的审查编排
    start = time.time()
    report = await orchestrator.review(
        code,
        language,
        max_retries=1,
        request_timeout=90.0,
        verbose=True,
    )
    total_elapsed = (time.time() - start) * 1000

    print()
    print("=" * 60)
    print("审查报告 (MessageBus 驱动版)")
    print("=" * 60)
    print(report)

    # 保存报告
    output_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "data",
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"code_review_bus_report_{timestamp}.md")

    full_report = f"""\
# 代码审查报告 (MessageBus 驱动版)

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 审查语言: {language}
> 审查引擎: Multi-Agent via MessageBus (Reviewer + Researcher + Reporter)
> LLM Provider: {provider}
> 编排方式: 消息驱动 (bus.request/reply)
> 总耗时: {total_elapsed:.0f}ms

---

{report}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n报告已保存到: {output_path}")
    print(f"总耗时: {total_elapsed:.0f}ms")

    print()
    print("Day 3 vs Day 4 关键区别:")
    print("  Day 3: orchestrator.reviewer.run(code)      -- 直接函数调用")
    print("  Day 4: await bus.request(msg, timeout=90)   -- 消息驱动, 带超时/重试")
    print("  Day 4: Agent 通过 MessageBus 通信, 编排器不直接接触 Agent 实例")
    print(f"  Day 4: 总线可记录所有消息流转 (今天记录了 {len(orchestrator.bus._history)} 条)")


if __name__ == "__main__":
    asyncio.run(main())
