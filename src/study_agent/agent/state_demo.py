"""Agent 状态管理 —— 演示脚本。

演示内容：
  1. 创建会话 + 状态追踪（Agent 从 IDLE → THINKING → COMPLETED）
  2. 多轮对话中的状态保持（上下文一致性）
  3. 并发会话隔离（两个用户同时用，状态互不干扰）
  4. 状态序列化/反序列化（Pydantic 免费提供）
  5. 会话过期与自动清理
"""

from study_agent.agent.state import (
    AgentState,
    AgentStatus,
    SessionManager,
    ToolCallRecord,
)


def demo_basic_state():
    """演示 1：基本状态管理 —— Agent 执行任务的状态变化。"""
    print("=" * 55)
    print("演示 1：Agent 状态生命周期")
    print("=" * 55)

    # 模拟一次 Agent 任务执行的状态变化
    state = AgentState(task_description="帮用户调试 Python 缩进错误")

    print(f"创建会话: {state.session_id}")
    print(f"初始状态: {state.status.value}")
    print(f"任务描述: {state.task_description}")
    print()

    # 用户发消息 → Agent 开始思考
    state.add_message("user", "我的 Python 代码报 IndentationError 了")
    state.set_status(AgentStatus.THINKING)
    print(f"用户发消息后 → status={state.status.value}")

    # Agent 制定计划
    step1 = state.add_step("分析错误信息，定位报错行号")
    step2 = state.add_step("检查缩进是否混用 Tab 和空格")
    step3 = state.add_step("给出修复建议")
    print(f"制定了 {len(state.steps)} 个步骤:")
    for s in state.steps:
        print(f"  第{s.step_index}步: {s.description} [{s.status}]")
    print()

    # 执行步骤 1
    state.mark_step_done(1, "报错在第 12 行，缩进多了 2 个空格")
    state.set_status(AgentStatus.ACTING)
    print(f"步骤1完成 → status={state.status.value}, result={state.steps[0].result}")

    # 模拟一次工具调用
    tool_record = ToolCallRecord(
        tool_name="read_file",
        arguments={"file_path": "main.py", "line": 12},
        result="    print('hello')  ← 这里多了 2 个空格",
        duration_ms=120.5,
    )
    state.add_tool_call(tool_record)
    print(f"工具调用: {tool_record.tool_name}() 耗时 {tool_record.duration_ms}ms")

    # 完成
    state.mark_step_done(2, "确认：Tab 和空格没有混用")
    state.mark_step_done(3, "删掉多余的 2 个空格即可")
    state.set_status(AgentStatus.COMPLETED)
    state.add_message("assistant", "第 12 行缩进多了 2 个空格，删掉就好。")

    print(f"任务完成 → status={state.status.value}")
    print(f"对话轮数: {state.round_count}")
    print(f"工具调用次数: {state.total_tool_calls}")
    print()
    print(state.summary())


def demo_multi_turn():
    """演示 2：多轮对话中状态保持一致。"""
    print("\n" + "=" * 55)
    print("演示 2：多轮对话中的上下文一致性")
    print("=" * 55)

    mgr = SessionManager()
    sid = mgr.create_session(task="帮用户规划旅行")

    # 第 1 轮
    state = mgr.get_session(sid)
    assert state is not None
    state.add_message("user", "我想去成都玩 3 天，推荐行程")
    state.add_message("assistant", "好的！第1天建议去宽窄巷子和锦里，第2天去大熊猫基地...")
    state.metadata["destination"] = "成都"
    state.metadata["days"] = 3
    mgr.update_session(sid, state)
    print(f"第1轮: 目的地={state.metadata['destination']}, 天数={state.metadata['days']}")

    # 第 2 轮——Agent 必须记得之前聊的是成都
    state2 = mgr.get_session(sid)
    assert state2 is not None
    print(f"第2轮开始时 messages 数量={len(state2.messages)}")
    print(f"  -> 记得目的地是: {state2.metadata.get('destination')}")
    print(f"  -> 记得天数是: {state2.metadata.get('days')}")

    state2.add_message("user", "第二天能加上都江堰吗？")
    state2.add_message("assistant", "可以的！第2天上午熊猫基地，下午去都江堰，晚上回市区。")
    mgr.update_session(sid, state2)
    print(f"第2轮结束: messages={len(state2.messages)}, rounds={state2.round_count}")
    print(f"会话状态摘要: {state2.summary()}")

    mgr.delete_session(sid)


def demo_concurrent_sessions():
    """演示 3：并发会话完全隔离。"""
    print("\n" + "=" * 55)
    print("演示 3：并发会话隔离")
    print("=" * 55)

    mgr = SessionManager()

    # 用户 Alice —— 在学 Python
    alice_id = mgr.create_session(task="Alice 学 Python 基础")
    alice = mgr.get_session(alice_id)
    assert alice is not None
    alice.add_message("user", "Python 的 list 和 tuple 有什么区别？")
    alice.add_message("assistant", "list 可变，tuple 不可变...")
    alice.metadata["user_name"] = "Alice"
    alice.metadata["topic"] = "Python"
    mgr.update_session(alice_id, alice)

    # 用户 Bob —— 在写 SQL
    bob_id = mgr.create_session(task="Bob 优化 SQL 查询")
    bob = mgr.get_session(bob_id)
    assert bob is not None
    bob.add_message("user", "SELECT 很慢怎么办？")
    bob.add_message("assistant", "先 EXPLAIN 看执行计划...")
    bob.metadata["user_name"] = "Bob"
    bob.metadata["topic"] = "SQL"
    mgr.update_session(bob_id, bob)

    # 验证隔离 —— Bob 的话题不会跑到 Alice 那去
    alice_check = mgr.get_session(alice_id)
    bob_check = mgr.get_session(bob_id)
    assert alice_check is not None
    assert bob_check is not None

    print(f"Alice 的会话: topic={alice_check.metadata['topic']}, rounds={alice_check.round_count}")
    print(f"Bob   的会话: topic={bob_check.metadata['topic']}, rounds={bob_check.round_count}")

    # 列出所有会话
    print(f"\n当前活跃会话数: {mgr.count()}")
    for info in mgr.list_sessions():
        print(
            f"  {info['session_id']}: [{info['status']}] {info['task'][:30]} ({info['rounds']}轮)"
        )

    # 删除 Alice 的会话
    mgr.delete_session(alice_id)
    print(f"\n删除 Alice 后，剩余会话数: {mgr.count()}")

    mgr.delete_session(bob_id)


def demo_serialization():
    """演示 4：Pydantic 状态序列化/反序列化。"""
    print("\n" + "=" * 55)
    print("演示 4：状态序列化与恢复")
    print("=" * 55)

    # 创建一个带完整信息的状态
    state = AgentState(task_description="代码审查任务")
    state.add_message("user", "帮我审查这段代码")
    state.add_message("assistant", "好的，请把代码发给我")
    state.add_step("读取代码文件")
    state.mark_step_done(1, "已读取 main.py")

    tool = ToolCallRecord(
        tool_name="read_file",
        arguments={"file_path": "main.py"},
        result="def foo():\n    pass",
    )
    state.add_tool_call(tool)
    state.metadata["language"] = "Python"
    state.set_status(AgentStatus.WAITING_USER)

    # 序列化 → JSON 字符串
    json_str = state.model_dump_json(indent=2)
    print(f"序列化后的 JSON ({len(json_str)} 字符):")
    print(json_str[:400])
    print("  ...")
    print()

    # 反序列化 → 恢复 AgentState 对象
    restored = AgentState.model_validate_json(json_str)
    print(f"恢复后的状态: {restored.summary()}")
    print(f"  messages 数: {len(restored.messages)}")
    print(f"  steps 数: {len(restored.steps)}")
    print(f"  tool_calls 数: {len(restored.tool_calls)}")
    print(f"  metadata: {restored.metadata}")

    # 验证数据完整性
    assert restored.session_id == state.session_id
    assert restored.messages == state.messages
    assert restored.metadata == state.metadata
    print("\n[OK] 序列化往返后数据完整性验证通过")


def demo_session_expiry():
    """演示 5：会话过期自动清理。"""
    print("\n" + "=" * 55)
    print("演示 5：会话过期与清理")
    print("=" * 55)

    # 创建一个短 TTL 的 SessionManager（2 秒过期）
    mgr = SessionManager(session_ttl_seconds=2)

    sid1 = mgr.create_session(task="短期会话")
    sid2 = mgr.create_session(task="也短期")

    print("创建 2 个会话，TTL=2秒")
    print(f"  会话1: {sid1}")
    print(f"  会话2: {sid2}")
    print(f"  当前活跃数: {mgr.count()}")

    # 等 3 秒让会话过期
    import time

    print("\n等待 3 秒让会话过期...")
    time.sleep(3)

    # 再次获取 —— 应该返回 None（已过期被清理）
    expired1 = mgr.get_session(sid1)
    expired2 = mgr.get_session(sid2)
    print(f"过期后获取会话1: {expired1}")  # None
    print(f"过期后获取会话2: {expired2}")  # None
    print(f"过期后活跃数: {mgr.count()}")  # 0


def demo_get_or_create():
    """演示 6：get_or_create 的便捷用法。"""
    print("\n" + "=" * 55)
    print("演示 6：get_or_create 模式")
    print("=" * 55)

    mgr = SessionManager()

    # 场景 1：第一次调用，没有 session_id → 自动创建
    sid1, state1 = mgr.get_or_create(task="第一次对话")
    print(f"场景1 (无session_id): 自动创建 → {sid1}")

    # 场景 2：再次用同样的 session_id → 复用已有会话
    sid2, state2 = mgr.get_or_create(session_id=sid1)
    print(f"场景2 (有session_id): 复用已有 → {sid2}")
    print(f"  两次返回同一个会话? {sid1 == sid2}")

    # 场景 3：传一个不存在的 session_id → 创建新的
    sid3, state3 = mgr.get_or_create(session_id="fake-id-12345", task="新任务")
    print(f"场景3 (假的session_id): 创建新的 → {sid3}")
    print(f"  和原来的不同? {sid1 != sid3}")

    print(f"\n最终活跃会话数: {mgr.count()}")
    for info in mgr.list_sessions():
        print(f"  {info['session_id']}: {info['task']}")

    # 清理
    mgr.delete_session(sid1)
    mgr.delete_session(sid3)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_basic_state()
    demo_multi_turn()
    demo_concurrent_sessions()
    demo_serialization()
    demo_session_expiry()
    demo_get_or_create()

    print("\n" + "=" * 55)
    print("全部 6 个演示运行完毕")
    print("=" * 55)
