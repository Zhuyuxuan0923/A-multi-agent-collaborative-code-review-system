"""
Agent 通信总线 (MessageBus)。

它是所有 Agent 之间的消息"邮局"——Agent 不直接通信，而是通过 Bus 中转。
这样做的好处：
1. 解耦：Agent A 不需要知道 Agent B 的地址
2. 可观测：所有消息经过 Bus，可以记录、监控
3. 可靠：Bus 可以处理超时、重试、消息持久化

通信模式对照 Day 1 的 4 种协作模式：
- 顺序模式：send() 点对点，A -> B -> C
- 辩论模式：publish() + request() 多轮
- 层级模式：request() 星型，Orchestrator <-> Workers
- 广播模式：publish() 一发多收
"""

import asyncio
import random
from collections import defaultdict
from collections.abc import Callable

from .message_protocol import (
    AgentInfo,
    AgentMessage,
    DeliveryResult,
)


class AgentNotRegisteredError(Exception):
    """目标 Agent 未注册。"""

    pass


class MessageTimeoutError(Exception):
    """消息等待回复超时。"""

    pass


class MessageBus:
    """
    内存消息总线，不依赖 Redis 等外部组件。

    使用示例：

        bus = MessageBus()

        # 注册 Agent
        bus.register_agent(AgentInfo(agent_id="reviewer_1", agent_type="worker"))

        # 订阅主题
        bus.subscribe("reviewer_1", "code_review.security")

        # 发布消息到主题
        msg = AgentMessage(sender_id="orch", topic="code_review.security", ...)
        bus.publish(msg)

        # 点对点发送
        bus.send(msg)

        # 请求-回复（带超时 + 重试）
        reply = await bus.request(msg, timeout=10.0, max_retries=3)
    """

    def __init__(self):
        # Agent 注册表: agent_id -> AgentInfo
        self._agents: dict[str, AgentInfo] = {}

        # 主题订阅: topic -> set[agent_id]
        self._subscriptions: dict[str, set[str]] = defaultdict(set)

        # 每个 Agent 的消息收件箱: agent_id -> asyncio.Queue
        self._inboxes: dict[str, asyncio.Queue[AgentMessage]] = {}

        # 待处理的请求: correlation_id -> asyncio.Future
        # 用于 request() 等待 reply()
        self._pending: dict[str, asyncio.Future[AgentMessage]] = {}

        # 消息日志（内存，不做持久化）
        self._history: list[AgentMessage] = []

        # 投递结果记录
        self._delivery_log: list[DeliveryResult] = []

        # 消息处理中间件链：每条消息投递前都会经过这些函数
        self._middleware: list[Callable[[AgentMessage], AgentMessage | None]] = []

    # ================================================================
    # Agent 注册与发现
    # ================================================================

    def register_agent(self, info: AgentInfo) -> None:
        """注册一个 Agent 到 Bus。

        注册后 Agent 才能收发消息。每个 Agent 会得到自己的收件箱（asyncio.Queue）。
        """
        self._agents[info.agent_id] = info
        if info.agent_id not in self._inboxes:
            self._inboxes[info.agent_id] = asyncio.Queue()

        # 自动订阅 Agent 声明的能力主题
        for cap in info.capabilities:
            for topic in cap.topics:
                self.subscribe(info.agent_id, topic)

    def unregister_agent(self, agent_id: str) -> None:
        """注销 Agent。"""
        self._agents.pop(agent_id, None)
        self._inboxes.pop(agent_id, None)
        # 清理订阅
        for topic_subs in self._subscriptions.values():
            topic_subs.discard(agent_id)

    def get_agent(self, agent_id: str) -> AgentInfo:
        """获取 Agent 注册信息。"""
        if agent_id not in self._agents:
            raise AgentNotRegisteredError(f"Agent '{agent_id}' 未注册")
        return self._agents[agent_id]

    def list_agents(self) -> list[AgentInfo]:
        """列出所有已注册 Agent。"""
        return list(self._agents.values())

    def find_by_capability(self, capability_name: str) -> list[str]:
        """按能力名称查找 Agent ID 列表。"""
        result = []
        for agent_id, info in self._agents.items():
            for cap in info.capabilities:
                if cap.name == capability_name:
                    result.append(agent_id)
        return result

    # ================================================================
    # 主题订阅（Pub/Sub）
    # ================================================================

    def subscribe(self, agent_id: str, topic: str) -> None:
        """订阅一个主题。"""
        if agent_id not in self._agents:
            raise AgentNotRegisteredError(f"Agent '{agent_id}' 未注册，先 register 再 subscribe")
        self._subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """取消订阅。"""
        self._subscriptions[topic].discard(agent_id)

    def get_subscribers(self, topic: str) -> list[str]:
        """获取某个主题的所有订阅者。"""
        return list(self._subscriptions.get(topic, set()))

    # ================================================================
    # 消息发送（核心）
    # ================================================================

    def publish(self, message: AgentMessage) -> list[str]:
        """发布消息到主题（pub/sub 模式）。

        消息会被投递到所有订阅了该主题的 Agent 的收件箱。

        返回：实际投递的 Agent ID 列表。
        """
        if message.topic is None:
            raise ValueError("publish() 要求 message.topic 不为 None")

        subscribers = self._subscriptions.get(message.topic, set())
        if message.receiver_id:
            # 如果指定了 receiver_id，只发给那个人（但仍需订阅该主题）
            subscribers = {message.receiver_id} & subscribers

        delivered = []
        for agent_id in subscribers:
            self._enqueue(agent_id, message)
            delivered.append(agent_id)

        self._log(message, f"publish -> topic={message.topic}, delivered={delivered}")
        return delivered

    def send(self, message: AgentMessage) -> None:
        """点对点发送消息（direct 模式）。

        receiver_id 必须指定。消息直接进入接收方的收件箱。
        """
        if message.receiver_id is None:
            raise ValueError("send() 要求 message.receiver_id 不为 None")

        receiver = message.receiver_id
        if receiver not in self._agents:
            raise AgentNotRegisteredError(
                f"收件人 '{receiver}' 未注册。已注册的 Agent: {list(self._agents.keys())}"
            )

        self._enqueue(receiver, message)
        self._log(message, f"send -> {receiver}")

    async def request(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
        max_retries: int = 0,
        retry_backoff: float = 1.0,
    ) -> AgentMessage:
        """发送请求并等待回复（request-reply 模式）。

        这是同步语义：发送一条消息，等待对方回复，返回回复消息。
        内部用 correlation_id 把请求和回复关联起来。

        参数：
            timeout: 等待回复的超时时间（秒）
            max_retries: 超时后重试次数（0 = 不重试）
            retry_backoff: 重试退避的基数（秒），实际等待 = backoff * (2 ** retry_num) + jitter

        返回：回复消息。

        抛出：
            MessageTimeoutError: 超时且重试次数耗尽
        """
        if message.correlation_id is None:
            message.correlation_id = message.message_id

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._request_once(message, timeout, attempt)
            except MessageTimeoutError as e:
                last_error = e
                if attempt < max_retries:
                    # 指数退避 + 随机抖动
                    wait = retry_backoff * (2**attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(wait)

        raise MessageTimeoutError(
            f"request '{message.message_id}' 超时: "
            f"{max_retries + 1} 次尝试, 每次超时 {timeout}s. "
            f"最后错误: {last_error}"
        )

    async def _request_once(
        self, message: AgentMessage, timeout: float, attempt: int
    ) -> AgentMessage:
        """单次 request 尝试。"""
        # 创建 Future 等待回复
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[message.correlation_id] = future

        try:
            # 发送消息（如果指定了 receiver 走 send，否则走 publish）
            if message.receiver_id:
                self.send(message)
            elif message.topic:
                self.publish(message)
            else:
                raise ValueError("request() 需要 receiver_id 或 topic")

            # 等待回复
            reply = await asyncio.wait_for(future, timeout=timeout)
            return reply

        except TimeoutError:
            raise MessageTimeoutError(
                f"等待回复超时 ({timeout}s): "
                f"message_id={message.message_id}, receiver={message.receiver_id}"
            )
        finally:
            self._pending.pop(message.correlation_id, None)

    def reply(self, original_message: AgentMessage, payload: dict) -> AgentMessage:
        """回复一条消息。

        创建一个回复消息并发送。这是给接收方 Agent 用的快捷方法。

        用法（在 Agent 的消息处理循环中）：
            msg = await bus.receive("my_agent")
            reply = bus.reply(msg, {"result": "done"})
            # reply 会自动发回给 original sender
        """
        reply_msg = original_message.create_reply(
            sender_id=original_message.receiver_id or "unknown",
            payload=payload,
        )

        # 检查是否有人在等这个回复（即发起了 request）
        if original_message.correlation_id and original_message.correlation_id in self._pending:
            future = self._pending[original_message.correlation_id]
            if not future.done():
                future.set_result(reply_msg)
                self._log(
                    reply_msg, f"reply -> 关联到 pending request {original_message.correlation_id}"
                )
                return reply_msg

        # 没有人在等，正常发送回复
        self.send(reply_msg)
        self._log(reply_msg, f"reply -> {reply_msg.receiver_id}")
        return reply_msg

    # ================================================================
    # 消息接收
    # ================================================================

    async def receive(self, agent_id: str, timeout: float | None = None) -> AgentMessage:
        """从收件箱取一条消息（异步阻塞）。

        参数：
            agent_id: 接收方 Agent ID
            timeout: 等待超时（秒），None 表示无限等待

        返回：收到的消息。

        抛出：
            asyncio.TimeoutError: 超时
            AgentNotRegisteredError: Agent 未注册
        """
        if agent_id not in self._inboxes:
            raise AgentNotRegisteredError(f"Agent '{agent_id}' 未注册")

        if timeout is not None:
            return await asyncio.wait_for(self._inboxes[agent_id].get(), timeout=timeout)
        else:
            return await self._inboxes[agent_id].get()

    def receive_nowait(self, agent_id: str) -> AgentMessage | None:
        """非阻塞取消息，没有消息立即返回 None。"""
        if agent_id not in self._inboxes:
            return None
        try:
            return self._inboxes[agent_id].get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ================================================================
    # 广播快捷方法
    # ================================================================

    async def broadcast_and_collect(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
    ) -> dict[str, AgentMessage | None]:
        """广播消息并收集所有回复。

        用法（对应 Day 1 的广播模式）：
            results = await bus.broadcast_and_collect(msg)
            # results = {"agent_a": reply_msg, "agent_b": reply_msg, "agent_c": None(超时)}

        返回：{agent_id: reply_message_or_None}
        """
        if message.topic is None:
            raise ValueError("broadcast_and_collect() 要求 message.topic 不为 None")

        subscribers = self.get_subscribers(message.topic)
        if not subscribers:
            return {}

        # 并行发送给所有订阅者，等待回复
        async def send_one(agent_id: str) -> tuple[str, AgentMessage | None]:
            msg_copy = AgentMessage(**message.model_dump())
            msg_copy.receiver_id = agent_id
            msg_copy.message_id = message.message_id + "_" + agent_id
            msg_copy.correlation_id = msg_copy.message_id

            try:
                reply = await self.request(msg_copy, timeout=timeout, max_retries=0)
                return (agent_id, reply)
            except MessageTimeoutError:
                return (agent_id, None)

        tasks = [send_one(aid) for aid in subscribers]
        results = await asyncio.gather(*tasks)
        return dict(results)

    # ================================================================
    # 内部方法
    # ================================================================

    def _enqueue(self, agent_id: str, message: AgentMessage) -> None:
        """将消息放入收件人的收件箱。"""
        # 检查 TTL
        if message.is_expired():
            self._log(message, "丢弃: TTL 过期")
            return

        # 中间件链处理
        processed = message
        for mw in self._middleware:
            result = mw(processed)
            if result is None:
                self._log(message, "丢弃: 中间件拦截")
                return
            processed = result

        self._inboxes[agent_id].put_nowait(processed)

    def _log(self, message: AgentMessage, detail: str) -> None:
        """记录消息日志。"""
        self._history.append(message)
        # 限制历史长度，防止内存泄漏
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    # ================================================================
    # 中间件
    # ================================================================

    def use(self, middleware: Callable[[AgentMessage], AgentMessage | None]) -> None:
        """注册一个中间件。

        中间件函数接收消息，返回处理后的消息。返回 None 表示丢弃该消息。

        用法（日志中间件）：
            def log_middleware(msg):
                print(f"[{msg.sender_id}] -> [{msg.receiver_id}]: {msg.message_type}")
                return msg
            bus.use(log_middleware)
        """
        self._middleware.append(middleware)

    # ================================================================
    # 可观测性
    # ================================================================

    def stats(self) -> dict:
        """返回 Bus 的运行统计。"""
        return {
            "agents_registered": len(self._agents),
            "agent_ids": list(self._agents.keys()),
            "topics": {topic: len(subs) for topic, subs in self._subscriptions.items()},
            "history_size": len(self._history),
            "pending_requests": len(self._pending),
            "inbox_sizes": {
                aid: inbox.qsize() for aid, inbox in self._inboxes.items() if inbox.qsize() > 0
            },
        }

    def reset(self) -> None:
        """重置 Bus（用于测试）。"""
        self._agents.clear()
        self._subscriptions.clear()
        self._inboxes.clear()
        self._pending.clear()
        self._history.clear()
        self._delivery_log.clear()
        self._middleware.clear()
