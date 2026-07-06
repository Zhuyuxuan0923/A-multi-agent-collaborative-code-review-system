"""
Agent 间通信协议定义。

消息格式使用 JSON Schema 约束，确保所有 Agent 说同一种"语言"。
类比：HTTP 定义了请求和响应的格式，这个模块定义了 Agent 间消息的格式。
"""

import json
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============================================================
# 1. 消息类型枚举 —— 就像 HTTP 的 GET / POST / PUT / DELETE
# ============================================================


class MessageType(str, Enum):
    """消息类型决定了接收方如何处理这条消息。"""

    TASK_ASSIGNMENT = "task_assignment"  # 编排器 -> 工作者：分配任务
    TASK_RESULT = "task_result"  # 工作者 -> 编排器：返回结果
    QUERY = "query"  # Agent -> Agent：查询信息
    RESPONSE = "response"  # Agent -> Agent：回复查询
    BROADCAST = "broadcast"  # 一对多：向所有订阅者发送
    HEARTBEAT = "heartbeat"  # 存活检测：我还活着
    ERROR = "error"  # 异常通知
    ACK = "ack"  # 确认收到（不等于任务完成）


# ============================================================
# 2. 消息优先级
# ============================================================


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# 3. 消息体 —— 就像 HTTP 的 request body
# ============================================================


class AgentMessage(BaseModel):
    """Agent 之间传递的标准消息格式。

    每条消息都有一个唯一 ID (message_id)，和一个可选的关联 ID (correlation_id)。
    correlation_id 用来把请求和回复串联起来——就像快递单号。
    """

    message_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12], description="消息唯一标识，自动生成"
    )
    correlation_id: str | None = Field(
        default=None, description="关联 ID，用于把回复和请求串起来。请求方填入，回复方原样返回。"
    )
    sender_id: str = Field(..., description="发送方 Agent 的 ID")
    receiver_id: str | None = Field(
        default=None, description="接收方 Agent 的 ID。None 表示广播（发给所有订阅者）。"
    )
    topic: str | None = Field(
        default=None, description="消息主题，用于 pub/sub 路由。如 'code_review.security'"
    )
    message_type: MessageType = Field(..., description="消息类型")
    priority: Priority = Field(default=Priority.NORMAL, description="消息优先级")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="消息内容，JSON 格式的自由数据"
    )
    timestamp: float = Field(default_factory=time.time, description="消息创建时间（Unix 时间戳）")
    ttl: float | None = Field(
        default=None, description="Time-To-Live，消息过期时间（秒）。超时后消息不再投递。"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="扩展元数据，如 trace_id, user_id 等"
    )

    def is_expired(self) -> bool:
        """检查消息是否已过期。"""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def create_reply(self, sender_id: str, payload: dict[str, Any]) -> "AgentMessage":
        """创建一个回复消息，自动设置 correlation_id。

        用法：
            reply = original_msg.create_reply(
                sender_id="agent_b",
                payload={"result": "done"}
            )
            # reply.correlation_id == original_msg.message_id
        """
        return AgentMessage(
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=MessageType.RESPONSE,
            correlation_id=self.message_id,
            payload=payload,
        )

    def create_error_reply(
        self, sender_id: str, error: str, error_code: str = "UNKNOWN"
    ) -> "AgentMessage":
        """创建一个错误回复消息。"""
        return AgentMessage(
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=MessageType.ERROR,
            correlation_id=self.message_id,
            priority=Priority.HIGH,
            payload={"error": error, "error_code": error_code},
        )

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """从 JSON 字符串反序列化。"""
        return cls.model_validate_json(json_str)


# ============================================================
# 4. Agent 注册信息 —— 让其他 Agent 知道"我能做什么"
# ============================================================


class AgentCapability(BaseModel):
    """描述一个 Agent 的能力，用于路由决策。"""

    name: str = Field(..., description="能力名称，如 'code_review'")
    description: str = Field(default="", description="能力描述")
    topics: list[str] = Field(default_factory=list, description="订阅的消息主题")
    max_concurrent: int = Field(default=1, description="最大并发任务数")


class AgentInfo(BaseModel):
    """Agent 注册信息。MessageBus 用这个来管理路由表。"""

    agent_id: str = Field(..., description="Agent 唯一 ID")
    agent_type: str = Field(
        default="worker", description="Agent 类型: orchestrator / worker / reporter"
    )
    capabilities: list[AgentCapability] = Field(default_factory=list)
    status: str = Field(default="idle", description="状态: idle / busy / offline")
    registered_at: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)


# ============================================================
# 5. 路由规则 —— 决定消息去哪
# ============================================================


class RouteType(str, Enum):
    DIRECT = "direct"  # 指定 receiver_id，点对点
    TOPIC = "topic"  # 按 topic 匹配订阅者
    CAPABILITY = "capability"  # 按 Agent 能力匹配
    BROADCAST = "broadcast"  # 所有注册 Agent
    ROUND_ROBIN = "round_robin"  # 轮询（负载均衡）


class RoutingRule(BaseModel):
    """一条路由规则。MessageBus 根据规则决定消息投递给谁。"""

    route_type: RouteType
    receiver_ids: list[str] | None = None  # DIRECT 时指定
    topic: str | None = None  # TOPIC 时指定
    capability: str | None = None  # CAPABILITY 时指定


# ============================================================
# 6. 消息投递结果
# ============================================================


class DeliveryResult(BaseModel):
    """一次消息投递的结果。"""

    message_id: str
    receiver_id: str
    success: bool
    error: str | None = None
    response: AgentMessage | None = None
    latency_ms: float = 0.0
    retry_count: int = 0


# ============================================================
# 7. JSON Schema 导出（方便文档化 + 跨语言使用）
# ============================================================


def export_json_schema() -> dict:
    """导出 AgentMessage 的 JSON Schema，可用于文档或跨语言校验。"""
    return AgentMessage.model_json_schema()


if __name__ == "__main__":
    # 快速自检：创建一条消息并序列化
    msg = AgentMessage(
        sender_id="orchestrator",
        receiver_id="reviewer_1",
        message_type=MessageType.TASK_ASSIGNMENT,
        topic="code_review.security",
        payload={"task": "审查 src/auth.py 的安全问题"},
        ttl=60.0,
    )

    print("[消息示例]")
    print(msg.to_json())

    # 创建回复
    reply = msg.create_reply(sender_id="reviewer_1", payload={"verdict": "pass", "issues": []})
    print("\n[回复消息]")
    print(f"correlation_id == 原消息 ID: {reply.correlation_id == msg.message_id}")

    # 导出 JSON Schema
    print("\n[JSON Schema 导出（前 500 字符）]")
    schema_json = json.dumps(export_json_schema(), indent=2, ensure_ascii=False)
    print(schema_json[:500])
