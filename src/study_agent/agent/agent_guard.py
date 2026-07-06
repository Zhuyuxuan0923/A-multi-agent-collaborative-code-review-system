"""Agent 安全守卫 —— 输入清洗 / 注入检测 / 工具校验 / 循环保护。

设计原则：
  1. 不修改原有 Agent 代码 — 守卫是独立的包装层
  2. 守卫失败不应影响正常功能 — 宁可漏过也不误杀合法请求
  3. 可组合 — InputGuard / ToolGuard / LoopGuard 可以独立使用

分层防护模型（Defense in Depth）：

  用户输入
      |
      v
  +-------------+
  | InputGuard  |  <-- 第 1 层：输入门禁（长度/注入/越狱检测）
  +-------------+
      | (清洗后的输入)
      v
  +-------------+
  | Agent.run() |  <-- 第 2 层：Agent 执行（原有逻辑不变）
  +-------------+
      | (Agent 输出)
      v
  +-------------+
  | OutputGuard |  <-- 第 3 层：输出审核（泄露检测/有害内容过滤）
  +-------------+
      |
      v
  最终输出
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ========================================================================
# 第 1 部分: InputGuard — 输入安全检测
# ========================================================================
# 在用户输入到达 Agent 之前进行检查。
# 检查项：空输入、超长输入、重复内容、越狱模式、注入模式。
#
# 为什么用正则而不是 ML 模型？
#   - 速度快（微秒级）
#   - 不需要额外依赖
#   - 误报率可接受（对于已知模式的检测）
#   - 缺点是新型攻击可能绕过——需要持续更新模式库
# ========================================================================


@dataclass
class InputCheckResult:
    """输入检查结果。"""

    allowed: bool  # True = 通过, False = 被拦截
    reason: str  # 拦截原因（如果 allowed=False）
    sanitized_input: str  # 清洗后的输入
    warnings: list[str] = field(default_factory=list)  # 警告信息（不拦截但记录）


class InputGuard:
    """输入安全守卫。

    使用方式：
        guard = InputGuard()
        result = guard.check(user_input)
        if not result.allowed:
            return result.reason  # 拒绝请求
        answer = agent.run(result.sanitized_input)  # 使用清洗后的输入
    """

    # ── 配置 ──────────────────────────────────────────────

    MAX_INPUT_LENGTH: int = 4000  # 单次输入最大字符数
    MIN_MEANINGFUL_LENGTH: int = 2  # 有意义输入的最短长度
    MAX_REPETITION: int = 10  # 相同子串重复次数阈值
    REPETITION_WINDOW: int = 50  # 检测重复的窗口大小

    # ── 越狱模式 (jailbreak patterns) ─────────────────────

    JAILBREAK_PATTERNS: list[tuple[str, str, str]] = [
        # (正则, 模式名, 严重度)
        (
            r"ignore\s+(all|your|the|previous)\s+(instructions?|prompts?|rules?)",
            "ignore-instructions",
            "critical",
        ),
        (
            r"you\s+are\s+(now|no\s+longer)\s+(a\s+)?(DAN|dark|evil|unfiltered|unrestricted)",
            "role-override",
            "critical",
        ),
        (
            r"(SYSTEM|System)\s*:\s*you",
            "system-override",
            "high",
        ),
        (
            r"developer\s*mode|dev\s*mode.*activ",
            "dev-mode",
            "high",
        ),
        (
            r"(pretend|act|pose)\s+(as|like)\s+(a\s+)?(different|another|evil|unfiltered)",
            "role-play-override",
            "high",
        ),
        (
            r"(bypass|override|disable)\s+(your\s+)?(filter|rules?|restrictions?|safety)",
            "bypass-filters",
            "critical",
        ),
    ]

    # ── 提示注入模式 (prompt injection patterns) ──────────

    INJECTION_PATTERNS: list[tuple[str, str, str]] = [
        # (正则, 模式名, 严重度)
        (
            r"<\|im_start\|>|<\|im_end\|>",
            "chatml-token-injection",
            "high",
        ),
        (
            r"---END OF USER INPUT---|---USER INPUT ENDS---",
            "separator-injection",
            "medium",
        ),
        (
            r"NEW INSTRUCTIONS?:",
            "new-instructions-injection",
            "medium",
        ),
    ]

    # ── 信息提取模式 ─────────────────────────────────────

    EXTRACTION_PATTERNS: list[tuple[str, str, str]] = [
        (
            r"(repeat|output|tell\s+me|show\s+me|print|display)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?|configuration)",
            "system-prompt-extraction",
            "high",
        ),
        (
            r"what\s+(is|are|were)\s+your\s+(system\s+)?(prompt|instructions?)",
            "prompt-query",
            "high",
        ),
    ]

    def __init__(
        self,
        max_input_length: int | None = None,
        enable_jailbreak_detection: bool = True,
        enable_injection_detection: bool = True,
        enable_repetition_detection: bool = True,
    ):
        self.max_input_length = max_input_length or self.MAX_INPUT_LENGTH
        self.enable_jailbreak_detection = enable_jailbreak_detection
        self.enable_injection_detection = enable_injection_detection
        self.enable_repetition_detection = enable_repetition_detection

        # 拦截统计
        self.blocked_count: int = 0
        self.warning_count: int = 0

    def check(self, user_input: str) -> InputCheckResult:
        """对用户输入进行全面的安全检查。

        返回:
            InputCheckResult: allowed=True 表示可以通过，False 表示被拦截。
            即使 allowed=True，sanitized_input 也可能与原始输入不同（经过了清洗）。
        """
        warnings: list[str] = []
        sanitized = user_input

        # 检查 1: 空输入
        if not user_input or not user_input.strip():
            return InputCheckResult(
                allowed=False,
                reason="输入为空，请提供有效的问题。",
                sanitized_input="",
            )

        # 检查 2: 长度限制
        if len(user_input) > self.max_input_length:
            self.blocked_count += 1
            logger.warning(
                "InputGuard: 拦截超长输入 (%d 字符, 限制 %d)",
                len(user_input),
                self.max_input_length,
            )
            return InputCheckResult(
                allowed=False,
                reason=(
                    f"输入过长 ({len(user_input)} 字符)，"
                    f"最多允许 {self.max_input_length} 字符。请精简问题后重试。"
                ),
                sanitized_input=user_input[: self.max_input_length],
            )

        # 检查 3: 纯特殊字符/无意义输入
        meaningful_chars = re.sub(r"[\s\d\W]", "", user_input)
        if len(meaningful_chars) < self.MIN_MEANINGFUL_LENGTH:
            return InputCheckResult(
                allowed=False,
                reason="无法识别有效的问题内容，请提供更具体的问题描述。",
                sanitized_input="",
            )

        # 检查 4: 重复内容检测 (Token 轰炸)
        if self.enable_repetition_detection:
            rep_result = self._check_repetition(user_input)
            if rep_result:
                self.blocked_count += 1
                return InputCheckResult(
                    allowed=False,
                    reason=rep_result,
                    sanitized_input="",
                )

        # 检查 5: 越狱模式检测
        if self.enable_jailbreak_detection:
            for pattern, name, severity in self.JAILBREAK_PATTERNS:
                if re.search(pattern, user_input, re.IGNORECASE):
                    self.blocked_count += 1
                    logger.warning("InputGuard: 检测到越狱模式 [%s] (severity=%s)", name, severity)
                    return InputCheckResult(
                        allowed=False,
                        reason="检测到不安全的输入模式，已被拦截。",
                        sanitized_input="",
                    )

            for pattern, name, severity in self.EXTRACTION_PATTERNS:
                if re.search(pattern, user_input, re.IGNORECASE):
                    self.blocked_count += 1
                    logger.warning("InputGuard: 检测到信息提取尝试 [%s]", name)
                    return InputCheckResult(
                        allowed=False,
                        reason="检测到不安全的输入模式，已被拦截。",
                        sanitized_input="",
                    )

        # 检查 6: 提示注入模式检测
        if self.enable_injection_detection:
            for pattern, name, severity in self.INJECTION_PATTERNS:
                if re.search(pattern, user_input):
                    self.blocked_count += 1
                    logger.warning("InputGuard: 检测到注入模式 [%s] (severity=%s)", name, severity)
                    return InputCheckResult(
                        allowed=False,
                        reason="检测到不安全的输入模式，已被拦截。",
                        sanitized_input="",
                    )

        # 7: 输入清洗（不拦截，但清理潜在危险内容）
        sanitized = self._sanitize(user_input)

        if warnings:
            self.warning_count += len(warnings)

        return InputCheckResult(
            allowed=True,
            reason="",
            sanitized_input=sanitized,
            warnings=warnings,
        )

    def _check_repetition(self, text: str) -> str | None:
        """检测文本中的重复模式（Token 轰炸攻击）。

        策略：检查是否有相同的短子串（10-100 字符）重复超过阈值次数。
        """
        if len(text) < self.REPETITION_WINDOW * 2:
            return None

        # 取一个窗口大小的样本，检查是否在文本中反复出现
        sample = text[: self.REPETITION_WINDOW]
        count = len(re.findall(re.escape(sample), text))

        if count > self.MAX_REPETITION:
            return f"检测到大量重复内容 (同一模式出现 {count} 次)，已被拦截。"

        return None

    def _sanitize(self, text: str) -> str:
        """清洗输入——转义危险的分隔符，但不改变语义内容。

        为什么是转义而不是删除？
          删除可能改变用户的原意（比如用户真的在讨论 "Final Answer" 这个话题）。
          转义让 Agent 的解析器不会误解析，但保留了用户的原始意图。
        """
        # 转义 ReAct 关键字（防止被 Agent 的解析器误读）
        # 用零宽空格间隔来破坏关键字，但不影响人类阅读和 LLM 理解
        if "Final Answer:" in text:
            text = text.replace("Final Answer:", "Final\\_Answer:")
        if "Observation:" in text:
            text = text.replace("Observation:", "Observation\\_note:")
        if "Action:" in text:
            text = text.replace("Action:", "Action\\_step:")
        if "Thought:" in text:
            text = text.replace("Thought:", "Thought\\_note:")

        # 移除控制字符（保留常见的换行和制表符）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        return text

    def is_safe(self, user_input: str) -> bool:
        """快速检查：输入是否安全？（不返回详情）"""
        return self.check(user_input).allowed


# ========================================================================
# 第 2 部分: ToolGuard — 工具调用安全检测
# ========================================================================
# 在 Agent 调用工具之前/之后进行检查。
# 注意：这个守卫需要 Agent 在调用工具时主动调用 ToolGuard 的 check 方法。
# 如果 Agent 不支持中间拦截，ToolGuard 作为独立的验证工具使用。
# ========================================================================


@dataclass
class ToolCheckResult:
    """工具调用检查结果。"""

    allowed: bool
    reason: str
    max_param_length: int = 500  # 单个参数最大长度
    warnings: list[str] = field(default_factory=list)


class ToolGuard:
    """工具调用安全守卫。

    使用方式：
        guard = ToolGuard()
        result = guard.check_tool_call("search", {"query": user_query})
        if not result.allowed:
            return f"工具调用被拦截: {result.reason}"
    """

    MAX_PARAM_LENGTH: int = 500

    # 危险参数模式
    DANGEROUS_PARAM_PATTERNS: list[tuple[str, str]] = [
        (r"__import__|eval\s*\(|exec\s*\(|compile\s*\(", "代码注入尝试"),
        (r"rm\s+-rf|del\s+/[fs]|format\s+[cC]:", "系统命令注入"),
        (r"os\.system|subprocess\.|shell_exec|popen", "系统调用尝试"),
        (r"\.\./\.\./|/etc/passwd|/etc/shadow|C:\\Windows\\System32", "路径遍历尝试"),
    ]

    def __init__(self):
        self.blocked_count: int = 0
        # 记录最近几次工具调用，用于检测重复调用
        self._call_history: list[tuple[str, str]] = []  # [(tool_name, params_hash), ...]
        self._max_history: int = 10

    def check_tool_call(self, tool_name: str, params: dict[str, Any]) -> ToolCheckResult:
        """在工具调用前进行安全检查。"""
        warnings: list[str] = []

        # 检查 1: 参数长度
        for key, value in params.items():
            if isinstance(value, str) and len(value) > self.MAX_PARAM_LENGTH:
                self.blocked_count += 1
                return ToolCheckResult(
                    allowed=False,
                    reason=(
                        f"工具 '{tool_name}' 的参数 '{key}' 过长 "
                        f"({len(value)} 字符, 限制 {self.MAX_PARAM_LENGTH})"
                    ),
                    warnings=warnings,
                )

        # 检查 2: 危险参数内容
        for key, value in params.items():
            if isinstance(value, str):
                for pattern, description in self.DANGEROUS_PARAM_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        self.blocked_count += 1
                        logger.warning(
                            "ToolGuard: 拦截危险参数 [%s] tool=%s param=%s",
                            description,
                            tool_name,
                            key,
                        )
                        return ToolCheckResult(
                            allowed=False,
                            reason=f"工具参数包含不安全内容 ({description})，已被拦截。",
                            warnings=warnings,
                        )

        # 检查 3: 重复调用检测
        params_hash = self._hash_params(params)
        self._call_history.append((tool_name, params_hash))
        if len(self._call_history) > self._max_history:
            self._call_history.pop(0)

        # 检查最近 5 次调用中相同 (tool_name, params_hash) 出现次数
        recent = self._call_history[-5:]
        same_count = sum(1 for tn, ph in recent if tn == tool_name and ph == params_hash)
        if same_count >= 3:
            warnings.append(
                f"工具 '{tool_name}' 使用相同参数连续调用了 {same_count} 次，" f"可能是循环"
            )

        return ToolCheckResult(
            allowed=True,
            reason="",
            warnings=warnings,
        )

    @staticmethod
    def _hash_params(params: dict[str, Any]) -> str:
        """对参数做简单哈希（用于重复检测）。"""
        return str(sorted(str(v)[:100] for v in params.values()))

    def reset_history(self) -> None:
        """重置调用历史。"""
        self._call_history.clear()


# ========================================================================
# 第 3 部分: LoopGuard — 循环/资源耗尽保护
# ========================================================================


@dataclass
class LoopCheckResult:
    """循环检查结果。"""

    should_stop: bool  # True = 检测到死循环，应强制终止
    reason: str
    round_num: int
    alert_level: str = "normal"  # normal / warning / critical


class LoopGuard:
    """循环保护守卫。

    监控 Agent 的执行循环，检测以下异常模式：
      - 达到 max_rounds 上限
      - 连续 3 轮没有实质性进展（无工具调用或相同工具调用）
      - 估算的 prompt 长度超过安全阈值

    使用方式：
        guard = LoopGuard(max_rounds=8)
        for round_num in range(1, 100):
            result = guard.check(round_num, last_action)
            if result.should_stop:
                break
    """

    def __init__(
        self,
        max_rounds: int = 8,
        max_prompt_length: int = 16000,  # 估算的 prompt 最大字符数
        no_progress_threshold: int = 3,  # 连续无进展轮次上限
    ):
        self.max_rounds = max_rounds
        self.max_prompt_length = max_prompt_length
        self.no_progress_threshold = no_progress_threshold

        self._last_actions: list[str] = []  # 最近几轮的动作记录
        self._rounds_without_progress: int = 0

    def check(
        self,
        round_num: int,
        current_action: str = "",
        estimated_prompt_length: int = 0,
    ) -> LoopCheckResult:
        """检查当前轮次是否应该终止。

        参数：
          round_num                -> 当前轮次编号
          current_action           -> 本轮执行的动作描述
          estimated_prompt_length  -> 当前 prompt 的估算长度

        返回：
          LoopCheckResult: should_stop=True 表示应立即终止循环
        """
        # 检查 1: 达到最大轮次
        if round_num > self.max_rounds:
            return LoopCheckResult(
                should_stop=True,
                reason=f"已达到最大轮次限制 ({self.max_rounds})",
                round_num=round_num,
                alert_level="critical",
            )

        # 检查 2: prompt 长度超限
        if estimated_prompt_length > self.max_prompt_length:
            return LoopCheckResult(
                should_stop=True,
                reason=(
                    f"Prompt 长度 ({estimated_prompt_length} 字符) "
                    f"超过安全阈值 ({self.max_prompt_length})"
                ),
                round_num=round_num,
                alert_level="warning",
            )

        # 检查 3: 连续无进展检测
        self._last_actions.append(current_action)
        if len(self._last_actions) > self.no_progress_threshold:
            self._last_actions.pop(0)

        # 如果最近 N 轮动作都一样 -> 可能是循环
        if (
            len(self._last_actions) >= self.no_progress_threshold
            and len(set(self._last_actions)) == 1
            and current_action != ""
        ):
            return LoopCheckResult(
                should_stop=True,
                reason=(
                    f"连续 {self.no_progress_threshold} 轮执行相同操作 "
                    f"('{current_action}')，疑似陷入循环"
                ),
                round_num=round_num,
                alert_level="critical",
            )

        return LoopCheckResult(
            should_stop=False,
            reason="",
            round_num=round_num,
        )

    def reset(self) -> None:
        """重置守卫状态。"""
        self._last_actions.clear()
        self._rounds_without_progress = 0


# ========================================================================
# 第 4 部分: GuardedAgent — 组合以上三个守卫的包装器
# ========================================================================


class GuardedAgent:
    """安全守卫 Agent 包装器。

    将 InputGuard + ToolGuard + LoopGuard 组合在一起，
    对任何实现了 .run(question) -> str 接口的 Agent 提供防护。

    使用方式：
        raw_agent = ReactAgent(client)
        guarded = GuardedAgent(raw_agent)
        answer = guarded.run(user_input)  # 自动经过安全检测
    """

    def __init__(
        self,
        agent: Any,
        input_guard: InputGuard | None = None,
        tool_guard: ToolGuard | None = None,
        loop_guard: LoopGuard | None = None,
        verbose: bool = True,
    ):
        self._agent = agent
        self.input_guard = input_guard or InputGuard()
        self.tool_guard = tool_guard or ToolGuard()
        self.loop_guard = loop_guard or LoopGuard(max_rounds=getattr(agent, "max_rounds", 8))
        self.verbose = verbose

        # 统计
        self.total_requests: int = 0
        self.blocked_requests: int = 0
        self.safe_requests: int = 0

    def run(self, question: str) -> str:
        """带安全检测的 Agent 执行。

        防御层次:
          1. InputGuard 检查输入 -> 不通过则直接返回拒绝信息
          2. Agent 正常执行 -> 通过则继续
          3. 返回结果
        """
        self.total_requests += 1

        # 第 1 层: 输入检查
        input_result = self.input_guard.check(question)
        if not input_result.allowed:
            self.blocked_requests += 1
            if self.verbose:
                logger.info("GuardedAgent: 拦截请求 - %s", input_result.reason)
            return f"[已被安全守卫拦截] {input_result.reason}"

        # 第 2 层: 使用清洗后的输入正常执行
        self.safe_requests += 1
        try:
            answer = self._agent.run(input_result.sanitized_input)
        except Exception as e:
            logger.error("GuardedAgent: Agent 执行异常: %s", e)
            return f"[ERR] Agent 执行出错: {type(e).__name__}: {e}"

        # 第 3 层: 输出可以直接返回，也可以在这里加输出检查
        # (当前版本不做输出过滤，避免误杀正常回答)

        return answer

    @property
    def stats(self) -> dict[str, int]:
        """返回守卫统计信息。"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "safe_requests": self.safe_requests,
            "input_blocked": self.input_guard.blocked_count,
        }

    def __getattr__(self, name: str) -> Any:
        """将其他属性/方法调用透传给内部 Agent。"""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._agent, name)
