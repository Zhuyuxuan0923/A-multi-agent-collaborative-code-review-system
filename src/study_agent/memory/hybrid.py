"""HybridMemory —— 三层混合记忆。

工作原理（类比人脑记忆）：
  1. Buffer 层  — "刚才在聊什么"——最近 N 轮对话，一字不差记住
  2. Summary 层 — "今天聊了什么"——早期对话的 LLM 压缩摘要
  3. Vector 层  — "什么时候提过这个"——语义检索，跨话题找回久远内容

检索优先级（从高到低）：
  Buffer > Summary > Vector

  这个顺序不是随意的：
  - Buffer 排第一：正在聊的话题最相关，需要完整上下文
  - Summary 排第二：提供对话的宏观背景
  - Vector 排第三：填补前两层漏掉的具体细节

Token 预算分配：
  假设总预算 2000 tokens：
    Buffer  35% = 700 tokens  —— 保留最近原文
    Summary 30% = 600 tokens  —— 压缩的历史
    Vector  35% = 700 tokens  —— 语义检索结果

去重机制：
  Vector 检索结果如果已经出现在 Buffer 中，会被自动过滤掉，
  避免同一段对话在上下文中出现两次。
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import tiktoken

from study_agent.llm.client import LLMClient
from study_agent.memory.base import BaseMemory
from study_agent.rag.embedding import Embedder


class HybridMemory(BaseMemory):
    """三层混合记忆 —— 近期原文 + 远期摘要 + 语义检索。"""

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder | None = None,
        recent_rounds: int = 5,
        vector_top_k: int = 3,
        token_budget: dict[str, float] | None = None,
    ) -> None:
        """创建一个 HybridMemory。

        llm           — 用于生成摘要的 LLM 客户端
        embedder      — 文本转向量的工具，不传则自动创建（三级降级策略）
        recent_rounds — Buffer 层保留多少轮原文（默认 5）
        vector_top_k  — Vector 层每次检索返回多少条（默认 3）
        token_budget  — 三层 token 分配比例，默认 {"buffer": 0.35, "summary": 0.30, "vector": 0.35}
        """
        self._llm = llm
        self._embedder = embedder or Embedder()
        self._recent_rounds = recent_rounds
        self._vector_top_k = vector_top_k
        self._token_budget = token_budget or {
            "buffer": 0.35,
            "summary": 0.30,
            "vector": 0.35,
        }

        # Buffer 层：所有消息的列表（最新 recent_rounds 条被当作 Buffer）
        self._messages: list[tuple[str, str]] = []

        # Summary 层：早期对话的压缩摘要文本
        self._summary: str = ""

        # Vector 层：ChromaDB 存所有对话的向量
        self._chroma_client = chromadb.EphemeralClient()
        self._collection = self._chroma_client.get_or_create_collection(
            name=f"hybrid_memory_{uuid.uuid4().hex[:8]}"
        )

        # 轮次计数（用作 ChromaDB 的文档 ID）
        self._round_count = 0

        # Token 编码器
        self._encoder = tiktoken.get_encoding("o200k_base")

    # ── 公开接口 ─────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """存入一轮对话，同时更新三层 Memory。

        每次 add 发生的事情：
          1. Buffer  层：追加到消息列表末尾
          2. Summary 层：检查是否需要触发摘要更新
          3. Vector  层：嵌入后写入 ChromaDB
        """
        self._messages.append((role, content))
        self._round_count += 1

        # Vector 层：嵌入 + 写入
        text = f"{role}: {content}"
        embedding = self._embedder.embed_query(text)
        self._collection.add(
            documents=[text],
            embeddings=[embedding],  # type: ignore[arg-type]
            ids=[f"turn-{self._round_count}"],
        )

        # Summary 层：当超出 recent_rounds 的轮数累积到 recent_rounds 条时触发更新
        # 不是每轮都摘要——太费 API 调用，而是攒够一批再做一次
        overflow = len(self._messages) - self._recent_rounds
        if overflow >= self._recent_rounds:
            self._update_summary()

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """从三层 Memory 中取出上下文，合并成一个字符串。

        合并顺序：Summary -> Buffer -> Vector（权重从低到高，后面的更相关）
        最终拼成 "[历史摘要] + [近期对话] + [相关历史记忆]"
        """
        buffer_budget = int(max_tokens * self._token_budget["buffer"])
        summary_budget = int(max_tokens * self._token_budget["summary"])
        vector_budget = int(max_tokens * self._token_budget["vector"])

        parts: list[str] = []

        # 1. Summary 层 —— 长期压缩记忆
        if self._summary:
            summary_text = f"[对话历史摘要]\n{self._summary}"
            summary_text = self._truncate(summary_text, summary_budget)
            parts.append(summary_text)

        # 2. Buffer 层 —— 最近原文
        recent = self._messages[-self._recent_rounds :]
        if recent:
            recent_lines = "\n".join(f"{r}: {c}" for r, c in recent)
            buffer_text = f"[近期对话]\n{recent_lines}"
            buffer_text = self._truncate(buffer_text, buffer_budget)
            parts.append(buffer_text)

            # 记录 buffer 中的文本，用于去重
            recent_set = {f"{r}: {c}" for r, c in recent}
        else:
            recent_set: set[str] = set()

        # 3. Vector 层 —— 语义检索
        if query and self._round_count > 0:
            query_embedding = self._embedder.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=min(self._vector_top_k, self._round_count),
            )

            raw = results.get("documents", None)
            if raw and len(raw) > 0 and raw[0] is not None:
                vector_docs: list[str] = raw[0]

                # 去重：过滤已在 Buffer 中的内容
                filtered_docs = [d for d in vector_docs if d not in recent_set]

                if filtered_docs:
                    vector_text = "[相关历史记忆]\n" + "\n".join(filtered_docs)
                    vector_text = self._truncate(vector_text, vector_budget)
                    parts.append(vector_text)

        return "\n\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """返回三层 Memory 各自的统计信息。"""
        full_text = "\n".join(f"{r}: {c}" for r, c in self._messages)
        total_tokens = len(self._encoder.encode(full_text))

        # 统计 vector 层的总 token
        all_data = self._collection.get()
        all_docs = all_data.get("documents", [])
        vector_text = "\n".join(all_docs) if all_docs else ""
        vector_tokens = len(self._encoder.encode(vector_text))

        return {
            "type": "HybridMemory",
            "rounds": len(self._messages),
            "total_chars": len(full_text),
            "total_tokens": total_tokens,
            "buffer_rounds": min(self._recent_rounds, len(self._messages)),
            "summary_chars": len(self._summary),
            "vector_rounds": len(all_docs),
            "vector_tokens": vector_tokens,
            "recent_rounds": self._recent_rounds,
            "vector_top_k": self._vector_top_k,
            "token_budget": self._token_budget,
        }

    # ── 内部方法 ─────────────────────────────────────────

    def _update_summary(self) -> None:
        """把超出 recent_rounds 的消息合并到摘要中。

        和 SummaryMemory 的逻辑一样：
        - 首次：从零生成摘要
        - 后续：把新对话合并到已有摘要中

        摘要完成后，被摘要掉的消息从 _messages 中删除，
        只保留最近 recent_rounds 条（这些作为 Buffer 层）。
        """
        to_summarize = self._messages[: -self._recent_rounds]
        if not to_summarize:
            return

        conversation = "\n".join(f"{r}: {c}" for r, c in to_summarize)

        if self._summary:
            prompt = (
                "下面是之前对话的摘要，以及新增的几轮对话。"
                "请把摘要更新一下，合并新对话的信息。只输出更新后的摘要，不要加额外说明。\n\n"
                f"[现有摘要]\n{self._summary}\n\n"
                f"[新增对话]\n{conversation}"
            )
        else:
            prompt = (
                "请为以下对话生成一段简洁的摘要（200 字以内），"
                "保留关键事实、用户偏好和重要决策。只输出摘要，不要加额外说明。\n\n"
                f"{conversation}"
            )

        try:
            self._summary = self._llm.chat(prompt)
        except Exception:
            # LLM 调用失败时静默处理，下次 add 会重试
            return

        # 删除已被摘要覆盖的消息，只保留 Buffer 层需要的 recent_rounds 条
        self._messages = self._messages[-self._recent_rounds :]

    def _truncate(self, text: str, max_tokens: int) -> str:
        """按 token 数从头部裁剪文本（保留末尾，即最近/最相关的内容）。"""
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoder.decode(tokens[-max_tokens:])
