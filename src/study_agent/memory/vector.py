"""VectorMemory —— 向量检索记忆。

工作原理：
  每条对话存入 ChromaDB 时附带一个向量（语义指纹）。
  取上下文时，把用户当前提问转成向量，去数据库里搜"最相似"的历史对话。

  类比：脑子里不是按时间顺序记住所有事情的——你听到"Python"，
  自然联想到相关的记忆，而不是从今天开始倒着回忆。

核心流程：
  1. add()  -> Embedder 把消息转成向量 -> 写入 ChromaDB
  2. get_context() -> 把 query 转成向量 -> ChromaDB 搜 top-K -> 返回

优点：不按时间、按相关性，能找到"久远但相关"的信息
缺点：可能漏掉"看起来不相关但实际重要"的上下文；依赖 embedding 质量
适用：需要跨话题检索的长对话、知识型对话
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import tiktoken

from study_agent.memory.base import BaseMemory
from study_agent.rag.embedding import Embedder


class VectorMemory(BaseMemory):
    """基于向量相似度的记忆检索。"""

    def __init__(self, embedder: Embedder | None = None, top_k: int = 5) -> None:
        """创建一个 VectorMemory。

        embedder — 文本转向量的工具，不传则自动创建（三级降级策略）
        top_k    — 每次检索返回多少条最相关的历史消息
        """
        self._embedder = embedder or Embedder()
        self._client = chromadb.EphemeralClient()
        # 每次实例化用唯一 collection 名，避免多实例冲突
        self._collection = self._client.get_or_create_collection(
            name=f"vector_memory_{uuid.uuid4().hex[:8]}"
        )
        self._top_k = top_k
        self._encoder = tiktoken.get_encoding("o200k_base")
        self._round_count = 0

    # ── 公开接口 ─────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """存入一轮对话（嵌入选向量后写入 ChromaDB）。"""
        self._round_count += 1
        doc_id = f"turn-{self._round_count}"
        text = f"{role}: {content}"

        # 生成向量
        embedding = self._embedder.embed_query(text)

        # 写入 ChromaDB
        self._collection.add(
            documents=[text],
            embeddings=[embedding],  # type: ignore[arg-type]
            ids=[doc_id],
        )

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """用 query 向量搜索最相关的历史消息，拼成上下文。

        如果 query 为空，退化为返回最近的消息。
        """
        if self._round_count == 0:
            return ""

        # 生成查询向量
        query_embedding = self._embedder.embed_query(query)

        # 检索 top-K（多取一些，后面按 token 裁剪）
        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=min(self._top_k * 3, self._round_count),
        )

        raw = results.get("documents", None)
        if raw and len(raw) > 0 and raw[0] is not None:
            doc_list: list[str] = raw[0]
        else:
            doc_list = []

        # 按 token 上限裁剪
        lines: list[str] = []
        tokens_used = 0
        for doc in doc_list:
            doc_tokens = len(self._encoder.encode(doc))
            if tokens_used + doc_tokens > max_tokens:
                break
            lines.append(doc)
            tokens_used += doc_tokens

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """统计当前状态。"""
        # 从 collection 读取全部文档来计算 token
        all_data = self._collection.get()
        all_docs = all_data.get("documents", [])
        full_text = "\n".join(all_docs) if all_docs else ""

        return {
            "type": "VectorMemory",
            "rounds": self._round_count,
            "total_chars": len(full_text),
            "total_tokens": len(self._encoder.encode(full_text)),
            "top_k": self._top_k,
        }
