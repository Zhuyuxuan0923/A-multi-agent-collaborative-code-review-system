"""Memory 对比评测引擎。

对三种 Memory 做同一组测试，对比：
1. 存储效率 — 存了相同对话后各自占用多少 token
2. 检索精准度 — 针对特定问题取出的上下文是否包含关键信息
3. 上下文 token 消耗 — 取出的上下文用了多少 token

最后生成 Markdown 格式的对比报告。
"""

from __future__ import annotations

import re
import time
from typing import Any

import tiktoken

from study_agent.memory.base import BaseMemory


class MemoryComparator:
    """对比评测引擎 —— 同一组对话 x 三种 Memory -> 量化对比报告。"""

    def __init__(self) -> None:
        self._encoder = tiktoken.get_encoding("o200k_base")
        self._results: list[dict[str, Any]] = []

    def run(
        self,
        conversations: list[tuple[str, str]],
        queries: list[dict[str, Any]],
        memories: dict[str, BaseMemory],
    ) -> list[dict[str, Any]]:
        """执行对比评测。

        conversations — 模拟对话列表 [(role, content), ...]
        queries       — 查询列表，每个查询包含:
                        - "query"      : 用户提问
                        - "keywords"   : 期望上下文中出现的关键词列表
                        - "topic"      : 话题名称（用于报告）
        memories      — {"名称": Memory实例} 字典

        返回评测结果列表，每个元素是一个 memory 的完整评测数据。
        """
        self._results = []

        for name, memory in memories.items():
            print(f"\n{'=' * 60}")
            print(f"评测: {name}")
            print(f"{'=' * 60}")

            # 1. 写入阶段 —— 把所有对话喂给 Memory
            print("  [1/3] 写入对话...")
            t0 = time.time()
            for role, content in conversations:
                memory.add(role, content)
            write_time = time.time() - t0

            # 2. 存储统计
            stats = memory.get_stats()
            print(f"         存储 {stats['rounds']} 轮, 耗时 {write_time:.3f}s")

            # 3. 查询阶段 —— 对每个查询测试检索效果
            print("  [2/3] 执行查询...")
            query_results = []
            for q in queries:
                t0 = time.time()
                context = memory.get_context(q["query"], max_tokens=2000)
                query_time = time.time() - t0

                context_tokens = len(self._encoder.encode(context))
                hits = self._check_keywords(context, q["keywords"])
                hit_count = sum(hits.values())
                total_keywords = len(q["keywords"])

                query_results.append(
                    {
                        "topic": q["topic"],
                        "query": q["query"],
                        "keywords": q["keywords"],
                        "hits": hits,
                        "hit_rate": hit_count / total_keywords if total_keywords > 0 else 0,
                        "context_tokens": context_tokens,
                        "context_chars": len(context),
                        "query_time": query_time,
                    }
                )

                print(
                    f"         [{q['topic']}] 命中 {hit_count}/{total_keywords} 关键词, "
                    f"上下文 {context_tokens} tokens"
                )

            # 4. 汇总
            avg_hit_rate = (
                sum(r["hit_rate"] for r in query_results) / len(query_results)
                if query_results
                else 0
            )
            avg_context_tokens = (
                sum(r["context_tokens"] for r in query_results) / len(query_results)
                if query_results
                else 0
            )

            print(
                f"  [3/3] 汇总: 平均命中率 {avg_hit_rate:.1%}, "
                f"平均上下文 {avg_context_tokens:.0f} tokens"
            )

            self._results.append(
                {
                    "name": name,
                    "stats": stats,
                    "write_time": write_time,
                    "queries": query_results,
                    "avg_hit_rate": avg_hit_rate,
                    "avg_context_tokens": avg_context_tokens,
                }
            )

        return self._results

    def _check_keywords(self, text: str, keywords: list[str]) -> dict[str, bool]:
        """检查 text 中包含哪些关键词（忽略大小写、允许词中匹配）。"""
        text_lower = text.lower()
        hits: dict[str, bool] = {}
        for kw in keywords:
            # 用正则做单词边界匹配——但中文没有空格分隔，所以用直接包含匹配
            # 英文关键词用 \b 边界
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_lower):
                hits[kw] = True
            elif kw.lower() in text_lower:
                hits[kw] = True
            else:
                hits[kw] = False
        return hits

    def generate_report(self) -> str:
        """根据 run() 的结果生成 Markdown 对比报告。"""
        if not self._results:
            return "还没有评测数据，请先调用 run()。"

        lines: list[str] = [
            "# Memory 机制对比评测报告",
            "",
            f"评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"评测 Memory 数: {len(self._results)}",
            "",
            "---",
            "",
            "## 一、存储效率对比",
            "",
            "| Memory 类型 | 存储轮数 | 总字符数 | 总 Token 数 | 写入耗时 |",
            "|------------|---------|---------|------------|---------|",
        ]

        for r in self._results:
            s = r["stats"]
            lines.append(
                f"| {r['name']} | {s['rounds']} | {s['total_chars']:,} | "
                f"{s['total_tokens']:,} | {r['write_time']:.3f}s |"
            )

        # 存储效率分析
        lines += [
            "",
            "### 分析",
            "",
        ]
        buffer_tokens = self._results[0]["stats"]["total_tokens"] if self._results else 1
        for r in self._results:
            t = r["stats"]["total_tokens"]
            ratio = t / buffer_tokens * 100 if buffer_tokens > 0 else 0
            lines.append(f"- **{r['name']}**: 存储 {t:,} tokens（相对于 Buffer 的 {ratio:.0f}%）")

        lines += [
            "",
            "---",
            "",
            "## 二、检索精准度对比",
            "",
        ]

        # 表头
        if self._results and self._results[0]["queries"]:
            topics = [q["topic"] for q in self._results[0]["queries"]]
            header = "| Memory 类型 | " + " | ".join(topics) + " | 平均命中率 |"
            sep = "|------------|" + "|".join(["-" * 8 for _ in topics]) + "|------------|"
            lines.append(header)
            lines.append(sep)

            for r in self._results:
                rates = [f"{q['hit_rate']:.0%}" for q in r["queries"]]
                avg = r["avg_hit_rate"]
                score = f"{avg:.0%}"
                row = f"| {r['name']} | " + " | ".join(rates) + f" | {score} |"
                lines.append(row)

        lines += [
            "",
            "### 各 Memory 检索详情",
            "",
        ]

        for r in self._results:
            lines.append(f"### {r['name']}")
            lines.append("")
            for q in r["queries"]:
                hits = q["hits"]
                hit_list = [f"{'[命中]' if v else '[丢失]'} `{k}`" for k, v in hits.items()]
                lines.append(
                    f"- **{q['topic']}**: {', '.join(hit_list)} "
                    f"（上下文 {q['context_tokens']} tokens）"
                )
            lines.append("")

        lines += [
            "---",
            "",
            "## 三、上下文 Token 消耗对比",
            "",
            "| Memory 类型 | 平均上下文 Token | 相对于 Buffer |",
            "|------------|-----------------|--------------|",
        ]

        buffer_ctx = self._results[0]["avg_context_tokens"] if self._results else 1
        for r in self._results:
            ct = r["avg_context_tokens"]
            ratio = ct / buffer_ctx * 100 if buffer_ctx > 0 else 0
            lines.append(f"| {r['name']} | {ct:.0f} | {ratio:.0f}% |")

        lines += [
            "",
            "### 分析",
            "",
            "- **BufferMemory**: 上下文长短取决于对话总长度，长对话时 token 消耗线性增长",
            "- **SummaryMemory**: 上下文长度稳定，由「摘要 + 最近N轮」决定，和总对话长度无关",
            "- **VectorMemory**: 上下文长度由检索结果数（top_k）控制，同样不受总对话长度影响",
            "",
            "---",
            "",
            "## 四、综合评分与建议",
            "",
            "| 维度 | Buffer | Summary | Vector |",
            "|------|--------|---------|--------|",
            "| 存储效率 | 差 | 优 | 中 |",
            "| 检索精准度 | 优 | 中 | 中 |",
            "| Token 效率 | 差 | 优 | 中 |",
            "| 实现复杂度 | 极简 | 中（需LLM） | 中（需Embedding+DB） |",
            "| 长对话适用性 | 差 | 优 | 中 |",
            "",
            "### 选型建议",
            "",
            "- **短对话（< 20 轮）**: 直接用 Buffer，简单可靠",
            "- **长对话 + 成本敏感**: 用 Summary，token 消耗恒定",
            "- **需要跨话题检索**: 用 Vector，能找回「久远但相关」的信息",
            "- **生产级方案**: 三者混合使用（Day 2 内容）",
            "",
            "---",
            "",
        ]

        # 各 Memory 优缺点表格
        lines.append("## 五、各 Memory 优缺点")
        lines.append("")

        pros_cons = {
            "BufferMemory": {
                "优点": ["实现最简单，零外部依赖", "不丢失任何信息", "不需要 LLM 调用，速度快"],
                "缺点": ["Token 消耗随对话线性增长", "长对话成本高", "不问的话题也全量带上"],
            },
            "SummaryMemory": {
                "优点": [
                    "Token 消耗稳定，不受总对话长度影响",
                    "保留最近原文 + 压缩历史，兼顾精准和成本",
                ],
                "缺点": [
                    "摘要可能丢失关键细节",
                    "依赖 LLM 调用（有延迟和费用）",
                    "摘要质量依赖 LLM 能力",
                ],
            },
            "VectorMemory": {
                "优点": [
                    "按语义检索，能找回久远但相关的内容",
                    "Token 消耗可控（top_k 控制）",
                    "适合跨话题的长对话",
                ],
                "缺点": [
                    "依赖 embedding 模型质量",
                    "可能漏掉「看起来不相关但重要」的信息",
                    "需要向量数据库支持",
                ],
            },
        }

        for r in self._results:
            name = r["name"]
            if name in pros_cons:
                lines.append(f"### {name}")
                lines.append("")
                lines.append("**优点:**")
                for pro in pros_cons[name]["优点"]:
                    lines.append(f"- {pro}")
                lines.append("")
                lines.append("**缺点:**")
                for con in pros_cons[name]["缺点"]:
                    lines.append(f"- {con}")
                lines.append("")

        return "\n".join(lines)
