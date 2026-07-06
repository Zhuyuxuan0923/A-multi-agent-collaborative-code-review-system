"""HybridMemory 混合记忆演示。

模拟一段 30 轮的多话题对话，展示 HybridMemory 三层协同工作的效果。

运行:
  poetry run python src/study_agent/memory/hybrid_demo.py

输出:
  - 终端打印每层的检索结果
  - 对比 HybridMemory 与单一 Memory 的 token 消耗
  - 生成 docs/week6/hybrid_memory_demo.md 报告文件

场景设计:
  一段技术辅导对话，跨越 6 个话题，模拟真实的长对话场景。
  话题顺序故意跳跃——用户不会把一个问题聊透了再换下一个。
"""

from __future__ import annotations

import os
import time

from study_agent.llm.client import LLMClient
from study_agent.memory.buffer import BufferMemory
from study_agent.memory.hybrid import HybridMemory
from study_agent.memory.summary import SummaryMemory
from study_agent.memory.vector import VectorMemory

# ─── 模拟对话数据（30 轮，6 个话题）────────────────────────

SIMULATED_CONVERSATION: list[tuple[str, str]] = [
    # ─ 话题 1: Python 基础（第 1-5 轮）──
    ("user", "Python 里的列表和元组有什么区别？"),
    (
        "assistant",
        "列表用方括号 []，创建后可以增删改元素；元组用圆括号 ()，创建后不可修改。列表适合存「待办事项」这种会变的数据，元组适合存「经纬度坐标」这种固定的数据。",
    ),
    ("user", "那字典呢？什么时候用字典而不是列表？"),
    (
        "assistant",
        "字典是键值对结构，用花括号 {}。当你需要通过「名字」而不是「位置」来查找数据时用字典——比如存用户信息 user['name'] 比 user[0] 可读性高得多。",
    ),
    ("user", "遍历字典有什么推荐的写法？"),
    (
        "assistant",
        "推荐用 for key, value in dict.items()，一次拿到键和值。如果只遍历键，for key in dict 就够了，不需要 dict.keys()。",
    ),
    ("user", "Python 的 GIL 是什么？它影响我的代码吗？"),
    (
        "assistant",
        "GIL 是全局解释器锁，同一时刻只允许一个线程执行 Python 字节码。如果你用多线程做 CPU 密集型计算（如图像处理），GIL 会导致性能瓶颈。但 IO 密集型任务（如网络请求）不受影响，因为线程在等待 IO 时会释放 GIL。",
    ),
    # ─ 话题 2: Git 版本控制（第 6-10 轮）──
    ("user", "git commit 之前一定要 git add 吗？为什么不能直接 commit？"),
    (
        "assistant",
        "Git 的设计哲学是「两阶段提交」。add 把修改放入暂存区，commit 把暂存区的内容永久记录。这让你可以仔细挑选哪些修改该放进同一个 commit——不是所有修改都属于一个提交。",
    ),
    ("user", "我 commit 完了发现漏了一个文件，怎么办？"),
    (
        "assistant",
        "两种方式：1) git add 漏掉的文件，然后 git commit --amend，把文件追加到上一个 commit 里。2) 如果已经 push 了，就新建一个 commit 来补充，不要 amend 已经公开的 commit。",
    ),
    ("user", "merge 和 rebase 有什么区别？我该用哪个？"),
    (
        "assistant",
        "merge 保留完整的分支历史，会生成一个「合并提交」；rebase 把你的提交「搬到」目标分支顶端，历史更线性。简单原则：公共分支用 merge（保留历史），个人分支用 rebase（保持整洁）。",
    ),
    ("user", "我怎么撤销一个已经 push 的 commit？"),
    (
        "assistant",
        "用 git revert <commit-hash>。它会创建一个「反向操作」的新 commit，而不是删除历史。不要用 git reset --hard 然后 force push——那会破坏别人的历史。",
    ),
    # ─ 话题 3: API 设计（第 11-15 轮）──
    ("user", "RESTful API 的「资源」到底是什么意思？"),
    (
        "assistant",
        "资源就是你的 API 要操作的对象——用户、订单、文章。REST 的核心思想是把一切都建模为资源，用 HTTP 方法表达操作：GET 读取、POST 创建、PUT 更新、DELETE 删除。URL 只表示「谁」，不表示「做什么」。",
    ),
    ("user", "那 /api/createUser 这种 URL 有什么问题？"),
    (
        "assistant",
        "问题在于它把「动作」嵌入了 URL。正确的 REST 风格是 POST /api/users——动作由 HTTP 方法表达，URL 只表示资源集合。如果有一天要改成批量创建，POST /api/users 不用改 URL，/api/createUser 就很尴尬。",
    ),
    ("user", "API 版本号放 URL 里还是 Header 里好？"),
    (
        "assistant",
        "两种都常见。URL 版本（/v1/users）最直观，GitHub、Stripe 都这么用。Header 版本（Accept: application/vnd.api+v2）更「纯 REST」，但调试和文档都更麻烦。对于小团队和个人项目，URL 版本更实用。",
    ),
    ("user", "分页要怎么设计？返回 all 还是加参数？"),
    (
        "assistant",
        "一定要加分页参数！返回全部数据会让接口变成「定时炸弹」——数据量一大就超时。标准做法是 ?offset=0&limit=20，响应里附带 total_count 让客户端知道还有多少页。",
    ),
    # ─ 话题 4: 数据库（第 16-20 轮）──
    ("user", "索引为什么能加速查询？它不占空间吗？"),
    (
        "assistant",
        "索引就像书的目录——没有目录要找「GIL」需要翻遍全书，有目录直接翻到对应页。索引确实占空间（通常是原数据的 10-30%），也会拖慢写入速度（因为每次 insert 都要更新索引），所以不是「每列都加索引」就好。",
    ),
    ("user", "什么叫「慢查询」？怎么看一个查询慢不慢？"),
    (
        "assistant",
        "慢查询就是执行时间超过阈值的 SQL。MySQL 可以用 EXPLAIN 看执行计划——如果看到 type=ALL（全表扫描）或者 rows=1000000，那就是慢查询。一般用 slow_query_log 自动记录超过 N 秒的查询。",
    ),
    ("user", "JOIN 和子查询哪个性能更好？"),
    (
        "assistant",
        "没有绝对答案，取决于具体场景。大多数情况下 JOIN 更快——优化器可以对 JOIN 做更多重写优化。但子查询在处理 EXISTS/NOT EXISTS 时往往比 JOIN 更好。关键看 EXPLAIN 结果，不要猜。",
    ),
    ("user", "什么是数据库事务？什么时候需要用？"),
    (
        "assistant",
        "事务是一组「要么全成功、要么全失败」的操作。经典例子是银行转账：A 扣 100 元 + B 加 100 元必须同时成功，如果 B 加钱失败，A 的扣款也要回滚。用 BEGIN/COMMIT/ROLLBACK 控制，核心是 ACID 四个特性。",
    ),
    # ─ 话题 5: Docker 容器（第 21-25 轮）──
    ("user", "Docker 和虚拟机有什么区别？"),
    (
        "assistant",
        "虚拟机虚拟整个操作系统（包括内核），每个 VM 有自己的 OS，启动要几分钟。Docker 容器共享宿主机的内核，只隔离应用层，启动只要几秒。形象地说：VM 是一栋独栋别墅，Docker 是公寓楼里的一个房间——共享基础设施，但互相隔离。",
    ),
    ("user", "Dockerfile 里的 FROM 是干什么的？"),
    (
        "assistant",
        "FROM 指定你的镜像「基于谁构建」。比如 FROM python:3.11-slim 表示在 Python 3.11 官方镜像的基础上加你的代码。这个基础镜像里已经装好了 Python 解释器，你只需要 COPY 代码 + RUN pip install 就行了。",
    ),
    ("user", "docker-compose 解决什么问题？"),
    (
        "assistant",
        "当你的应用需要多个容器协作时（如 web 服务 + 数据库 + Redis 缓存），手动一个个 docker run 很痛苦。docker-compose 用一个 YAML 文件描述所有服务，一条命令 docker-compose up 全部启动，网络自动配置好。",
    ),
    ("user", "容器退出了数据还在吗？"),
    # ─ 话题 6: 部署（第 26-30 轮）──
    (
        "assistant",
        "默认情况下容器退出后数据消失——容器的文件系统是临时的。要持久化数据需要用 Volume：docker run -v /host/path:/container/path，把宿主机的目录「挂载」到容器里，数据就存在宿主机上。",
    ),
    ("user", "我代码写好了，怎么部署到服务器上让别人访问？"),
    (
        "assistant",
        "最简方案：1) 买一台云服务器（阿里云/AWS 学生优惠），2) 用 git 把代码拉到服务器，3) docker-compose up 启动服务，4) 配置 Nginx 反向代理把域名指向你的应用端口。整个流程熟练后 30 分钟搞定。",
    ),
    ("user", "Nginx 反向代理是干什么的？"),
    (
        "assistant",
        "Nginx 像一个「前台接待员」——外界请求先到 Nginx（端口 80/443），Nginx 再转发给后端的实际服务（如你的 FastAPI 跑在 8000 端口）。它还能做 HTTPS 证书管理、负载均衡、静态文件服务。",
    ),
    ("user", "HTTPS 证书怎么搞？要花钱吗？"),
    (
        "assistant",
        "不用花钱！Let's Encrypt 提供免费证书，用 certbot 工具一行命令自动申请和续期。证书 90 天过期，但 certbot 可以设置自动续期 cron job，完全不用手动管。",
    ),
    ("user", "部署完了怎么监控服务有没有挂？"),
    (
        "assistant",
        "基础方案：用 systemd 或 Docker 的 restart policy（--restart=always）保证进程挂了自动重启。进阶方案：用 Grafana + Prometheus 监控 CPU/内存/QPS，用 Uptime Kuma 或 Pingdom 做健康检查告警。",
    ),
]

# ─── 测试查询（覆盖不同检索需求）─────────────────────────

# 设计原则：
#   Q1-Q2 是「近期话题」—— Buffer 层应该命中
#   Q3-Q4 是「久远话题」—— Vector 层应该发力
#   Q5-Q6 是「跨话题总览」—— Summary 层应该提供背景

TEST_QUERIES = [
    {
        "query": "部署服务怎么监控？",
        "topic": "近期话题(部署)",
        "expected_layer": "Buffer",
        "keywords": ["Grafana", "Prometheus", "restart", "Uptime", "告警"],
    },
    {
        "query": "Docker 的数据持久化怎么做？",
        "topic": "近期话题(Docker)",
        "expected_layer": "Buffer",
        "keywords": ["Volume", "挂载", "持久化", "宿主机"],
    },
    {
        "query": "Python 列表怎么用？",
        "topic": "久远话题(Python)",
        "expected_layer": "Vector",
        "keywords": ["列表", "方括号", "增删改", "元组"],
    },
    {
        "query": "Git commit 怎么追加漏掉的文件？",
        "topic": "久远话题(Git)",
        "expected_layer": "Vector",
        "keywords": ["amend", "漏掉", "追加"],
    },
    {
        "query": "整段对话讨论了哪些技术？",
        "topic": "跨话题总览",
        "expected_layer": "Summary",
        "keywords": ["Python", "Git", "API", "数据库", "Docker", "部署"],
    },
    {
        "query": "API 分页怎么设计？",
        "topic": "中间话题(API)",
        "expected_layer": "Vector",
        "keywords": ["分页", "offset", "limit", "total_count"],
    },
]


def print_separator(title: str) -> None:
    """打印分隔标题。"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def format_tokens(n: int) -> str:
    """格式化 token 数量。"""
    return f"{n:,}"


def main() -> None:
    """运行 HybridMemory 演示。"""
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("  HybridMemory 三层混合记忆演示")
    print("=" * 60)
    print()
    print(f"模拟对话: {len(SIMULATED_CONVERSATION)} 条消息, {len(SIMULATED_CONVERSATION)//2} 轮")
    print(f"测试查询: {len(TEST_QUERIES)} 个")
    print()

    # ── 初始化 LLM ───────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"[初始化] LLM 客户端 (provider={provider})...")
    try:
        llm = LLMClient(provider=provider)
        print(f"         模型: {llm.model}")
    except Exception as e:
        print(f"         [警告] 创建失败: {e}")
        print("         Summary 层将不生成摘要。请检查 .env 中的 API key。")
        llm = None

    # ── 创建 HybridMemory ────────────────────────────────
    print()
    print("[初始化] 创建 HybridMemory（三层混合记忆）...")

    hybrid = HybridMemory(
        llm=llm,  # type: ignore[arg-type]
        recent_rounds=5,  # 最近 5 轮原文保留
        vector_top_k=3,  # 每次向量检索取 3 条
        token_budget={  # 三层 token 分配
            "buffer": 0.35,
            "summary": 0.30,
            "vector": 0.35,
        },
    )
    print("         HybridMemory [OK]")
    print("         配置: Buffer 5 轮 | Summary 压缩 | Vector top-3")
    print("         Token 预算: Buffer 35% / Summary 30% / Vector 35%")

    # ── 写入对话 ─────────────────────────────────────────
    print_separator("第一阶段: 写入 30 轮对话")

    t0 = time.time()
    for i, (role, content) in enumerate(SIMULATED_CONVERSATION, 1):
        hybrid.add(role, content)
        # 每 10 条打印一次进度
        if i % 10 == 0:
            stats = hybrid.get_stats()
            print(
                f"  写入 {i:2d}/{len(SIMULATED_CONVERSATION)} 条消息 | "
                f"Buffer: {stats['buffer_rounds']}轮 | "
                f"摘要: {stats['summary_chars']}字 | "
                f"Vector: {stats['vector_rounds']}条"
            )

    write_time = time.time() - t0
    stats = hybrid.get_stats()
    print()
    print(f"  写入完成, 耗时 {write_time:.2f}s")
    print(f"  总存储: {format_tokens(stats['total_tokens'])} tokens")

    # ── 对比单一 Memory ─────────────────────────────────
    print_separator("第二阶段: 检索测试")

    # 同时创建三种单一 Memory 作为对照组
    buffer_only = BufferMemory()
    vector_only = VectorMemory()
    summary_only = SummaryMemory(llm=llm, recent_rounds=5) if llm else None  # type: ignore[arg-type]

    for role, content in SIMULATED_CONVERSATION:
        buffer_only.add(role, content)
        vector_only.add(role, content)
        if summary_only:
            summary_only.add(role, content)

    for q in TEST_QUERIES:
        query = q["query"]
        topic = q["topic"]
        expected = q["expected_layer"]
        keywords = q["keywords"]

        print(f'--- 查询: "{query}" ({topic}) ---')
        print(f"    期望主导层: {expected}")
        print(f"    关键词: {keywords}")
        print()

        # HybridMemory 结果
        ctx = hybrid.get_context(query, max_tokens=2000)
        print(f"  [HybridMemory] 上下文 {len(ctx)} 字符:")
        # 缩进展示上下文
        for line in ctx.split("\n")[:15]:  # 最多展示 15 行
            print(f"    | {line}")
        if len(ctx.split("\n")) > 15:
            print(f"    | ... (共 {len(ctx.splitlines())} 行)")
        print()

        # 关键词命中检查
        ctx_lower = ctx.lower()
        hits = [kw for kw in keywords if kw.lower() in ctx_lower]
        misses = [kw for kw in keywords if kw.lower() not in ctx_lower]
        print(f"    关键词命中: {len(hits)}/{len(keywords)}")
        if hits:
            print(f"      命中: {', '.join(hits)}")
        if misses:
            print(f"      丢失: {', '.join(misses)}")

        # 对比单一 Memory 的命中率
        buffer_ctx = buffer_only.get_context(query, max_tokens=2000)
        buffer_hits = sum(1 for kw in keywords if kw.lower() in buffer_ctx.lower())
        vector_ctx = vector_only.get_context(query, max_tokens=2000)
        vector_hits = sum(1 for kw in keywords if kw.lower() in vector_ctx.lower())

        print(
            f"    对比: Buffer={buffer_hits}/{len(keywords)}, "
            f"Vector={vector_hits}/{len(keywords)}, "
            f"Hybrid={len(hits)}/{len(keywords)}"
        )
        print()

    # ── 统计对比 ─────────────────────────────────────────
    print_separator("第三阶段: 存储效率对比")

    buffer_stats = buffer_only.get_stats()
    vector_stats = vector_only.get_stats()

    print(f"  {'Memory 类型':20s} {'存储轮数':>8s} {'总 Token':>10s} {'摘要大小':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    print(
        f"  {'BufferMemory':20s} {buffer_stats['rounds']:>8} "
        f"{format_tokens(buffer_stats['total_tokens']):>10} {'N/A':>8}"
    )
    if summary_only:
        s_stats = summary_only.get_stats()
        print(
            f"  {'SummaryMemory':20s} {s_stats['rounds']:>8} "
            f"{format_tokens(s_stats['total_tokens']):>10} {s_stats['summary_chars']:>8}"
        )
    print(
        f"  {'VectorMemory':20s} {vector_stats['rounds']:>8} "
        f"{format_tokens(vector_stats['total_tokens']):>10} {'N/A':>8}"
    )
    print(
        f"  {'HybridMemory':20s} {stats['rounds']:>8} "
        f"{format_tokens(stats['total_tokens']):>10} {stats['summary_chars']:>8}"
    )
    print()
    print("  [分析] HybridMemory 同时维护了三份索引，存储开销最大，")
    print("         但换来的是三种检索策略的协同——这就是空间换时间(精度)。")
    print("         在生产环境中，摘要层和向量层可以持久化到磁盘，Buffer 层内存即可。")

    # ── 生成报告 ─────────────────────────────────────────
    print_separator("第四阶段: 生成报告")

    report_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "week6")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "hybrid_memory_demo.md")

    report = generate_report(
        conversation_rounds=len(SIMULATED_CONVERSATION) // 2,
        queries=TEST_QUERIES,
        hybrid_stats=stats,
        buffer_stats=buffer_stats,
        vector_stats=vector_stats,
        write_time=write_time,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  [OK] 报告已保存到: {report_path}")
    print()
    print("=" * 60)
    print("  演示完成")
    print("=" * 60)
    print()
    print(f"  完整报告: {report_path}")


def generate_report(
    conversation_rounds: int,
    queries: list[dict],
    hybrid_stats: dict,
    buffer_stats: dict,
    vector_stats: dict,
    write_time: float,
) -> str:
    """生成 Markdown 格式的演示报告。"""

    lines = [
        "# HybridMemory 三层混合记忆演示报告",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 一、架构概览",
        "",
        "HybridMemory 将三种记忆策略组合为一个三层架构:",
        "",
        "```",
        "用户提问: 'Python 列表怎么用?'",
        "    |",
        "    v",
        "HybridMemory.get_context(query, max_tokens=2000)",
        "    |",
        "    +-- [1] Summary 层 (30% = 600 tokens)",
        "    |       历史摘要: '对话涵盖了 Python基础、Git、API设计...'",
        "    |       作用: 提供宏观背景，知道对话讨论了哪些领域",
        "    |",
        "    +-- [2] Buffer 层 (35% = 700 tokens)",
        "    |       最近 5 轮原文: 部署相关讨论",
        "    |       作用: 保留当前话题的完整上下文",
        "    |",
        "    +-- [3] Vector 层 (35% = 700 tokens)",
        "    |       语义检索 'Python 列表怎么用' -> 找到第 1 轮对话",
        "    |       作用: 跨话题找回久远但相关的内容",
        "    |       去重: 过滤已在 Buffer 中的条目",
        "    |",
        "    v",
        "合并输出: [历史摘要] + [近期对话] + [相关历史记忆]",
        "```",
        "",
        "### 检索优先级",
        "",
        "| 优先级 | 层级 | 数据来源 | 适用场景 |",
        "|--------|------|---------|---------|",
        "| 1 (最高) | Buffer | 最近 N 轮原文 | 当前话题的延续性问题 |",
        "| 2 | Summary | LLM 压缩的早期对话 | 需要理解对话全貌的宽泛问题 |",
        "| 3 | Vector | 向量相似度检索 | 跨话题找回久远但相关的内容 |",
        "",
        "Buffer 排第一是因为: 正在聊的话题最可能被继续追问。",
        "如果当前话题和问题不相关，Vector 层会自动补位——",
        "这就是三层协同的价值。",
        "",
        "### Token 预算分配",
        "",
        "| 层级 | 占比 | 默认 2000 tokens 时的分配 |",
        "|------|------|--------------------------|",
        "| Buffer | 35% | 700 tokens |",
        "| Summary | 30% | 600 tokens |",
        "| Vector | 35% | 700 tokens |",
        "",
        "预算不是固定的——你可以根据场景调整:",
        "- 客服机器人: 降低 Vector，提高 Buffer（对话连续性强）",
        "- 知识问答: 提高 Vector，降低 Buffer（跨话题检索多）",
        "- 个人助理: 提高 Summary（需要理解全天对话全貌）",
        "",
        "---",
        "",
        "## 二、演示数据",
        "",
        f"- 模拟对话: {conversation_rounds} 轮, 6 个话题",
        f"- 测试查询: {len(queries)} 个（覆盖近期/久远/跨话题三种需求）",
        f"- 写入耗时: {write_time:.2f}s",
        "",
        "### 话题分布",
        "",
        "| 轮次 | 话题 | 内容 |",
        "|------|------|------|",
        "| 1-5 | Python 基础 | 列表/元组/字典/GIL |",
        "| 6-10 | Git 版本控制 | add/commit/merge/rebase/revert |",
        "| 11-15 | API 设计 | RESTful/URL设计/版本号/分页 |",
        "| 16-20 | 数据库 | 索引/慢查询/JOIN/事务 |",
        "| 21-25 | Docker 容器 | 镜像/Dockerfile/docker-compose/Volume |",
        "| 26-30 | 部署 | 云服务器/Nginx/HTTPS/监控 |",
        "",
        "### 测试查询设计",
        "",
        "| 查询 | 话题 | 期望主导层 | 原因 |",
        "|------|------|-----------|------|",
    ]

    for q in queries:
        lines.append(f"| {q['query']} | {q['topic']} | {q['expected_layer']} | - |")

    lines += [
        "",
        "---",
        "",
        "## 三、存储效率对比",
        "",
        "| Memory 类型 | 存储轮数 | 总 Token | 摘要大小 |",
        "|------------|---------|---------|---------|",
        f"| BufferMemory | {buffer_stats['rounds']} | {buffer_stats['total_tokens']:,} | N/A |",
        f"| VectorMemory | {vector_stats['rounds']} | {vector_stats['total_tokens']:,} | N/A |",
        f"| HybridMemory | {hybrid_stats['rounds']} | {hybrid_stats['total_tokens']:,} | {hybrid_stats['summary_chars']} |",
        "",
        "### 分析",
        "",
        "- HybridMemory 存储开销最大——它同时维护三份索引（消息列表 + ChromaDB + 摘要文本）",
        "- 这是空间换精度的权衡: 更多的存储，换来更全面的检索能力",
        "- 生产环境中: Buffer 和 Summary 存在内存中，Vector 存在磁盘上（ChromaDB 持久化模式）",
        "- 对于 30 轮对话，HybridMemory 的额外开销完全可以接受",
        "",
        "---",
        "",
        "## 四、三层协同的核心机制",
        "",
        "### 1. Token 预算分配",
        "",
        "不是「谁先占满算谁的」，而是提前划分好每层的额度:",
        "",
        "```python",
        "buffer_budget  = int(max_tokens * 0.35)  # 700 tokens for 2000 total",
        "summary_budget = int(max_tokens * 0.30)  # 600 tokens",
        "vector_budget  = int(max_tokens * 0.35)  # 700 tokens",
        "```",
        "",
        "每层拿到自己的额度后独立截断，互不影响。",
        "这样保证 Summary 层始终有空间展示，不会被 Buffer 完全挤掉。",
        "",
        "### 2. 去重机制",
        "",
        "Vector 检索可能找到「已经在 Buffer 中的」消息——同一轮对话不需要出现两次:",
        "",
        "```python",
        'recent_set = {f"{r}: {c}" for r, c in recent}  # Buffer 中的消息',
        "filtered_docs = [d for d in vector_docs if d not in recent_set]  # 去重",
        "```",
        "",
        "### 3. 摘要触发策略",
        "",
        "不是每轮对话都调 LLM 生成摘要——而是在「超出 Buffer 窗口的消息」累积到",
        "recent_rounds 条时才触发一次。对于 30 轮对话，大约摘要 2-3 次，不会造成 API 浪费。",
        "",
        "### 4. 输出顺序",
        "",
        "Summary -> Buffer -> Vector 的顺序是故意的:",
        "- Summary 先出场: 给 LLM 一个宏观背景",
        "- Buffer 紧接着: 当前话题的细节",
        "- Vector 收尾: 补充久远但相关的信息",
        "",
        "这个顺序让 LLM 先理解「整体在聊什么」，再看「现在在聊什么」，",
        "最后看「过去还有什么相关的」——和人类回忆的顺序一致。",
        "",
        "---",
        "",
        "## 五、适用场景与选型建议",
        "",
        "| 场景 | 推荐配置 | 原因 |",
        "|------|---------|------|",
        "| 客服机器人 | buffer 40% / summary 20% / vector 40% | 需要从历史中找相似问题 |",
        "| 个人助理 | buffer 30% / summary 40% / vector 30% | 需要全天对话的宏观理解 |",
        "| 代码审查 Agent | buffer 25% / summary 25% / vector 50% | 需要在大量代码中检索相关片段 |",
        "| 教育辅导 | buffer 40% / summary 30% / vector 30% | 当前知识点最重要，但需要回顾前置知识 |",
        "",
        "### 什么时候不需要 Hybrid?",
        "",
        "- 对话轮数 < 20: Buffer 就够了，Hybrid 是过度设计",
        "- 单话题对话: Vector 的价值很低，因为不需要跨话题检索",
        "- 成本极度敏感: Summary 的 LLM 调用和 Vector 的 Embedding 都有成本",
        "",
        "---",
        "",
        "## 六、与 Day 1 三种 Memory 的关系",
        "",
        "HybridMemory 不是替代品，而是组合器。它内部的思想来自三种 Memory:",
        "",
        "- BufferMemory -> 提供「保留最近原文」的思路",
        "- SummaryMemory -> 提供「LLM 压缩早期对话」的思路",
        "- VectorMemory -> 提供「语义检索历史」的思路",
        "",
        "区别在于: HybridMemory 把这些思路放在一个类里，协调它们之间的 token 分配、",
        "去重、输出顺序。把三个各有所长的工具组合成一个「没有明显短板」的系统。",
        "",
        "---",
        "",
        "## 七、核心代码结构",
        "",
        "```python",
        "class HybridMemory(BaseMemory):",
        "    def __init__(self, llm, embedder, recent_rounds, vector_top_k, token_budget):",
        "        self._messages = []          # Buffer 层: 消息列表",
        '        self._summary = ""           # Summary 层: 压缩摘要',
        "        self._collection = ...       # Vector 层: ChromaDB 向量库",
        "",
        "    def add(self, role, content):",
        "        # 1. 追加到 Buffer (消息列表)",
        "        # 2. 嵌入并写入 Vector (ChromaDB)",
        "        # 3. 检查是否需要触发摘要更新",
        "",
        "    def get_context(self, query, max_tokens):",
        "        # 1. 分配每层的 token 预算",
        "        # 2. 从 Summary 层取摘要（截断到预算）",
        "        # 3. 从 Buffer 层取最近原文（截断到预算）",
        "        # 4. 从 Vector 层做语义检索（去重后截断到预算）",
        "        # 5. 按 Summary -> Buffer -> Vector 顺序拼接输出",
        "```",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
