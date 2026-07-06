"""Memory 机制对比演示。

模拟一段 20 轮的客服对话，分别用 Buffer / Summary / Vector 三种 Memory 管理，
然后查询 4 个不同话题，对比每种 Memory 的 token 消耗和检索精准度。

运行:
  poetry run python src/study_agent/memory/demo.py

输出:
  - 终端打印实时评测结果
  - 生成 docs/week6/memory_comparison_report.md 报告文件
"""

from __future__ import annotations

import os

from study_agent.llm.client import LLMClient
from study_agent.memory.base import BaseMemory
from study_agent.memory.buffer import BufferMemory
from study_agent.memory.comparator import MemoryComparator
from study_agent.memory.summary import SummaryMemory
from study_agent.memory.vector import VectorMemory

# ─── 模拟对话数据 ────────────────────────────────────────

# 一段客服对话，覆盖 4 个话题，每个话题 5 轮（一问一答算 2 轮消息）
# 话题故意交错排列，模拟真实场景——用户不会把一个问题问透了再换下一个

SIMULATED_CONVERSATION: list[tuple[str, str]] = [
    # ─ 话题 1: 订单查询 ─
    ("user", "你好，我想查一下我的订单状态，订单号是 ORD-8821。"),
    (
        "assistant",
        "好的，正在为您查询订单 ORD-8821... 查到啦，您的订单目前处于「已发货」状态，预计 3 月 15 日送达。",
    ),
    ("user", "太好了，能帮我查一下物流单号吗？"),
    ("assistant", "当然，您的物流单号是 SF1234567890，承运商是顺丰速运。"),
    ("user", "我在订单里加了一个无线鼠标，这个鼠标的保修期是多久？"),
    ("assistant", "您购买的罗技 MX Master 3S 鼠标享受 2 年官方保修，保修起始日期是订单签收当天。"),
    ("user", "发货地址是我公司地址吗？我担心周末没人签收。"),
    (
        "assistant",
        "是的，配送地址是「北京市朝阳区建国路 88 号 SOHO 现代城 A 座 15 层」。如果您想改为家庭地址，我现在就可以帮您操作。",
    ),
    # ─ 话题 2: 产品技术规格 ─
    ("user", "对了，我买的那台笔记本是什么配置来着？"),
    (
        "assistant",
        "您订购的是 ThinkPad X1 Carbon Gen 11，配置如下：Intel Core i7-1365U / 16GB RAM / 512GB SSD / 14 英寸 2.8K OLED 屏。",
    ),
    ("user", "这个 i7-1365U 和 i7-13700H 有什么区别？"),
    (
        "assistant",
        "i7-1365U 是低功耗处理器，TDP 只有 15W，适合轻薄本；i7-13700H 是标压处理器，TDP 45W，适合游戏本和移动工作站。您的 X1 Carbon 用 U 系列是为了兼顾续航和散热。",
    ),
    ("user", "内存是焊接的还是可扩展的？"),
    (
        "assistant",
        "X1 Carbon 的内存是板载焊接的，不可自行升级。不过 16GB LPDDR5 对日常办公和开发完全够用。如果需要 32GB，需要下单时选高配版本。",
    ),
    ("user", "屏幕色域呢？我做设计对色彩要求比较高。"),
    (
        "assistant",
        "您选的 2.8K OLED 屏覆盖 100% DCI-P3 色域，支持 HDR True Black 500，出厂经过逐台校色，Delta E < 2——做设计完全没问题。",
    ),
    # ─ 话题 3: 退换货政策 ─
    ("user", "如果收到后发现屏幕有坏点，可以退货吗？"),
    (
        "assistant",
        "当然可以。我们有 7 天无理由退货和 15 天质量问题换货政策。屏幕坏点属于质量缺陷，在 15 天内可以免费换新。如果超过 15 天但在保修期内，可以免费维修。",
    ),
    ("user", "退货的运费谁承担？"),
    (
        "assistant",
        "如果是质量问题（如坏点、无法开机），运费由我们承担，顺丰上门取件。如果是无理由退货，需要您自行承担退货运费，通常 20-30 元。",
    ),
    ("user", "那我需要保留原包装吗？"),
    (
        "assistant",
        "是的，退货时需要完整的原包装——包括包装盒、泡沫内衬、说明书和所有配件。缺少包装可能会影响退款金额。建议收到货后至少保留包装 15 天。",
    ),
    ("user", "退款多久能到账？"),
    (
        "assistant",
        "仓库签收退货后 1-3 个工作日质检，质检通过后 3-5 个工作日退款到原支付方式。总共大约 5-8 个工作日。",
    ),
    # ─ 话题 4: 账户管理 ─
    ("user", "我想改一下账户的绑定手机号，原来的号码不用了。"),
    (
        "assistant",
        "可以的。修改绑定手机号需要进行身份验证——我们会向您当前绑定的手机号发送验证码。如果您已经收不到验证码，需要通过人工审核流程。",
    ),
    ("user", "我确实收不到验证码了，走人工审核吧。需要提供什么？"),
    (
        "assistant",
        "人工审核需要您上传身份证正反面照片，以及最近一笔订单的支付截图。审核通常 1-2 个工作日。审核通过后会给您的新手机号发验证短信完成绑定。",
    ),
    ("user", "那积分呢？换了手机号积分还在吗？"),
    (
        "assistant",
        "积分是跟账户走的，不是跟手机号。换绑手机号不会影响您现有的 3,280 积分。其实再攒 720 积分就可以兑换 50 元优惠券了。",
    ),
    ("user", "好的，那我先去准备材料。最后一个问题——你们有 App 吗？"),
    (
        "assistant",
        "有的！在 App Store 和各大安卓应用商店搜索「速达商城」即可下载。App 支持订单追踪、消息推送、一键售后，比网页版方便很多。",
    ),
]

# ─── 测试查询 ────────────────────────────────────────────

TEST_QUERIES = [
    {
        "topic": "订单查询",
        "query": "我的订单号是多少？配送地址是哪里？",
        "keywords": ["ORD-8821", "建国路", "SOHO"],
    },
    {
        "topic": "产品规格",
        "query": "我买的笔记本是什么处理器？内存多大？",
        "keywords": ["i7-1365U", "16GB", "X1 Carbon"],
    },
    {
        "topic": "退换货政策",
        "query": "退货需要什么条件？运费谁出？",
        "keywords": ["无理由退货", "原包装", "顺丰"],
    },
    {
        "topic": "账户管理",
        "query": "怎么修改绑定的手机号？我的积分还有多少？",
        "keywords": ["人工审核", "身份证", "3,280"],
    },
]


def main() -> None:
    """运行 Memory 对比演示。"""
    # 确保 .env 已加载
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("  Agent Memory 机制对比评测")
    print("=" * 60)
    print()
    print(f"模拟对话: {len(SIMULATED_CONVERSATION)} 条消息")
    print(f"测试查询: {len(TEST_QUERIES)} 个话题")
    print()

    # ── 创建 LLM 客户端（SummaryMemory 需要）──
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"[初始化] 创建 LLM 客户端 (provider={provider})...")
    try:
        llm = LLMClient(provider=provider)
        print(f"         模型: {llm.model}")
    except Exception as e:
        print(f"         [警告] LLM 客户端创建失败: {e}")
        print("         SummaryMemory 将跳过。可用的 provider 请检查 .env 配置。")
        llm = None

    # ── 创建三种 Memory ────────────────────────────────
    print()
    print("[初始化] 创建 Memory 实例...")

    buffer = BufferMemory()
    print("         BufferMemory [OK]")

    summary = None
    if llm:
        summary = SummaryMemory(llm=llm, recent_rounds=5)
        print("         SummaryMemory [OK]")
    else:
        print("         SummaryMemory [跳过—无可用 LLM]")

    vector = VectorMemory()
    print("         VectorMemory  [OK]")

    # ── 构建 Memory 字典 ───────────────────────────────
    memories: dict[str, BaseMemory] = {
        "BufferMemory": buffer,
    }
    if summary:
        memories["SummaryMemory"] = summary
    memories["VectorMemory"] = vector

    # ── 运行对比 ──────────────────────────────────────
    comparator = MemoryComparator()
    comparator.run(
        conversations=SIMULATED_CONVERSATION,
        queries=TEST_QUERIES,
        memories=memories,
    )

    # ── 生成报告 ──────────────────────────────────────
    print()
    report = comparator.generate_report()

    # 确保输出目录存在
    report_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "week6")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "memory_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[OK] 报告已保存到: {report_path}")

    # ── 打印终端摘要 ──────────────────────────────────
    print()
    print("=" * 60)
    print("  终端摘要")
    print("=" * 60)
    print()

    results = comparator._results
    if results:
        for r in results:
            s = r["stats"]
            print(
                f"  {r['name']:20s} | "
                f"存储 {s['total_tokens']:>5,} tokens | "
                f"命中率 {r['avg_hit_rate']:.0%} | "
                f"上下文 {r['avg_context_tokens']:.0f} tokens"
            )

    print()
    print(f"完整报告请查看: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
