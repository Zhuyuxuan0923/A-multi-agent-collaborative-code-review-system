"""
Week 3 Day 5 — 生成与引用(Generation & Citation)Demo

运行: poetry run python src/study_agent/w3d5_generation_demo.py

RAG 链路最后一步，四个环节：
  检索(Retrieve) -> 增强(Augment) -> 生成(Generate) -> 引用(Cite)

今天聚焦后三步——检索部分复用 Day 4。

Demo 结构：
  Part 1: 纯 LLM 无 RAG -> 看看幻觉长什么样
  Part 2: RAG 但不引用 -> 检索到正确资料，但无法追溯
  Part 3: RAG + 引用 -> [1] [2] 标注，可验证
  Part 4: 引用解析 -> 从答案中提取编号，映射回原文档
  Part 5: 边界情况 -> 知识库没有答案时怎么办
  Part 6: 端到端管道 -> 一个函数搞定全流程(Week 4 项目直接复用)
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

import chromadb

from study_agent.llm.client import LLMClient
from study_agent.prompt.templates import PromptManager

# ==================================================================
# 第 0 部分：知识库(复用 Day 4 的 CORPUS)和简易检索器
# ==================================================================

CORPUS = [
    # 0-3: 报销相关
    "员工报销流程：在系统中填写报销申请单，附上发票照片，提交给直属上级审批。审批通过后财务部在 5 个工作日内打款到工资卡。",
    "出差报销标准：一线城市住宿标准 500 元每晚，二线城市 350 元每晚。交通费实报实销，餐补每天 100 元。所有报销需提供正规发票。",
    "招待费报销规定：招待客户需提前申请，人均标准不超过 200 元。报销时需注明招待对象、事由和参与人员名单。",
    "报销单填写规范：发票抬头必须为公司全称，发票日期需在 3 个月以内。电子发票需打印后贴在 A4 纸上，附在报销单后面。",
    # 4-7: 请假/考勤
    "年假政策：入职满 1 年享有 5 天带薪年假，满 3 年 10 天，满 5 年 15 天。年假需提前一周向直属上级申请。",
    "病假规定：请病假需提供二级以上医院开具的病假证明。紧急情况可在 24 小时内补交证明。病假期间工资按国家规定执行。",
    "事假管理：事假每年累计不超过 10 天。超过 3 天的事假需提前 3 个工作日申请，并经部门负责人审批。",
    "考勤打卡制度：公司实行弹性工作制，核心工作时间 10:00-16:00。忘记打卡需在 24 小时内通过企业微信提交补卡申请。",
    # 8-11: IT/技术
    "办公网络配置：公司 Wi-Fi 密码为 Study@2026，支持 5G 频段。访客网络密码请向前台索取，有效期 24 小时。",
    "VPN 使用指南：远程办公需使用公司 VPN 连接内网。VPN 客户端下载地址见 IT 知识库首页，使用企业邮箱账号登录。",
    "信息安全规范：公司配发的电脑不得安装未经授权的软件。所有内部文档禁止通过个人微信、QQ 或私人邮箱外传。",
    "代码提交规范：所有代码提交前必须通过 Ruff lint 检查和 Black 格式化。PR 需至少一位同事 Code Review 通过后方可合并。",
    # 12-15: 招聘/入职
    "招聘面试流程：技术岗位面试分为技术面、算法面、HR 面三轮。非技术岗位为 HR 面和部门负责人面两轮。",
    "入职手续办理：入职第一天需携带身份证原件、学历学位证书复印件、离职证明、体检报告到 HR 部门办理入职手续。",
    "试用期管理规定：试用期为 3 个月，表现优秀者可提前转正。试用期工资为转正工资的 80%。试用期内双方均可提前 3 天通知解除劳动关系。",
    "培训与发展：新员工入职培训为期 2 天，涵盖企业文化、规章制度、信息安全等内容。每月举办一次技术分享会。",
    # 16-19: 薪酬/福利
    "薪资发放说明：每月 10 号发放上月工资，如遇节假日顺延至下一个工作日。工资条通过企业微信推送，请注意查收。",
    "五险一金缴纳：公司按照国家规定为员工缴纳五险一金，缴纳基数每年 7 月调整一次。详情可咨询 HR 薪酬专员。",
    "加班与调休：工作日加班可安排调休或领取加班费。周末加班优先安排调休。调休需在加班后一个月内使用完毕。",
    "节日福利政策：春节、中秋、端午三大节日发放节日礼品或购物卡。员工生日当月发放生日蛋糕券。",
]

# ChromaDB 简易向量检索器(Day 2-4 的技术，这里只做轻量封装)


def _make_doc_vector(idx: int) -> list[float]:
    """8 维示意向量(生产环境用 OpenAI Embedding 替换此函数即可)"""
    vec = [0.01] * 8
    if idx <= 3:
        vec[0] = 0.85 + idx * 0.03
        vec[1] = 0.15
    elif idx <= 7:
        vec[2] = 0.85 + (idx - 4) * 0.03
        vec[3] = 0.12
    elif idx <= 11:
        vec[4] = 0.85 + (idx - 8) * 0.03
        vec[5] = 0.12
    elif idx <= 15:
        vec[6] = 0.80 + (idx - 12) * 0.04
        vec[7] = 0.15
    else:
        vec[6] = 0.60 + (idx - 16) * 0.05
        vec[7] = 0.50 + (idx - 16) * 0.05
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class SimpleRetriever:
    """简易 ChromaDB 检索器——和 Day 2-4 同样的逻辑，只做轻量封装。"""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="w3d5_retrieval",
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = [_make_doc_vector(i) for i in range(len(corpus))]
        self.collection.add(
            documents=corpus,
            embeddings=embeddings,
            ids=[f"doc-{i}" for i in range(len(corpus))],
        )

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, str, float]]:
        """返回 [(doc_idx, text, similarity), ...]"""
        # 用查询所属主题生成向量
        results = self.collection.query(
            query_embeddings=[_make_doc_vector(self._guess_topic(query))],
            n_results=top_k,
        )
        hits = []
        for doc_id, text, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            idx = int(doc_id.split("-")[1])
            hits.append((idx, text, round(1.0 - dist, 4)))
        return hits

    @staticmethod
    def _guess_topic(query: str) -> int:
        """根据查询关键词猜测主题索引(用于生成合适的示意向量)"""
        q = query.lower()
        if any(w in q for w in ["报销", "发票", "出差", "住宿", "招待"]):
            return 1  # 报销类向量
        if any(w in q for w in ["请假", "病假", "年假", "事假", "考勤", "打卡", "看病", "发烧"]):
            return 5  # 请假类向量
        if any(
            w in q for w in ["wifi", "wi-fi", "vpn", "密码", "网络", "上网", "代码", "提交", "安全"]
        ):
            return 8  # IT 类向量
        if any(w in q for w in ["入职", "面试", "招聘", "试用期", "转正", "培训"]):
            return 13  # 入职类向量
        if any(
            w in q for w in ["工资", "薪资", "发薪", "五险一金", "加班", "调休", "福利", "节日"]
        ):
            return 16  # 薪酬类向量
        return 0  # 默认


# ==================================================================
# 第 1 部分：纯 LLM —— 没有 RAG 时，LLM 会怎样？
# ==================================================================


def demo_no_rag(llm: LLMClient) -> None:
    """演示：不提供任何资料，LLM 如何回答关于"公司内部规定"的问题。

    一个外面的大模型(GPT / Claude / DeepSeek)不可能知道你公司的内部规定。
    但它不会说「我不知道」——它会编。这就叫幻觉(Hallucination)。
    """
    print("\n" + "=" * 65)
    print("  Part 1: 纯 LLM(无 RAG)—— 幻觉长什么样？")
    print("=" * 65)

    question = "公司 Wi-Fi 密码是什么？如果忘记打卡了怎么办？"

    print("\n  [LLM] 问 LLM(不提供任何公司资料)：")
    print(f'  "{question}"')
    print("\n  [!] LLM 不知道你公司的真实规定，但它会「猜」一个听起来合理的答案...\n")

    # 给一个通用的 system prompt，但不提供公司资料
    system = "你是一个乐于助人的助手，请根据你的知识回答问题。"

    try:
        answer = llm.chat(question, system=system)
        print("  LLM 回答：")
        for line in answer.strip().split("\n"):
            print(f"    {line}")
    except Exception as e:
        print(f"  [!] API 调用失败(可能没配 Key，不影响理解概念): {e}")
        print("\n  [提示] 模拟回答(典型的 LLM 幻觉)：")
        print('    公司 Wi-Fi 密码通常是 "admin123" 或贴在路由器背面。')
        print("    忘记打卡可以联系 HR 或行政补卡。")
        print("    [!] 以上内容听起来合理，但**完全不是**你公司的真实规定！")

    print("\n  ,-------------------------------------------------,")
    print("  | 关键认知：                                        |")
    print("  | LLM 的知识截止于训练日期，它不知道你公司的内规。    |")
    print("  | 它会编造听起来合理的答案——这就是「幻觉」。           |")
    print("  | RAG 要解决的就是这个问题：先检索真实资料，          |")
    print("  | 再让 LLM 基于资料回答，而不是凭记忆。              |")
    print("  ,-------------------------------------------------'")


# ==================================================================
# 第 2 部分：RAG 但不引用 —— 检索到了正确答案，但没法验证
# ==================================================================


def demo_rag_no_citation(llm: LLMClient, retriever: SimpleRetriever) -> None:
    """演示：检索了资料并喂给 LLM，但没让它标注来源。

    这样做的答案可能是对的，但用户没法验证。
    问 "Wi-Fi 密码" 时 LLM 说 Study@2026 —— 这是真的吗？还是编的？
    没有引用标注，用户只能选择全部相信 或 全部不信。
    """
    print("\n" + "=" * 65)
    print("  Part 2: RAG 但不引用 —— 答案对但无法验证")
    print("=" * 65)

    question = "出差去北京住酒店，一晚能报销多少钱？"

    # Step 1: 检索
    print("\n  [检索] Step 1 — 检索相关文档：")
    hits = retriever.search(question, top_k=3)
    for idx, text, score in hits:
        print(f"    命中 doc-{idx} (sim={score:.3f}): {text[:60]}...")

    # Step 2: 拼接 Prompt(但不要求引用)
    context_parts = [text for _, text, _ in hits]
    context_block = "\n\n".join(f"资料 {i+1}: {text}" for i, text in enumerate(context_parts))
    prompt = f"""{context_block}

请根据以上资料回答：{question}"""

    print("\n  [提示] Step 2 — Prompt(不含引用要求)：")
    print(f"    {prompt[:200]}...")

    try:
        answer = llm.chat(prompt)
        print("\n  [LLM] Step 3 — LLM 回答：")
        for line in answer.strip().split("\n"):
            print(f"    {line}")

        print("\n  ,-------------------------------------------------,")
        print("  | 问题：答案说「一线城市 500 元/晚」——这是真的吗？         |")
        print("  | 用户不知道。ta 必须自己去翻原始资料才能验证。         |")
        print("  | 如果 LLM 编了一个「800 元/晚」呢？用户无从察觉。        |")
        print("  | 引用标注(Citation)就是为了解决这个问题。            |")
        print("  ,-------------------------------------------------'")
    except Exception as e:
        print(f"  [!] API 调用失败: {e}")


# ==================================================================
# 第 3 部分：RAG + 引用 —— 正确做法
# ==================================================================


def demo_rag_with_citation(llm: LLMClient, retriever: SimpleRetriever) -> None:
    """演示正确的 RAG 生成：检索 -> 增强 Prompt -> 生成 -> 引用。

    这是你今天要掌握的"标准流程"——Week 4 做项目时，所有 RAG 问答都走这个流程。
    """
    print("\n" + "=" * 65)
    print("  Part 3: RAG + 引用 —— 标准流程")
    print("=" * 65)

    question = "入职满 3 年的员工有多少天年假？请病假需要什么证明？"

    # Step 1: 检索
    print("\n  [检索] Step 1 — 检索：")
    hits = retriever.search(question, top_k=4)
    for idx, text, score in hits:
        print(f"    doc-{idx} (sim={score:.3f}): {text[:60]}...")

    # Step 2: 用 Jinja2 模板渲染 Prompt
    chunk_texts = [text for _, text, _ in hits]
    manager = PromptManager("src/study_agent/prompt/templates")
    prompt = manager.render(
        "rag_generation",
        role="公司制度咨询专家",
        chunks=chunk_texts,
        question=question,
    )

    print("\n  [提示] Step 2 — 渲染后的 Prompt：")
    print(f"  {'-' * 55}")
    # 缩进显示 prompt
    for line in prompt.strip().split("\n"):
        print(f"  | {line}")
    print(f"  {'-' * 55}")

    # Step 3: 调用 LLM 生成
    print("\n  [LLM] Step 3 — LLM 生成(基于资料，带引用)：")
    try:
        # prompt 本身已包含 system 内容(role + 参考资料 + 回答要求)，
        # 所以整个 prompt 作为 user message 发送
        answer = llm.chat(prompt)
        print()
        for line in answer.strip().split("\n"):
            print(f"    {line}")

        print("\n  ,-------------------------------------------------,")
        print("  | [OK] 答案中出现了 [1] [4] 等编号                    |")
        print("  | -> 你可以直接追溯到 doc-4(年假政策)               |")
        print("  | -> 也可以追溯 [1] 对应哪条资料                      |")
        print("  | -> 每条关键信息都可验证，不再「全信或全不信」         |")
        print("  ,-------------------------------------------------'")
    except Exception as e:
        print(f"  [!] API 调用失败: {e}")
        print("\n  [提示] 模拟回答(展示了引用格式)：")
        print("    入职满 3 年的员工享有 10 天带薪年假 [1]。")
        print("    请病假需提供二级以上医院开具的病假证明 [4]。")
        print("    紧急情况可在 24 小时内补交证明 [4]。")
        print("    注意：年假需提前一周向直属上级申请 [1]。")
        print("    ")


# ==================================================================
# 第 4 部分：引用解析 —— 把 [1] [2] 变成可点击的溯源卡片
# ==================================================================


@dataclass
class CitationResult:
    """一个带引用的 RAG 生成结果。"""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    # citations = [{"number": 1, "text": "原文...", "doc_idx": 5}, ...]


def parse_citations(answer: str, source_map: dict[int, str]) -> CitationResult:
    """从 LLM 回复中提取引用编号，映射回原文档。

    answer: LLM 生成的文本(含 [1] [2] 等标注)
    source_map: 编号 -> 原文档内容的映射，如 {1: "年假政策：入职满1年...", 2: "病假规定：请病假..."}

    返回 CitationResult，包含原答案和引用详情列表。
    """
    # 找到所有被引用的编号
    cited_numbers = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        cited_numbers.add(int(match.group(1)))

    citations = []
    for num in sorted(cited_numbers):
        if num in source_map:
            citations.append(
                {
                    "number": num,
                    "text": source_map[num],
                }
            )

    return CitationResult(answer=answer, citations=citations)


def demo_citation_parsing() -> None:
    """演示：如何解析 LLM 回复中的 [1] [2] 并映射回原文档。"""
    print("\n" + "=" * 65)
    print("  Part 4: 引用解析 —— 把 [1] 变成可追溯的原文")
    print("=" * 65)

    # 模拟 LLM 返回的带引用的答案
    mock_answer = textwrap.dedent(
        """\
        入职满 3 年的员工享有 10 天带薪年假 [1]。请假需提前一周向直属上级申请 [1]。
        关于病假，需要提供二级以上医院开具的病假证明 [2]。紧急情况可在 24 小时内补交证明 [2]。
        关于事假，每年累计不超过 10 天 [3]。"""
    ).strip()

    # 模拟第一次检索的结果(编号 -> 原文映射)
    source_map = {
        1: "年假政策：入职满 1 年享有 5 天带薪年假，满 3 年 10 天，满 5 年 15 天。年假需提前一周向直属上级申请。",
        2: "病假规定：请病假需提供二级以上医院开具的病假证明。紧急情况可在 24 小时内补交证明。",
        3: "事假管理：事假每年累计不超过 10 天。超过 3 天的事假需提前 3 个工作日申请。",
    }

    result = parse_citations(mock_answer, source_map)

    print("\n  [文档] LLM 回复：")
    for line in result.answer.split("\n"):
        print(f"    {line}")

    print(f"\n  [解析] 解析出的引用(共 {len(result.citations)} 条)：")
    for c in result.citations:
        print(f"    [{c['number']}] -> {c['text'][:70]}...")

    print("\n  ,-------------------------------------------------,")
    print("  | 引用解析的意义：                                     |")
    print("  | 1. 用户看到 [1] 可以点击/跳转到原文                  |")
    print("  | 2. 如果用户怀疑答案，可以直接查看引用对应的文档       |")
    print("  | 3. 答案和引用分离——答案可读，引用可追溯             |")
    print("  ,-------------------------------------------------'")


# ==================================================================
# 第 5 部分：边界情况 —— 资料里没有答案怎么办？
# ==================================================================


def demo_edge_cases(llm: LLMClient, retriever: SimpleRetriever) -> None:
    """演示 RAG 的两个经典边界情况。

    边界 1：资料里没有相关答案 -> LLM 应该说「我不知道」
    边界 2：多条资料给出不同信息 -> LLM 应该综合并标注来源
    """
    print("\n" + "=" * 65)
    print("  Part 5: 边界情况处理")
    print("=" * 65)

    manager = PromptManager("src/study_agent/prompt/templates")

    # -- 边界 1：知识库没有答案 --
    print("\n  【边界 1：知识库覆盖不到的问题】")
    question_out_of_scope = "公司附近有什么好吃的餐厅？"

    print(f'  查询: "{question_out_of_scope}"')
    print("  说明: 知识库里是公司制度文档，没有餐厅信息")

    hits = retriever.search(question_out_of_scope, top_k=3)
    print("\n  检索结果(相似度都很低，因为确实没有相关文档)：")
    for idx, text, score in hits:
        print(f"    doc-{idx} (sim={score:.4f}): {text[:55]}...")
    print(f"  注意: 最高的相似度也只有 {hits[0][2]:.3f}——说明确实找不到相关内容")

    chunks = [text for _, text, _ in hits]
    prompt = manager.render(
        "rag_generation",
        role="公司制度咨询专家",
        chunks=chunks,
        question=question_out_of_scope,
    )

    print("\n  Prompt 中的关键指令：")
    print('    "如果参考资料中没有相关信息，直接说根据现有资料无法回答，不要编造"')

    try:
        answer = llm.chat(prompt)
        print("\n  LLM 回答：")
        for line in answer.strip().split("\n"):
            print(f"    {line}")
    except Exception:
        print("\n  模拟回答: 根据现有资料无法回答这个问题。知识库中的资料涵盖")
        print("  报销、请假、IT、入职、薪酬等公司制度，不包含周边餐厅信息。")

    print("\n  [OK] 关键设计：Prompt 模板中的'无法回答'指令防止了 LLM 编造答案。")

    # -- 边界 2：相似度阈值过滤 --
    print("\n  【边界 2：相似度过滤 —— 防止低质量资料混入 Prompt】")
    print("  策略: 只保留相似度 >= 阈值的文档块")
    print("  阈值 0.6: sim < 0.6 的块被丢弃，不进入 Prompt")
    print("  原因: 低相似度的块和问题无关，塞给 LLM 反而会误导它。")

    # 用一个有明确答案的问题演示
    question_good = "公司 Wi-Fi 密码是什么？"
    hits_good = retriever.search(question_good, top_k=5)

    print(f"\n  查询: {question_good}")
    print(f"  {'doc-idx':<8} {'similarity':<12} {'pass?':<8}")
    print(f"  {'-'*35}")
    for idx, text, score in hits_good:
        passed = "[OK] 保留" if score >= 0.6 else "[X] 丢弃"
        print(f"  doc-{idx:<4} {score:<12.4f} {passed:<8}")


# ==================================================================
# 第 6 部分：端到端管道 —— 一个函数搞定全流程
# ==================================================================


@dataclass
class RAGResult:
    """RAG 管道的一次完整结果。"""

    question: str
    answer: str
    sources: list[dict[str, Any]]  # [{"idx": 0, "text": "...", "similarity": 0.92}, ...]
    citations: list[dict[str, Any]]  # [{"number": 1, "text": "原文..."}, ...]


def rag_pipeline(
    question: str,
    retriever: SimpleRetriever,
    llm: LLMClient,
    top_k: int = 5,
    sim_threshold: float = 0.6,
) -> RAGResult:
    """端到端 RAG 管道：检索 -> 增强 -> 生成 -> 引用解析。

    这是一个完整的、可复用的函数。Week 4 做项目时可以直接搬过去用。

    参数：
      question: 用户的问题
      retriever: 检索器(向量/关键词/混合都行，只要有 .search() 方法)
      llm: LLM 客户端
      top_k: 检索返回的文档数
      sim_threshold: 相似度阈值，低于此值的文档块丢弃

    返回：
      RAGResult: 包含答案、来源、引用详情的完整结果
    """
    # Step 1: 检索 + 相似度过滤
    all_hits = retriever.search(question, top_k=top_k)
    hits = [(idx, text, sim) for idx, text, sim in all_hits if sim >= sim_threshold]

    if not hits:
        return RAGResult(
            question=question,
            answer="根据现有资料无法回答您的问题。",
            sources=[],
            citations=[],
        )

    # 构建编号 -> 原文的映射(等会儿解析引用时用)
    source_map = {i + 1: text for i, (_, text, _) in enumerate(hits)}

    # Step 2: 用 Jinja2 模板渲染 Prompt
    manager = PromptManager("src/study_agent/prompt/templates")
    prompt = manager.render(
        "rag_generation",
        role="知识库问答助手",
        chunks=[text for _, text, _ in hits],
        question=question,
    )

    # Step 3: LLM 生成答案
    raw_answer = llm.chat(prompt)

    # Step 4: 解析引用
    result = parse_citations(raw_answer, source_map)

    return RAGResult(
        question=question,
        answer=result.answer,
        sources=[{"idx": idx, "text": text, "similarity": sim} for idx, text, sim in hits],
        citations=result.citations,
    )


def demo_pipeline(llm: LLMClient, retriever: SimpleRetriever) -> None:
    """演示端到端 RAG 管道。"""
    print("\n" + "=" * 65)
    print("  Part 6: 端到端 RAG 管道")
    print("=" * 65)

    test_questions = [
        "试用期多久？试用期工资怎么算？",
        "每个月几号发工资？",
        "公司 Wi-Fi 密码是什么？",
    ]

    for q in test_questions:
        print(f"\n{'-'*55}")
        print(f'  [Q] 用户: "{q}"')

        try:
            result = rag_pipeline(q, retriever, llm, top_k=3)

            print(f"\n  [检索] 检索到 {len(result.sources)} 条相关资料：")
            for s in result.sources:
                print(f"     [doc-{s['idx']}] (sim={s['similarity']:.3f}) {s['text'][:60]}...")

            print("\n  [LLM] 回答：")
            for line in result.answer.strip().split("\n"):
                print(f"    {line}")

            if result.citations:
                print(f"\n  [引用] 引用详情({len(result.citations)} 条)：")
                for c in result.citations:
                    snippet = c["text"][:70].replace("\n", " ")
                    print(f"     [{c['number']}] -> {snippet}...")
            else:
                print("  [引用] 未找到相关引用")
        except Exception as e:
            print(f"  [!] 管道执行失败: {e}")


# ==================================================================
# 主入口
# ==================================================================


def main() -> None:
    print("+======================================================+")
    print("|  Week 3 Day 5 — 生成与引用(Generation & Citation)  |")
    print("+======================================================+")

    # 初始化
    retriever = SimpleRetriever(CORPUS)

    # 尝试连接 LLM(DeepSeek / OpenAI / Anthropic 任选一个)
    llm: LLMClient | None = None
    use_llm: bool = False
    try:
        llm = LLMClient.from_env()
        print(f"\n[OK] 已连接 {llm.provider} / {llm.model}")
        use_llm = True
    except Exception as e:
        print(f"\n[!] 无法连接 LLM: {e}")
        print("将使用模拟回答来展示概念——这不影响理解 RAG 生成与引用的原理。")

    # Part 1: 纯 LLM 的幻觉
    if use_llm:
        assert llm is not None
        demo_no_rag(llm)

    # Part 2: RAG 但不引用
    if use_llm:
        assert llm is not None
        demo_rag_no_citation(llm, retriever)

    # Part 3: RAG + 引用(核心)
    if use_llm:
        assert llm is not None
        demo_rag_with_citation(llm, retriever)
    else:
        # 离线模式也展示 Prompt 模板和引用格式
        print("\n" + "=" * 65)
        print("  Part 3: RAG + 引用 —— Prompt 模板预览(离线模式)")
        print("=" * 65)
        question = "入职满 3 年的员工有多少天年假？请病假需要什么证明？"
        hits = retriever.search(question, top_k=4)
        manager = PromptManager("src/study_agent/prompt/templates")
        prompt = manager.render(
            "rag_generation",
            role="公司制度咨询专家",
            chunks=[text for _, text, _ in hits],
            question=question,
        )
        print("\n  渲染后的 Prompt 模板：")
        print(f"  {'-'*55}")
        for line in prompt.strip().split("\n"):
            print(f"  | {line}")
        print(f"  {'-'*55}")
        print("\n  [提示] 模拟回答(展示了 [1] [2] 引用格式)：")
        print("    入职满 3 年的员工享有 10 天带薪年假 [1]。")
        print("    年假需提前一周向直属上级申请 [1]。")
        print("    请病假需提供二级以上医院开具的病假证明 [2]。")
        print("    紧急情况可在 24 小时内补交证明 [2]。")

    # Part 4: 引用解析(不需要 LLM，纯本地逻辑)
    demo_citation_parsing()

    # Part 5: 边界情况
    if use_llm:
        assert llm is not None
        demo_edge_cases(llm, retriever)

    # Part 6: 端到端管道
    if use_llm:
        assert llm is not None
        demo_pipeline(llm, retriever)
    else:
        print("\n" + "=" * 65)
        print("  Part 6: 端到端管道(离线模式 — 跳过 LLM 调用)")
        print("=" * 65)
        print("\n  rag_pipeline() 函数已就绪，代码结构：")
        print("    1. retriever.search(question) -> 检索 Top-K")
        print("    2. sim_threshold 过滤低分文档")
        print("    3. Jinja2 渲染 rag_generation.j2 模板")
        print("    4. llm.chat(prompt) -> 生成答案")
        print("    5. parse_citations() -> 提取 [1] [2] 映射原文")
        print("  \n  配置好 API Key 后运行即可看到完整效果。")

    # 总结
    print(f"\n{'='*65}")
    print("  Day 5 核心收获：")
    print(f"{'='*65}")
    print(
        """
    1. 纯 LLM 会「幻觉」——它不知道你公司的内部规定，但会编造
    2. RAG 解决幻觉：先检索真实资料 -> 塞进 Prompt -> LLM 基于资料回答
    3. 引用让答案可验证：[1] [2] 标注 -> 用户能追溯到原始文档
    4. Prompt 模板是「软约束」：不是代码逻辑，但 LLM 通常遵守
    5. 端到端管道 = 检索 + 增强 + 生成 + 引用 -> 一个函数搞定
    """
    )


if __name__ == "__main__":
    main()
