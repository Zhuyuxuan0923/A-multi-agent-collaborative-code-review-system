"""LangGraph 路由 Agent —— 带条件分支的 Agent 工作流。

用 LangGraph 搭建一个智能路由器：
  - 用户问技术问题 -> 走搜索+回答分支
  - 用户闲聊 -> 走对话分支

核心概念（按学习顺序）：
  1. State  —— 在图中流动的数据（TypedDict 定义）
  2. Node   —— 处理 State 的函数（输入 State，返回 State 的更新部分）
  3. Edge   —— 节点之间的连线（普通边 vs. 条件边）
  4. Graph  —— 节点 + 边的集合（编译后可反复调用）
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from study_agent.agent.research_tools import mock_search
from study_agent.llm.client import LLMClient

# ═══════════════════════════════════════════════════════════════════
# 1. State 定义 —— 在图中流动的数据
# ═══════════════════════════════════════════════════════════════════


class RouterState(TypedDict, total=False):
    """LangGraph 的 State —— 在节点之间流动的数据包。

    每个字段的含义：
      user_input:    用户输入（原始消息）
      intent:        路由分类结果（"technical" / "chat"，router_node 填充）
      search_result: 搜索结果（search_node 填充，只有技术问题会触发）
      final_answer:  最终答案（answer_node 或 chat_node 填充）

    total=False 表示所有字段都是可选的 —— 状态可以逐步填充。
    """

    user_input: str
    intent: str
    search_result: str
    final_answer: str


# ═══════════════════════════════════════════════════════════════════
# 2. Node 函数 —— 图的处理单元
# ═══════════════════════════════════════════════════════════════════


def router_node(state: RouterState) -> dict:
    """路由器节点 —— 判断用户意图。

    用简单的关键词规则做分类（不需要 LLM 调用，毫秒级）。
    真实场景可以用 LLM 做更精准的意图分类。

    分类逻辑：
      - 包含技术关键词 -> "technical"，走搜索分支
      - 其他 -> "chat"，走对话分支

    返回 dict 形式的 State 更新（LangGraph 会自动合并到当前 State）。
    """
    technical_keywords = [
        "是什么",
        "怎么用",
        "如何",
        "教程",
        "原理",
        "代码",
        "python",
        "react",
        "fastapi",
        "docker",
        "git",
        "langchain",
        "langgraph",
        "agent",
        "ai",
        "框架",
        "编程",
        "函数",
        "api",
        "数据库",
        "部署",
        "错误",
        "bug",
        "性能",
        "优化",
        "配置",
    ]

    question = state["user_input"].lower()
    score = sum(1 for kw in technical_keywords if kw in question)

    intent = "technical" if score >= 1 else "chat"
    return {"intent": intent}


def search_node(state: RouterState) -> dict:
    """搜索节点 —— 获取技术信息。

    调用模拟搜索工具，用 user_input 作为搜索关键词。
    真实场景可以接入 Tavily / SerpAPI / Brave Search 等。
    """
    result = mock_search(state["user_input"])
    return {"search_result": result}


def answer_node(state: RouterState) -> dict:
    """技术回答节点 —— 基于搜索结果生成答案。

    用 LLM 综合搜索结果，给出结构化的技术回答。
    如果搜索无结果，给出诚实的回复。
    """
    client = LLMClient.from_env()

    search_result = state.get("search_result", "")
    user_input = state.get("user_input", "")

    if not search_result or "未找到" in search_result:
        return {
            "final_answer": (
                f"关于「{user_input}」，我目前的知识库中没有找到相关信息。\n"
                "建议：1) 尝试换个关键词搜索；2) 查阅官方文档；"
                "3) 在技术社区（如 Stack Overflow）提问。"
            )
        }

    system = (
        "你是一个技术助手。根据搜索结果回答用户的问题。"
        "要求：1) 结构化回答（分点列出）；2) 准确引用搜索结果中的信息；"
        "3) 如果不确定，直接说'不确定'；4) 用中文回答。"
    )
    prompt = f"用户问题：{user_input}\n\n搜索结果：\n{search_result}\n\n请根据以上信息回答。"

    try:
        reply = client.chat(prompt, system=system)
        return {"final_answer": reply}
    except Exception:
        return {"final_answer": f"抱歉，LLM 调用失败。以下是原始搜索结果：\n\n{search_result}"}


def chat_node(state: RouterState) -> dict:
    """闲聊节点 —— 对非技术问题的友好回复。

    直接返回预设回复，不需要调用 LLM（也可以改成 LLM 回复）。
    """
    user_input = state.get("user_input", "")

    greetings = ["你好", "hi", "hello", "嗨", "早上好", "晚上好", "下午好"]
    if any(g in user_input.lower() for g in greetings):
        return {"final_answer": "你好！有什么技术问题我可以帮你吗？"}

    thanks_words = ["谢谢", "感谢", "thank", "thanks"]
    if any(t in user_input.lower() for t in thanks_words):
        return {"final_answer": "不客气！有问题随时问我。"}

    return {
        "final_answer": "我主要擅长回答技术问题。试试问我关于 Python、Agent、LangGraph 的话题吧！"
    }


# ═══════════════════════════════════════════════════════════════════
# 3. 条件边 —— 路由函数（决定下一步走哪个节点）
# ═══════════════════════════════════════════════════════════════════


def route_by_intent(state: RouterState) -> Literal["search_node", "chat_node"]:
    """根据 intent 字段决定下一步。

    这是 LangGraph 的核心概念之一 —— 条件边 (Conditional Edge)。
    它不是固定的 A->B，而是根据 State 的内容动态选择下一个节点。

    返回值必须是已注册的节点名字符串。
    """
    if state.get("intent") == "technical":
        return "search_node"
    return "chat_node"


# ═══════════════════════════════════════════════════════════════════
# 4. 构建图 —— 把节点和边组合起来
# ═══════════════════════════════════════════════════════════════════


def build_router_graph() -> StateGraph:
    r"""构建路由 Agent 的 LangGraph 图。

    图结构（ASCII）：

        START
          |
      router_node  (判断意图：technical 还是 chat？)
         / \
        /   \  <-- 条件边：route_by_intent
       /     \
  search_node   chat_node
      |            |
  answer_node      |
       \          /
        \        /
         END <---

    技术问题路径：router -> search -> answer -> END
    闲聊路径：    router -> chat -> END
    """
    # Step 1: 创建 StateGraph，绑定 State 类型
    graph = StateGraph(RouterState)

    # Step 2: 添加节点 —— 每个节点是一个处理函数
    graph.add_node("router_node", router_node)
    graph.add_node("search_node", search_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("chat_node", chat_node)

    # Step 3: 添加边
    # 3a. 普通边：START 固定走到 router_node（入口固定）
    graph.add_edge(START, "router_node")

    # 3b. 条件边：router_node 根据 route_by_intent 返回值选择下一站
    #     route_by_intent 返回 "search_node" -> 走搜索分支
    #     route_by_intent 返回 "chat_node"   -> 走对话分支
    graph.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "search_node": "search_node",
            "chat_node": "chat_node",
        },
    )

    # 3c. 普通边：搜索完成后固定走到回答
    graph.add_edge("search_node", "answer_node")

    # 3d. 终点：回答和闲聊最终都走到 END
    graph.add_edge("answer_node", END)
    graph.add_edge("chat_node", END)

    return graph


# ═══════════════════════════════════════════════════════════════════
# 5. 顶层接口 —— 一行调用
# ═══════════════════════════════════════════════════════════════════


class LangGraphRouterAgent:
    """LangGraph 路由 Agent 的顶层封装。

    用法：
      agent = LangGraphRouterAgent()
      result = agent.run("FastAPI 怎么用？")
      print(result["final_answer"])
    """

    def __init__(self):
        self._graph = build_router_graph()
        self._app = self._graph.compile()

    def run(self, user_input: str) -> RouterState:
        """运行 Agent，返回完整的 State（包含所有中间结果）。"""
        initial_state: RouterState = {
            "user_input": user_input,
            "intent": "",
            "search_result": "",
            "final_answer": "",
        }
        result: RouterState = self._app.invoke(initial_state)
        return result

    @property
    def graph(self) -> StateGraph:
        """暴露底层图对象，供可视化和调试使用。"""
        return self._graph
