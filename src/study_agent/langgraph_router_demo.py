"""LangGraph 路由 Agent 演示脚本。

演示内容：
  1. 技术问题路由 —— 走搜索+回答分支
  2. 闲聊路由 —— 走对话分支
  3. 可视化图结构 —— 打印节点和边
  4. 对比：手工 if/else vs LangGraph

运行方式：
  poetry run python src/study_agent/langgraph_router_demo.py
"""

from __future__ import annotations

from study_agent.agent.langgraph_router import (
    LangGraphRouterAgent,
    build_router_graph,
)

SEP = "=" * 60


def demo1_tech_question():
    """演示 1：技术问题 -> router_node -> search_node -> answer_node"""
    print(SEP)
    print("演示 1: 技术问题路由")
    print(SEP)
    agent = LangGraphRouterAgent()

    questions = [
        "FastAPI 怎么用？",
        "LangGraph 是什么？",
        "Python 代码性能怎么优化？",
    ]

    for q in questions:
        result = agent.run(q)
        print(f"\n--- 用户: {q}")
        print(f"意图: {result['intent']}")
        print(f"搜索结果长度: {len(result.get('search_result', ''))} 字符")
        answer = result.get("final_answer", "")
        # 截取前 150 字符展示
        preview = answer[:150] + ("..." if len(answer) > 150 else "")
        print(f"回答: {preview}")


def demo2_chit_chat():
    """演示 2：闲聊 -> router_node -> chat_node"""
    print(f"\n{SEP}")
    print("演示 2: 闲聊路由")
    print(SEP)
    agent = LangGraphRouterAgent()

    messages = [
        "你好",
        "谢谢你的帮助",
        "今天天气不错",
    ]

    for msg in messages:
        result = agent.run(msg)
        print(f"\n--- 用户: {msg}")
        print(f"意图: {result['intent']}")
        print(f"回答: {result['final_answer']}")


def demo3_inside_the_graph():
    """演示 3：看看图里面有什么（可视化）"""
    print(f"\n{SEP}")
    print("演示 3: LangGraph 图结构可视化")
    print(SEP)

    graph = build_router_graph()
    app = graph.compile()

    # get_graph() 返回图的描述
    graph_repr = app.get_graph()
    print("\n图结构 (Mermaid 格式):")
    print(graph_repr.draw_mermaid())

    print("\n--- 图中包含的节点 ---")
    for node in graph_repr.nodes:
        print(f"  - {node}")

    print("\n--- 图中包含的边 ---")
    for edge in graph_repr.edges:
        src = edge.source if hasattr(edge, "source") else edge[0]
        dst = edge.target if hasattr(edge, "target") else edge[1]
        cond = ""
        if hasattr(edge, "conditional") and edge.conditional:
            cond = " [条件边]"
        print(f"  {src} -> {dst}{cond}")


def demo4_compare_ifelse():
    """演示 4：对比 —— 为什么不用手工 if/else？"""
    print(f"\n{SEP}")
    print("演示 4: LangGraph vs 手工 if/else")
    print(SEP)

    print(
        """
用 if/else 也能实现路由，代码大概长这样：

    if is_tech_question(user_input):
        result = search(user_input)
        answer = answer_with_llm(result)
    else:
        answer = chat_reply(user_input)

这在小项目里完全够用。那么为什么需要 LangGraph？

1. 可扩展性: if/else 只有两个分支时还行，10 个分支时就像意大利面
   LangGraph 的图结构天然支持复杂分支，容易理解和修改

2. 可观测性: if/else 执行过程是黑盒，LangGraph 的每个节点输入/输出都可见
   可以通过 get_graph() 导出图结构，直接放进文档

3. 状态管理: if/else 靠局部变量传来传去，LangGraph 的 State 是明确的数据契约
   多人协作时，State 类型就是团队的"接口定义"

4. 中断与恢复: if/else 只能从头跑到尾，LangGraph 支持 checkpoint（本周稍后会学）
   可以在任意节点暂停、人工审核、再继续

一句话总结：
  if/else = 手写路线图，小路径 OK，高速公路扛不住
  LangGraph = 导航系统，复杂路网也能清晰导航
"""
    )


if __name__ == "__main__":
    demo1_tech_question()
    demo2_chit_chat()
    demo3_inside_the_graph()
    demo4_compare_ifelse()
