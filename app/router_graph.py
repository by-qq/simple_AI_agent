# 12. app/router_graph.py（顶层路由）
# 我们这里搭了一个顶层Router图，
# 这个非常简单，现在只有一种模式：不管怎样都把请求丢给qa_graph去做RAG。
#
# 后面我们可以在这里挂更多模式，比如：
# • action -> 走前面那个工单创建workflow
# • chat -> 走一个纯闲聊LLM
# • tool -> 调用别的工具
#
# 现在是一个最小可用版本，只实现了Q&A路由
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from app.rag.qa_graph import build_qa_graph

class RouterState(TypedDict, total=False):  # 顶层状态结构，total=False表示下面所有字段都是可选的
    question: str  # 给QA的问题
    text: str  # 用户原始文本
    user_role: str  # 用户角色
    mode: str  # 模式标记，比如qa，rag，kb等等
    answer: str  # 答案
    docs: list[Any]  # QA检索到的文档列表

def decide_route(state: RouterState) -> str:
    # 决定走哪个路由
    mode = (state.get("mode") or "").lower().strip()
    if mode in {"qa", "rag", "kb"}:
        return "qa"
    return "qa" # 这个代码反正最后都是qa


def route_node(state: RouterState) -> dict:
    # 路由前的站位节点，返回{}意味着不修改状态
    return {}


def build_router_graph():
    qa_graph = build_qa_graph()

    g = StateGraph(RouterState)

    g.add_node("route", route_node)
    g.add_node("qa", qa_graph)

    g.add_edge(START, "route")

    g.add_conditional_edges(
        "route",
        decide_route,
        {"qa": "qa"},
    )

    g.add_edge("qa", END)

    return g.compile()

router_graph = build_router_graph()
