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
from app.workflows.leave.leave_graph import build_leave_graph

# 这个模型能够存储路由的状态
class RouterState(TypedDict, total=False):  # 顶层状态结构，total=False表示下面所有字段都是可选的
    question: str  # 给QA的问题
    text: str  # 用户原始文本
    user_role: str  # 用户角色
    mode: str  # 模式标记，比如qa，rag，kb等等

    requester:str
    active_route:str
    req: dict
    missing_field: list[str]
    violations: list[str]

    answer: str  # 答案
    docs: list[Any]  # QA检索到的文档列表
    leave_id:str

def decide_route(state: RouterState) -> str:
    # 决定走哪个路由
    mode = (state.get("mode") or "").lower().strip()

    active = (state.get("active_route") or "").lower().strip()
    if active == "leave" and mode not in {"qa", "rag", "kb"}:
        return "leave"

    # 显示mode优先
    if mode in {"qa", "rag", "kb"}:
        return "qa"
    if mode in {"leave" , "hr"}:
        return "leave"
    # 关键词路由，根据图的状态或者根据用户的输入
    text = (state.get("text") or "").lower().strip()
    if any(k in text for k in ["请假","年假","病假","事假","休假","调休","假期"]):
        return "leave"
    return "qa" # 这个代码反正最后都是qa


def route_node(state: RouterState) -> dict:
    # 路由前的站位节点，将qa变成字典
    return {"active_route":decide_route(state)}


def build_router_graph():
    qa_graph = build_qa_graph()
    leave_graph = build_leave_graph()

    g = StateGraph(RouterState)

    g.add_node("route", route_node)
    g.add_node("qa", qa_graph)
    g.add_node("leave", leave_graph)

    g.add_edge(START, "route")

    g.add_conditional_edges(
        "route",
        decide_route,
        {"qa": "qa", "leave": "leave"},
    )

    g.add_edge("qa", END)
    g.add_edge("leave", END)

    return g.compile()

router_graph = build_router_graph()
