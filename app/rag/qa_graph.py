from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.prompts import QA_SYSTEM, QA_USER
from app.deps import get_llm, get_vs


class QAState(TypedDict, total=False):
    question: str
    text: str
    user_role: str
    docs: List[Any]
    answer: str
    messages: List[Any]


def decide_retrieve(state: QAState) -> str:
    return "retrieve"

def decide_retrieve_node(state: QAState) -> dict:
    """
    节点 runnable：必须返回 dict
    这里只是一个no-op节点，真正路由在decide_retrieve()里完成
    """
    return {}

def retrieve(state: QAState) -> dict:
    vs = get_vs()
    role = state.get("user_role", "public")
    query = state.get("question") or state.get("text") or ""

    retriever = vs.as_retriever(
        search_kwargs={
            "k": 8,
            "filter": {"visibility": {"$in": ["public", role]}},
        }
    )
    docs = retriever.invoke(query)

    if not docs:
        retriever2 = vs.as_retriever(search_kwargs={"k": 8})
        docs = retriever2.invoke(query)
        return {"docs": docs, "question": query, "debug": "fallback_unfiltered"}

    return {"docs": docs, "question": query, "debug": "filtered"}


def grade_evidence(state: QAState) -> str:
    """检索后判断是否有证据。"""
    return "good" if state.get("docs") else "bad"


def generate_answer(state: QAState) -> dict:
    """带引用生成答案。"""
    llm = get_llm()
    docs = state.get("docs", [])

    context = "\n\n".join(
        f"[{i+1}] {d.page_content}\n(source={d.metadata.get('source')}, page={d.metadata.get('page')})"
        for i, d in enumerate(docs[:6])
    )

    prompt = QA_USER.format(question=state["question"], context=context)
    messages = [AIMessage(content=QA_SYSTEM), HumanMessage(content=prompt)]
    ans = llm.invoke(messages).content
    return {"answer": ans}


def refuse_or_clarify(state: QAState) -> dict:
    """无证据兜底。"""
    return {
        "answer": "我没有在当前可见知识库中找到足够证据回答。请提供更具体的关键词/文档来源。"
    }


def build_qa_graph():
    g = StateGraph(QAState)

    # 注意：节点注册用 runnable（返回 dict）
    g.add_node("decide_retrieve", decide_retrieve_node)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate_answer)
    g.add_node("refuse", refuse_or_clarify)

    g.add_edge(START, "decide_retrieve")

    g.add_conditional_edges(
        "decide_retrieve",
        decide_retrieve,
        {
            "retrieve": "retrieve",
            "direct": "generate",
        },
    )

    g.add_conditional_edges(
        "retrieve",
        grade_evidence,
        {
            "good": "generate",
            "bad": "refuse",
        },
    )

    g.add_edge("generate", END)
    g.add_edge("refuse", END)

    return g.compile()



