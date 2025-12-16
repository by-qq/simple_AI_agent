from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from app.db.worker_mysql import create_ticket
from app.prompts.RAG_prompts import QA_SYSTEM, QA_USER
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

# def retrieve(state: QAState) -> dict:
#     vs = get_vs()
#     role = state.get("user_role", "public")
#     query = state.get("question") or state.get("text") or ""
#
#     retriever = vs.as_retriever(
#         search_kwargs={
#             "k": 8,
#             "filter": {"visibility": {"$in": ["public", role]}},
#         }
#     )
#     docs = retriever.invoke(query)
#
#     if not docs:
#         retriever2 = vs.as_retriever(search_kwargs={"k": 8})
#         docs = retriever2.invoke(query)
#         return {"docs": docs, "question": query, "debug": "fallback_unfiltered"}
#
#     return {"docs": docs, "question": query, "debug": "filtered"}
def retrieve(state: QAState) -> dict:
    vs = get_vs()
    role = state.get("user_role", "public")
    query = state.get("question") or state.get("text") or ""

    # 设置相似度阈值 - 建议根据实际测试调整
    SIMILARITY_THRESHOLD = 0.1  # Chroma默认使用余弦相似度，范围[-1, 1]，0.5是个合理的起点

    try:
        # 方法1: 带过滤和分数的检索（首选）
        results = vs.similarity_search_with_score(
            query=query,
            k=8,  # 最多获取8个
            filter={"visibility": {"$in": ["public", role]}}
        )

        # 过滤低于阈值的文档
        filtered_docs = []
        scores = []
        for doc, score in results:
            if score >= SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                scores.append(score)

        debug_info = f"filtered_threshold={SIMILARITY_THRESHOLD}, found={len(filtered_docs)}"

        # 如果没有找到符合阈值的文档，尝试不带过滤的检索
        if not filtered_docs:
            return {}

        # 将分数附加到文档的metadata中，方便调试
        for i, doc in enumerate(filtered_docs):
            doc.metadata["similarity_score"] = float(scores[i])

        return {
            "docs": filtered_docs,
            "question": query,
            "debug": debug_info,
            "scores": scores
        }

    except Exception as e:
        # 如果出错，回退到原来的方法
        print(f"Error in similarity_search_with_score: {e}")

        retriever = vs.as_retriever(
            search_kwargs={
                "k": 8,
                "filter": {"visibility": {"$in": ["public", role]}},
            }
        )
        docs = retriever.invoke(query)

        return {"docs": docs, "question": query, "debug": "filtered_error"}

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
    text = state.get("text") or state.get("question")
    role = state.get("user_role")
    if role in ["admin", "hr"]:
        # 为管理员和HR添加文件上传链接
        base_response = "我没有在当前可见知识库中找到足够证据回答。请提供更具体的关键词/文档来源。"
        upload_link = "http://127.0.0.1:8002/ingest"
        upload_instruction = f"是否要添加相关文件？您可以点击链接上传文件：{upload_link}"

        return {
            "text": text,
            "answer": f"{base_response} {upload_instruction}"
        }

        # 普通用户返回基本提示
    return {
        "answer": "我没有在当前可见知识库中找到足够证据回答。请提供更具体的关键词/文档来源。"
    }

# 普通用户搜索不到增加自动提交工单的节点
def auto_confirm_node(state:QAState) -> None:
    text = state.get("text") or state.get("question")
    role = state.get("user_role")
    if role in ["public"]:
        # 直接调用创建工单的API
        create_ticket(text)

def build_qa_graph():
    g = StateGraph(QAState)

    # 注意：节点注册用 runnable（返回 dict）
    g.add_node("decide_retrieve", decide_retrieve_node)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate_answer)
    g.add_node("refuse", refuse_or_clarify)

    g.add_node("auto_confirm",auto_confirm_node)

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
    g.add_edge("refuse", "auto_confirm")
    g.add_edge("auto_confirm",END)

    return g.compile()



