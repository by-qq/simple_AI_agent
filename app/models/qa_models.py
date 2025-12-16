from typing import TypedDict, Any, List


class QAState(TypedDict, total=False):
    question: str
    text: str
    user_role: str
    docs: List[Any]
    answer: str
    messages: List[Any]
