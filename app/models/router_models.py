# 这个模型能够存储路由的状态
from typing import TypedDict, Any


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