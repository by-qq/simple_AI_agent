# 定义模型
# 新开业务线：就是新开发功能
# V1版本是RAG规则的检索，大部分用的是EMBEDDING,没怎么用到LLM

# 作业务都是模型先定好
from enum import Enum
from typing import Optional, TypedDict, List

from pydantic import BaseModel


class LeaveType(str, Enum):
    annual = "annual"
    sick = "sick"
    personal = "personal"
    other = "other"
# 请假单，模型就是一组数据，在项目之间传来传去，一般和数据库是对应的
# 一个模型类对应一个关系型数据库的表,一个对象对应表中的一行（对象关系映射ORM）
class LeaveRequest(BaseModel):
    requester : str
    leave_type : LeaveType = LeaveType.annual
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_days: Optional[float] = None
    reason: Optional[str] = None
# 请求状态
class LeaveState(TypedDict, total=False):
    text:str
    requester: str
    user_role: str

    req: dict
    missing_fields:List[str]
    violations: List[str]

    answer: str
    confirmed: bool # 批准状态
    leave_id: Optional[str]

    user_id: Optional[int]
    roles: List[str]    # 可能和roles重复，但为了避免更改过多选择保留user_role
    permissions: List[str]
    is_super_admin: bool



