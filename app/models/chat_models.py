# 这个类继承自BaseModel，转化成json
from typing import Optional

from pydantic import BaseModel


class ChatReq(BaseModel):
    text: str
    user_role: str = "public"
    requester: str = "anonymous"
    session_id: Optional[str] = None  # 通过这一行给大模型添加记忆

class ChatResp(BaseModel):
    answer: str
    session_id: Optional[str] = None
    activate_route: Optional[str] = None
