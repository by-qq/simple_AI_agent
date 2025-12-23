# 下面的几个模型基本上都是数据库里查出来后，用户要用到的属性组成的。
# 至于哪些内容会用到，要看后续代码，所以可能随时增加或者删除。

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class KBDocListItem(BaseModel):
    doc_id: str
    original_filename: str
    stored_path: str
    visibility: str
    uploader_user_id: Optional[int] = None
    uploader_username: Optional[str] = None
    chunk_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class KBDocDetail(KBDocListItem):
    chroma_chunk_count: int = 0


class KBDocVisibilityUpdateReq(BaseModel):
    visibility: str = Field(..., min_length=1)


class KBDocReembedResp(BaseModel):
    doc_id: str
    deleted_chunks: int
    new_chunks: int
    visibility: str



