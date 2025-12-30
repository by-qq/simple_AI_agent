from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

# AudioIngestResp: 音频摄取响应
class AudioIngestResp(BaseModel):
    audio_id: str
    stored_as: str
    duration_ms: int
    language: Optional[str] = None
    visibility: str
    segments: int


# AudioDocDetail: 音频文档详情
class AudioDocDetail(BaseModel):
    audio_id: str
    original_filename: str
    stored_path: str
    duration_ms: int
    language: Optional[str] = None
    visibility: str
    status: str
    segment_count: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# AudioSearchHit: 搜索命中结果
class AudioSearchHit(BaseModel):
    audio_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    score: Optional[float] = None  # 向量库有些返回不了score就留空
    clip_url: Optional[str] = None  # ⚠️加这行就行


# AudioSearchResp: 搜索响应
class AudioSearchResp(BaseModel):
    q: str
    k: int
    allowed_visibilities: List[str]
    hits: List[AudioSearchHit]

class FileResponse(BaseModel):
    path : pathlib.Path
    media_type: str
    filename: str

