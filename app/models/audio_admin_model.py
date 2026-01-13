from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field


class AudioDocItem(BaseModel):
    audio_id: str
    original_filename: str
    stored_path: str
    duration_ms: int = 0
    language: Optional[str] = None
    visibility: str
    status: str
    uploader_user_id: Optional[int] = None
    uploader_username: Optional[str] = None
    segment_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AudioDocListResp(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[AudioDocItem]


class AudioSegmentItem(BaseModel):
    audio_id: str
    segment_idx: int
    start_ms: int
    end_ms: int
    text: str


class AudioSegmentsResp(BaseModel):
    audio_id: str
    items: List[AudioSegmentItem]


class AudioTranscriptResp(BaseModel):
    audio_id: str
    transcript: str


class AudioStatsResp(BaseModel):
    by_visibility: Dict[str, int]
    by_status: Dict[str, int]
    total: int


class UpdateVisibilityReq(BaseModel):
    visibility: str = Field(..., description="public/internal")


class UpdateVisibilityResp(BaseModel):
    audio_id: str
    visibility: str
    vectors_updated: int


class BulkDeleteReq(BaseModel):
    audio_ids: List[str]
    delete_files: bool = False


class BulkDeleteResp(BaseModel):
    deleted: List[str]
    missing: List[str]
    vectors_deleted: Dict[str, int]
    files_deleted: List[str]


class BulkReindexReq(BaseModel):
    audio_ids: List[str]


class ReindexItem(BaseModel):
    audio_id: str
    job_id: str
    celery_task_id: str
    status_url: str


class BulkReindexResp(BaseModel):
    submitted: List[ReindexItem]
    skipped_running: List[str]
    missing: List[str]


class ResetAudioCollectionResp(BaseModel):
    ok: bool
    collection: str