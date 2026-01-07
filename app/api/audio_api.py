from __future__ import annotations

import time, uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile,)
from fastapi.responses import FileResponse
from langchain_core.documents import Document

from app.api.auth_api import UserInDB, get_current_user
from app.config import settings
from app.db import mysql_audio, mysql_audio_job
from app.deps import get_audio_vs
from app.tools.asr import ASR
from app.tools.audio_loader import ffprobe_duration_ms
from app.tools.audio_clip import clip_audio_to_mp3
from app.tools.pipeline import transcode_to_wav_16k_mono
from app.tools.segments import merge_by_max_duration
from app.models.audio_models import AudioDocDetail, AudioSearchResp, AudioSearchHit, AudioIngestAsyncResp, AudioJobResp
from app.security.rbac.perms import check_permission, allowed_kb_visibilities
from app.tasks.audio_tasks import audio_ingest_task

router = APIRouter(prefix="/audio", tags=["audio"])

AUDIO_DIR = Path(getattr(settings, "audio_dir", "data/audio"))
AUDIO_WAV_DIR = Path("data/audio_wav")
CLIP_DIR = Path(getattr(settings, "audio_clip_dir", "data/audio_clips"))

def _require_manage_docs(user: UserInDB) -> None:
    check_permission(user, "kb.manage_docs")


def _normalize_visibility(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("public", "internal"):
        return v
    return "public"


def _compute_allowed_visibilities(user: UserInDB) -> List[str]:
    perms = getattr(user, "permissions", None)
    allowed = allowed_kb_visibilities(perms)
    if "public" not in allowed:
        allowed = ["public"] + [x for x in allowed if x != "public"]
    return allowed


def _ensure_can_access_visibility(user: UserInDB, doc_visibility: str) -> List[str]:
    allowed = _compute_allowed_visibilities(user)
    vis = (doc_visibility or "").strip().lower()
    if vis not in set(allowed):
        raise HTTPException(status_code=403, detail="no permission to access this audio")
    return allowed


def _absolute_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("/ingest", response_model=AudioIngestAsyncResp)
async def ingest_audio(
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    audio_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    overwrite: bool = Form(False),
    delete_old_file: bool = Form(False),
    current_user: UserInDB = Depends(get_current_user),):
    _require_manage_docs(current_user)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    visibility = _normalize_visibility(visibility or "public")
    audio_id = (audio_id or f"aud-{uuid.uuid4().hex[:12]}").strip()
    job_id = f"job-{uuid.uuid4().hex[:12]}"

    if mysql_audio.is_audio_running(audio_id):
        raise HTTPException(status_code=409, detail="audio is running, try later")

    existed = mysql_audio.get_audio_document(audio_id)
    if existed and not overwrite:
        raise HTTPException(status_code=409, detail="audio_id already exists; set overwrite=true")

    old_stored_path = existed["stored_path"] if existed else None

    # 1) 保存原始文件
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".bin"
    raw_path = AUDIO_DIR / f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    raw_path.write_bytes(raw_bytes)

    mysql_audio.upsert_audio_document(
        audio_id=audio_id,
        original_filename=file.filename,
        stored_path=str(raw_path),
        duration_ms=0,
        language=language,
        visibility=visibility,
        status="queued",
        uploader_user_id=int(getattr(current_user, "id", 0) or 0) or None,
        uploader_username=getattr(current_user, "username", None),
        segment_count=0,
    )

    mysql_audio_job.create_job(
        job_id,
        audio_id,
        overwrite=bool(overwrite),
        delete_old_file=bool(delete_old_file),
        old_stored_path=old_stored_path if overwrite else None,
    )

    async_result = audio_ingest_task.apply_async(
        args=[job_id, audio_id],
        queue=getattr(settings, "celery_audio_queue", "audio"),
    )
    mysql_audio_job.bind_task(job_id, async_result.id)

    return AudioIngestAsyncResp(
        job_id=job_id,
        audio_id=audio_id,
        stored_as=str(raw_path),
        visibility=visibility,
        celery_task_id=async_result.id,
        status_url=f"/audio/jobs/{job_id}",
    )


@router.get("/jobs/{job_id}", response_model=AudioJobResp)
def get_audio_job(
    job_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    row = mysql_audio_job.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return row


@router.post("/jobs/{job_id}/cancel")
def cancel_audio_job(
    job_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    ok = mysql_audio_job.request_cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "cancel_requested": True}


@router.get("/docs/{audio_id}", response_model=AudioDocDetail)
def get_audio_doc(
    audio_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    row = mysql_audio.get_audio_document(audio_id)
    if not row:
        raise HTTPException(status_code=404, detail="audio not found")
    return row


@router.get("/query", response_model=AudioSearchResp)
def query_audio(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(6, ge=1, le=20),
    current_user: UserInDB = Depends(get_current_user),) -> AudioSearchResp:
    allowed_vis = _compute_allowed_visibilities(current_user)

    vs = get_audio_vs()
    where = {"visibility": {"$in": allowed_vis}}

    try:
        docs_scores = vs.similarity_search_with_score(q, k=k, filter=where)
    except TypeError:
        docs_scores = vs.similarity_search_with_score(q, k=k, where=where)

    hits: list[AudioSearchHit] = []
    base = _absolute_base(request)

    for doc, score in docs_scores:
        md = doc.metadata or {}
        audio_id = str(md.get("audio_id") or "")
        segment_id = str(md.get("segment_id") or "")
        start_ms = int(md.get("start_ms") or 0)
        end_ms = int(md.get("end_ms") or 0)
        text = (doc.page_content or "").strip()

        if not audio_id or not segment_id:
            continue

        clip_url = f"{base}/audio/docs/{audio_id}/clip?start_ms={start_ms}&end_ms={end_ms}"

        hits.append(
            AudioSearchHit(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                score=float(score) if score is not None else None,
                clip_url=clip_url,
            )
        )

    return AudioSearchResp(q=q, k=k, allowed_visibilities=allowed_vis, hits=hits)


@router.get("/docs/{audio_id}/clip")
def get_audio_clip(
    audio_id: str,
    background_tasks: BackgroundTasks,
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    segment_id: Optional[str] = Query(default=None),  # e.g. aud-xxx:3
    current_user: UserInDB = Depends(get_current_user),):
    doc = mysql_audio.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    _ensure_can_access_visibility(current_user, doc.get("visibility") or "")

    if segment_id:
        if ":" not in segment_id:
            raise HTTPException(status_code=400, detail="invalid segment_id format")
        seg_audio_id, seg_idx_str = segment_id.split(":", 1)
        if seg_audio_id != audio_id:
            raise HTTPException(status_code=400, detail="segment_id does not match audio_id")
        try:
            seg_idx = int(seg_idx_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid segment_idx in segment_id")

        seg = mysql_audio.get_audio_segment(audio_id, seg_idx)
        if not seg:
            raise HTTPException(status_code=404, detail="segment not found")

        start_ms = int(seg["start_ms"])
        end_ms = int(seg["end_ms"])
    else:
        if start_ms is None or end_ms is None:
            raise HTTPException(
                status_code=400,
                detail="start_ms and end_ms are required when segment_id is not provided",
            )

    if end_ms <= start_ms:
        raise HTTPException(status_code=400, detail="end_ms must be greater than start_ms")

    max_clip_ms = 5 * 60 * 1000
    if (end_ms - start_ms) > max_clip_ms:
        raise HTTPException(status_code=400, detail="clip too long")

    src_path = Path(doc["stored_path"])
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="stored audio file missing")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clip_name = f"{audio_id}_{start_ms}_{end_ms}_{uuid.uuid4().hex[:8]}.mp3"
    clip_path = CLIP_DIR / clip_name

    try:
        clip_audio_to_mp3(
            src_path=src_path,
            dst_path=clip_path,
            start_ms=int(start_ms),
            end_ms=int(end_ms),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="failed to generate clip")

    background_tasks.add_task(lambda p=str(clip_path): Path(p).unlink(missing_ok=True))

    return FileResponse(
        path=str(clip_path),
        media_type="audio/mpeg",
        filename=clip_name,
    )
