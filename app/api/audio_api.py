from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from langchain_core.documents import Document

from app.api.auth_api import UserInDB, get_current_user
from app.api.kb_api import normalize_visibility
from app.db import mysql_audio
from app.ingestion.asr import ASR
from app.ingestion.audio_loader import transcode_to_wav_16k_mono, ffprobe_duration_ms
from app.ingestion.retrieve_audio import audio_similarity_search_for_user
from app.ingestion.segments import merge_by_max_duration
from app.security.rbac.perms import check_permission
from app.models.audio_models import AudioIngestResp, AudioDocDetail, AudioSearchResp, AudioSearchHit

router = APIRouter(prefix="/audio", tags=["audio"])

AUDIO_DIR = Path("data/audio")          # 你也可以接到 settings
AUDIO_WAV_DIR = Path("data/audio_wav")


@router.post("/ingest", response_model=AudioIngestResp)
async def ingest_audio(
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    audio_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),     # 可传 "zh"/"en"，不传就自动
    current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    visibility = normalize_visibility(visibility or "public")
    audio_id = (audio_id or f"aud-{uuid.uuid4().hex[:12]}").strip()

    # 1) 保存原始文件
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".bin"
    raw_path = AUDIO_DIR / f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    raw_path.write_bytes(raw_bytes)

    # 2) 转码到 16k mono wav
    AUDIO_WAV_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = AUDIO_WAV_DIR / f"{audio_id}.wav"
    transcode_to_wav_16k_mono(raw_path, wav_path)

    # 3) 时长
    duration_ms = ffprobe_duration_ms(wav_path)

    # 4) ASR
    asr = ASR(model_name="base", device="cpu", compute_type="int8")
    asr_segs, detected_lang = asr.transcribe(str(wav_path), language=language)
    lang = language or detected_lang

    # 5) 切成更适合检索的 chunks
    chunks = merge_by_max_duration(asr_segs, max_ms=25_000, min_ms=6_000)

    # 6) 写 Chroma（文本 embedding）
    from app.deps import get_vs
    vs = get_vs()

    docs: list[Document] = []
    segment_rows: list[dict] = []
    for idx, c in enumerate(chunks):
        seg_id = f"{audio_id}:{idx}"
        text = c.text.strip()
        if not text:
            continue

        meta = {
            "doc_type": "audio",
            "audio_id": audio_id,
            "segment_id": seg_id,
            "segment_idx": idx,
            "start_ms": c.start_ms,
            "end_ms": c.end_ms,
            "visibility": visibility,
            "original_filename": file.filename,
            "stored_path": str(raw_path),
            "wav_path": str(wav_path),
            "language": lang,
        }
        docs.append(Document(page_content=text, metadata=meta))
        segment_rows.append(
            {"segment_idx": idx, "start_ms": c.start_ms, "end_ms": c.end_ms, "text": text}
        )

    if not docs:
        raise HTTPException(status_code=400, detail="No transcript produced")

    vs.add_documents(docs)
    try:
        vs.persist()
    except Exception:
        pass

    # 7) 落库
    mysql_audio.upsert_audio_document(
        audio_id=audio_id,
        original_filename=file.filename,
        stored_path=str(raw_path),
        duration_ms=duration_ms,
        language=lang,
        visibility=visibility,
        status="indexed",
        uploader_user_id=int(current_user.id),
        uploader_username=current_user.username,
        segment_count=len(segment_rows),
    )
    mysql_audio.replace_audio_segments(audio_id, segment_rows)

    return AudioIngestResp(
        audio_id=audio_id,
        stored_as=str(raw_path),
        duration_ms=duration_ms,
        language=lang,
        visibility=visibility,
        segments=len(segment_rows),
    )

@router.get("/search", response_model=AudioSearchResp)
def search_audio(
    q: str = Query(..., min_length=1),
    k: int = Query(default=6, ge=1, le=20),
    current_user: UserInDB = Depends(get_current_user),
):
    # 只要能看 public KB，就应该能用检索（你也可以要求 kb.view_public）
    docs, allowed = audio_similarity_search_for_user(q, current_user, k=k)

    hits: list[AudioSearchHit] = []
    for d in docs:
        m = d.metadata or {}
        hits.append(
            AudioSearchHit(
                audio_id=str(m.get("audio_id", "")),
                segment_id=str(m.get("segment_id", "")),
                start_ms=int(m.get("start_ms", 0) or 0),
                end_ms=int(m.get("end_ms", 0) or 0),
                text=d.page_content,
                score=None,
            )
        )

    return AudioSearchResp(q=q, k=k, allowed_visibilities=allowed, hits=hits)

@router.get("/{audio_id}", response_model=AudioDocDetail)
def get_audio(audio_id: str, current_user: UserInDB = Depends(get_current_user)):
    check_permission(current_user, "kb.manage_docs")
    row = mysql_audio.get_audio_document(audio_id)
    if not row:
        raise HTTPException(status_code=404, detail="audio not found")
    return row


