from __future__ import annotations

import json
import os
import time, uuid
from pathlib import Path
from typing import List, Optional, Iterable, Any

import httpx
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile,)
from fastapi.responses import FileResponse,StreamingResponse

from app.api.auth_api import UserInDB, get_current_user
from app.config import settings
from app.db import mysql_audio, mysql_audio_job
from app.db.chroma_admin import delete_by_audio_id
from app.deps import get_audio_vs
from app.tools.audio_clip import clip_audio_to_mp3
from app.models.audio_models import AudioDocDetail, AudioSearchResp, AudioSearchHit, AudioIngestAsyncResp, AudioJobResp, \
    AudioCitation, AudioAskResp, AudioAskReq
from app.security.rbac.perms import check_permission, allowed_kb_visibilities
from app.tasks.audio_tasks import audio_ingest_task, audio_reindex_task

router = APIRouter(prefix="/audio", tags=["audio"])

AUDIO_DIR = Path(getattr(settings, "audio_dir", "data/audio"))
CLIP_DIR = Path(getattr(settings, "audio_clip_dir", "data/audio_clips"))
WAV_DIR = Path(getattr(settings, "audio_wav_dir", "data/audio_wav"))


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

# 取整个项目的超链接
def _absolute_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")

def _clip_url(base: str, audio_id: str, start_ms: int, end_ms: int) -> str:
    return f"{base}/audio/docs/{audio_id}/clip?start_ms={start_ms}&end_ms={end_ms}"

def _openai_chat_complete(*, model: str, api_key: str, messages: list[dict[str, str]], timeout_s: float = 60.0) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {r.status_code} {r.text[:300]}")
        data = r.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise HTTPException(status_code=500, detail="OpenAI response parse error")


def _build_rag_messages(question: str, citations: list[AudioCitation], system_prompt: Optional[str]) -> list[dict[str, str]]:
    sys = (system_prompt or "").strip() or (
        "你是企业知识库助手，回答必须基于给定的【音频片段】内容。"
        "如果片段不足以回答，就明确说“不确定/片段中没有”。"
        "回答要简洁，并在结尾给出引用列表（用 [1][2]... 标注）。"
    )

    ctx_lines: list[str] = []
    for i, c in enumerate(citations, start=1):
        ctx_lines.append(
            f"[{i}] audio_id={c.audio_id} segment_id={c.segment_id} "
            f"start_ms={c.start_ms} end_ms={c.end_ms}\n"
            f"片段文本：{c.text}"
        )
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "（无片段）"

    user = (
        f"问题：{question}\n\n"
        f"【音频片段】\n{ctx}\n\n"
        "要求：\n"
        "1) 只用片段信息回答。\n"
        "2) 如果引用了某个片段，请用 [序号] 标注。\n"
        "3) 不要编造片段里没有的信息。"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

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
            continue    # mysql和chroma中的数据不匹配

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
    background_tasks: BackgroundTasks,      # FASTapi提供的小功能，和多线程不一样（可能会占据当前线程，也可能是后台任务），可以被当作轻量级的多线程，
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

# ⚠️位了兼容老借口，调用一下之前的query就可以了，项目中经常这样做委托
@router.get("/search", response_model=AudioSearchResp)
def search_audio(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(6, ge=1, le=20),
    current_user: UserInDB = Depends(get_current_user),
) -> AudioSearchResp:
    return query_audio(request=request, q=q, k=k, current_user=current_user)

@router.post("/ask", response_model=AudioAskResp)
def ask_audio(req: AudioAskReq, request: Request, current_user: UserInDB = Depends(get_current_user)) -> AudioAskResp:
    question = (req.question or "").strip()
    k = max(1, min(int(req.k or 6), 20))

    allowed_vis = _compute_allowed_visibilities(current_user)
    allowed_vis_set = set(allowed_vis)
    vs = get_audio_vs()
    base = _absolute_base(request)

    where: dict[str, Any] = {"visibility": {"$in": allowed_vis}}
    if req.audio_id:
        where = {"$and": [{"visibility": {"$in": allowed_vis}}, {"audio_id": req.audio_id}]}

    fetch_k = min(max(k * 5, k), 50)
    try:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, filter=where)
    except TypeError:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, where=where)

    citations: list[AudioCitation] = []
    seen: set[tuple[str, str, int, int]] = set()

    for doc, score in docs_scores:
        md = doc.metadata or {}
        audio_id = str(md.get("audio_id") or "").strip()
        segment_id = str(md.get("segment_id") or "").strip()
        if not audio_id or not segment_id:
            continue

        try:
            start_ms = int(md.get("start_ms") or 0)
            end_ms = int(md.get("end_ms") or 0)
        except Exception:
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue

        key = (audio_id, segment_id, start_ms, end_ms)
        if key in seen:
            continue
        seen.add(key)

        db_doc = mysql_audio.get_audio_document(audio_id)
        if not db_doc:
            continue
        doc_vis = (db_doc.get("visibility") or "").strip().lower()
        if doc_vis not in allowed_vis_set:
            continue

        text = (doc.page_content or "").strip()
        citations.append(
            AudioCitation(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                clip_url=_clip_url(base, audio_id, start_ms, end_ms),
                score=float(score) if score is not None else None,
            )
        )
        if len(citations) >= k:
            break

    if not citations:
        return AudioAskResp(question=question, answer="没有检索到相关音频片段。", citations=[])

    api_key = getattr(settings, "openai_api_key", "") or ""
    model = getattr(settings, "model_name", "") or "gpt-4o-mini"

    if not api_key:
        return AudioAskResp(
            question=question,
            answer="(未配置 OPENAI_API_KEY) 已返回相关音频片段引用，可先基于citations手动判断。",
            citations=citations,
        )

    messages = _build_rag_messages(question, citations, req.system_prompt)
    answer = _openai_chat_complete(model=model, api_key=api_key, messages=messages, timeout_s=90.0)

    return AudioAskResp(question=question, answer=answer, citations=citations)

@router.get("/docs/{audio_id}/segments")
def list_audio_segments_api(audio_id: str, current_user: UserInDB = Depends(get_current_user)):
    doc = mysql_audio.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    _ensure_can_access_visibility(current_user, doc.get("visibility") or "")
    segs = mysql_audio.list_audio_segments(audio_id)
    return {
        "audio_id": audio_id,
        "visibility": (doc.get("visibility") or "public"),
        "segment_count": len(segs),
        "segments": segs,
    }

@router.get("/docs/{audio_id}/transcript")
def get_audio_transcript_api(audio_id: str, current_user: UserInDB = Depends(get_current_user)):
    doc = mysql_audio.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    _ensure_can_access_visibility(current_user, doc.get("visibility") or "")
    out = mysql_audio.get_audio_transcript(audio_id)
    out["visibility"] = (doc.get("visibility") or "public")
    return out

@router.delete("/docs/{audio_id}")
def delete_audio_doc_api(audio_id: str, current_user: UserInDB = Depends(get_current_user)):
    _require_manage_docs(current_user)

    doc = mysql_audio.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    # 1) delete vectors
    vec_deleted = 0
    try:
        vec_deleted = int(delete_by_audio_id(audio_id) or 0)
    except Exception:
        vec_deleted = 0

    # 2) delete db rows
    db_deleted = mysql_audio.delete_audio_document_cascade(audio_id)

    # 3) delete files
    files_deleted: list[str] = []
    errors: list[str] = []

    # raw audio
    try:
        p = Path(doc.get("stored_path") or "")
        if p.exists() and p.is_file():
            p.unlink()
            files_deleted.append(str(p))
    except Exception as e:
        errors.append(f"delete raw failed: {e}")

    # wav
    try:
        wav = WAV_DIR / f"{audio_id}.wav"
        if wav.exists() and wav.is_file():
            wav.unlink()
            files_deleted.append(str(wav))
    except Exception as e:
        errors.append(f"delete wav failed: {e}")

    # clips: normally temp-delete, but if crashed, may remain; best-effort cleanup prefix
    try:
        if CLIP_DIR.exists():
            for fp in CLIP_DIR.glob(f"{audio_id}_*.mp3"):
                try:
                    fp.unlink()
                    files_deleted.append(str(fp))
                except Exception:
                    pass
    except Exception as e:
        errors.append(f"delete clips failed: {e}")

    return {
        "audio_id": audio_id,
        "vectors_deleted": vec_deleted,
        "db_deleted": db_deleted,
        "files_deleted": files_deleted,
        "errors": errors,
    }


@router.post("/docs/{audio_id}/reindex")
def reindex_audio_doc_api(audio_id: str, current_user: UserInDB = Depends(get_current_user)):
    _require_manage_docs(current_user)

    if mysql_audio.is_audio_running(audio_id):
        raise HTTPException(status_code=409, detail="audio is running, try later")

    doc = mysql_audio.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    mysql_audio_job.create_job(job_id, audio_id, overwrite=False, delete_old_file=False, old_stored_path=None)

    async_result = audio_reindex_task.apply_async(
        args=[job_id, audio_id],
        queue=getattr(settings, "celery_audio_queue", "audio"),
    )
    mysql_audio_job.bind_task(job_id, async_result.id)

    return {
        "job_id": job_id,
        "audio_id": audio_id,
        "celery_task_id": async_result.id,
        "status_url": f"/audio/jobs/{job_id}",
    }

def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _openai_stream(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float = 120.0,
) -> Iterable[str]:
    """
    Yield token strings from OpenAI ChatCompletions stream.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": True,
    }

    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code >= 400:
                text = r.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI error: {r.status_code} {text[:300]}")

            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    raw = line[len("data: ") :].strip()
                else:
                    continue

                if raw == "[DONE]":
                    break

                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        yield str(content)
                except Exception:
                    continue


@router.post("/ask/stream")
def ask_audio_stream(req: AudioAskReq, request: Request, current_user: UserInDB = Depends(get_current_user)):
    question = (req.question or "").strip()
    k = max(1, min(int(req.k or 6), 20))
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    allowed_vis = _compute_allowed_visibilities(current_user)
    allowed_vis_set = set(allowed_vis)

    vs = get_audio_vs()
    base = _absolute_base(request)

    where: dict[str, Any] = {"visibility": {"$in": allowed_vis}}
    if req.audio_id:
        where = {"$and": [{"visibility": {"$in": allowed_vis}}, {"audio_id": req.audio_id}]}

    fetch_k = min(max(k * 5, k), 50)
    try:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, filter=where)
    except TypeError:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, where=where)

    citations: list[AudioCitation] = []
    seen: set[tuple[str, str, int, int]] = set()
    for doc, score in docs_scores:
        md = doc.metadata or {}
        audio_id = str(md.get("audio_id") or "").strip()
        segment_id = str(md.get("segment_id") or "").strip()
        if not audio_id or not segment_id:
            continue

        try:
            start_ms = int(md.get("start_ms") or 0)
            end_ms = int(md.get("end_ms") or 0)
        except Exception:
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue

        key = (audio_id, segment_id, start_ms, end_ms)
        if key in seen:
            continue
        seen.add(key)

        db_doc = mysql_audio.get_audio_document(audio_id)
        if not db_doc:
            continue
        doc_vis = (db_doc.get("visibility") or "").strip().lower()
        if doc_vis not in allowed_vis_set:
            continue

        text = (doc.page_content or "").strip()
        citations.append(
            AudioCitation(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                clip_url=_clip_url(base, audio_id, start_ms, end_ms),
                score=float(score) if score is not None else None,
            )
        )
        if len(citations) >= k:
            break

    api_key = getattr(settings, "openai_api_key", "") or ""
    model = getattr(settings, "model_name", "") or "deepseek-chat"

    def gen():
        # meta first
        yield _sse(
            "meta",
            {
                "question": question,
                "allowed_visibilities": allowed_vis,
                "citations": [c.model_dump() for c in citations],
            },
        )

        if not citations:
            yield _sse("token", {"text": "没有检索到相关音频片段。"})
            yield _sse("done", {"ok": True})
            return

        if not api_key:
            yield _sse("token", {"text": "(未配置 OPENAI_API_KEY) 只能返回 citations，无法流式生成答案。"})
            yield _sse("done", {"ok": True})
            return

        messages = _build_rag_messages(question, citations, req.system_prompt)

        # stream tokens
        try:
            for tok in _openai_stream(api_key=api_key, model=model, messages=messages, timeout_s=180.0):
                yield _sse("token", {"text": tok})
        except Exception as e:
            yield _sse("error", {"detail": str(e)})
        finally:
            yield _sse("done", {"ok": True})

    return StreamingResponse(gen(), media_type="text/event-stream")
