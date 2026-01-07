from __future__ import annotations

from pathlib import Path

from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.config import settings
from app.db import mysql_audio
from app.db import mysql_audio_job
from app.db.chroma_admin import delete_by_audio_id
from app.tools.pipeline import run_audio_ingest_pipeline



def _check_cancel(job_id: str):
    if mysql_audio_job.is_cancel_requested(job_id):
        mysql_audio_job.mark_cancelled(job_id)
        raise Ignore()


@celery_app.task(
    name="app.tasks.audio_tasks.audio_ingest_task",  # 任务的全局唯一标识符，命名约定为应用名.模块路径.函数名
    bind=True,                                      # 让任务函数的第一个参数变成self，绑定到celery中
    autoretry_for=(Exception,),                     # 指定所有异常类型触发自动重试
    retry_backoff=True,                             # 启用指数退避重试，也就是重试间隔时间成指数增长，避免崩
    retry_jitter=True,                              # 为重试增加随机抖动，防止多个任务同时重试造成惊群效应（）
    retry_kwargs={"max_retries": 3},)               # 最多重试3次
def audio_ingest_task(self, job_id: str, audio_id: str):
    flags = mysql_audio_job.get_job_flags(job_id) # flags包含5列overwrite, delete_old_file, old_stored_path, cancel_requested
    old_path = flags.get("old_stored_path")
    delete_old_file = bool(int(flags.get("delete_old_file", 0) or 0))

    mysql_audio_job.update_job(job_id, status="running", progress=1, message="starting")
    mysql_audio.update_audio_status(audio_id, status="running") # 控制元数据，让任务开始进度为xxx

    _check_cancel(job_id)   # 若前台在这个期间发出取消请求，这里才会真正的取消

    doc = mysql_audio.get_audio_document(audio_id)  # doc是取这个音频文件在数据库中这一行的全部信息
    if not doc:
        raise RuntimeError("audio_document not found")

    raw_path = Path(doc["stored_path"])
    if not raw_path.exists():   # 存储路径不存在也抛出异常
        raise RuntimeError("stored audio file missing")

    mysql_audio_job.update_job(job_id, progress=5, message="cleaning old vectors")
    delete_by_audio_id(audio_id)    # 清空音频文档向量数据库中这个音频id相关的内容

    _check_cancel(job_id)

    mysql_audio_job.update_job(job_id, progress=10, message="transcribing/indexing")
    res = run_audio_ingest_pipeline(
        audio_id=audio_id,
        raw_path=raw_path,
        original_filename=doc["original_filename"],
        visibility=doc["visibility"],
        language=doc.get("language"),
        wav_dir=Path(settings.audio_wav_dir),
    )       # 调用流水线真正的做文件上传、切割等操作

    _check_cancel(job_id)

    mysql_audio.update_audio_indexed(
        audio_id=audio_id,
        duration_ms=int(res["duration_ms"]),
        language=res.get("language"),
        segment_count=int(res["segments"]),
        status="indexed",
    )

    mysql_audio_job.update_job(job_id, status="succeeded", progress=100, message=f"indexed {res['segments']} segments")
    # 新的文件已经处理好，旧的文件就可以删除了
    if delete_old_file and old_path and old_path != str(raw_path):
        try:
            p = Path(str(old_path))
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:   # 实际上这里记录日志
            print('注意异常！！！！！！！！！！！！！！！！1')

    return res

