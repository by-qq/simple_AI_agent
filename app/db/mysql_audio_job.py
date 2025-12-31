from __future__ import annotations

from typing import Optional, Any

from app.db.mysql_pool import get_conn


def create_job(
    job_id: str,
    audio_id: str,
    *,
    overwrite: bool = False,
    delete_old_file: bool = False,
    old_stored_path: str | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audio_jobs (job_id, audio_id, status, progress, overwrite, delete_old_file, old_stored_path) "
                "VALUES (%s,%s,'queued',0,%s,%s,%s)",
                (job_id, audio_id, int(overwrite), int(delete_old_file), old_stored_path),
            )


def bind_task(job_id: str, celery_task_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audio_jobs SET celery_task_id=%s WHERE job_id=%s",
                (celery_task_id, job_id),
            )


def update_job(job_id: str, *, status: str | None = None, progress: int | None = None, message: str | None = None) -> None:
    fields = []
    args: list[Any] = []
    if status is not None:
        fields.append("status=%s")
        args.append(status)
    if progress is not None:
        fields.append("progress=%s")
        args.append(int(progress))
    if message is not None:
        fields.append("message=%s")
        args.append(message)
    if not fields:
        return
    sql = f"UPDATE audio_jobs SET {', '.join(fields)} WHERE job_id=%s"
    args.append(job_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))


def get_job(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, audio_id, celery_task_id, status, progress, message, cancel_requested, overwrite, "
                "delete_old_file, old_stored_path, cancelled_at, created_at, updated_at "
                "FROM audio_jobs WHERE job_id=%s LIMIT 1",
                (job_id,),
            )
            return cur.fetchone()


def request_cancel(job_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audio_jobs SET cancel_requested=1, message='cancel requested' WHERE job_id=%s",
                (job_id,),
            )
            return cur.rowcount > 0


def is_cancel_requested(job_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT cancel_requested FROM audio_jobs WHERE job_id=%s LIMIT 1", (job_id,))
            row = cur.fetchone()
            return bool(row and int(row.get("cancel_requested", 0)) == 1)


def mark_cancelled(job_id: str, message: str = "cancelled") -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audio_jobs SET status='cancelled', progress=100, message=%s, cancelled_at=NOW() WHERE job_id=%s",
                (message, job_id),
            )


def get_job_flags(job_id: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT overwrite, delete_old_file, old_stored_path, cancel_requested FROM audio_jobs WHERE job_id=%s LIMIT 1",
                (job_id,),
            )
            return cur.fetchone() or {}
