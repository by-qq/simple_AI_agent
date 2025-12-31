from __future__ import annotations

from typing import Any, Optional

from app.db.mysql_pool import get_conn


def upsert_audio_document(
    *,
    audio_id: str,
    original_filename: str,
    stored_path: str,
    duration_ms: int,
    language: str | None,
    visibility: str,
    status: str,
    uploader_user_id: int | None,
    uploader_username: str | None,
    segment_count: int,
) -> None:
    sql = """
    INSERT INTO audio_documents
      (audio_id, original_filename, stored_path, duration_ms, language, visibility, status,
       uploader_user_id, uploader_username, segment_count)
    VALUES
      (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      original_filename=VALUES(original_filename),
      stored_path=VALUES(stored_path),
      duration_ms=VALUES(duration_ms),
      language=VALUES(language),
      visibility=VALUES(visibility),
      status=VALUES(status),
      uploader_user_id=VALUES(uploader_user_id),
      uploader_username=VALUES(uploader_username),
      segment_count=VALUES(segment_count)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    audio_id, original_filename, stored_path, int(duration_ms), language,
                    visibility, status, uploader_user_id, uploader_username, int(segment_count)
                ),
            )


def replace_audio_segments(audio_id: str, segments: list[dict[str, Any]]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audio_segments WHERE audio_id=%s", (audio_id,))
            if segments:
                cur.executemany(
                    "INSERT INTO audio_segments (audio_id, segment_idx, start_ms, end_ms, text) VALUES (%s,%s,%s,%s,%s)",
                    [
                        (audio_id, int(s["segment_idx"]), int(s["start_ms"]), int(s["end_ms"]), s["text"])
                        for s in segments
                    ],
                )


def get_audio_document(audio_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT audio_id, original_filename, stored_path, duration_ms, language, visibility, status, "
                "uploader_user_id, uploader_username, segment_count, created_at, updated_at "
                "FROM audio_documents WHERE audio_id=%s LIMIT 1",
                (audio_id,),
            )
            return cur.fetchone()

def get_audio_segment(audio_id: str, segment_idx: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT audio_id, segment_idx, start_ms, end_ms, text "
                "FROM audio_segments WHERE audio_id=%s AND segment_idx=%s LIMIT 1",
                (audio_id, int(segment_idx)),
            )
            return cur.fetchone()


def update_audio_status(audio_id: str, status: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE audio_documents SET status=%s WHERE audio_id=%s", (status, audio_id))


def update_audio_indexed(audio_id: str, duration_ms: int, language: str | None, segment_count: int, status: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audio_documents SET duration_ms=%s, language=%s, segment_count=%s, status=%s WHERE audio_id=%s",
                (int(duration_ms), language, int(segment_count), status, audio_id),
            )


def is_audio_running(audio_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM audio_documents WHERE audio_id=%s LIMIT 1", (audio_id,))
            row = cur.fetchone()
            return bool(row and row.get("status") in ("queued", "running"))



