
# 将chromadb中的每一个文档信息都存入数据库

from __future__ import annotations

from typing import Optional, Any

from app.db.mysql_leave import get_conn


def upsert_kb_document(
    *,
    doc_id: str,
    original_filename: str,
    stored_path: str,
    visibility: str,
    uploader_user_id: int | None,
    uploader_username: str | None,
    chunk_count: int = 0,
) -> None:
# 插入或者更新一个文档的信息
    sql = """
        INSERT INTO kb_documents
          (doc_id, original_filename, stored_path, visibility, uploader_user_id, uploader_username, chunk_count)
        VALUES
          (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          original_filename=VALUES(original_filename),
          stored_path=VALUES(stored_path),
          visibility=VALUES(visibility),
          uploader_user_id=VALUES(uploader_user_id),
          uploader_username=VALUES(uploader_username),
          chunk_count=VALUES(chunk_count),
          is_deleted=0
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    doc_id,
                    original_filename,
                    stored_path,
                    visibility,
                    uploader_user_id,
                    uploader_username,
                    int(chunk_count or 0),
                ),
            )


def list_kb_documents(*, limit: int = 50,
                      offset: int = 0,
                      visibility: str | None = None,
                      q: str | None = None,
                      order_by: str = "updated_at",
                      desc: bool = True,
                      ) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    allowed_order = {"updated_at", "created_at", "original_filename", "doc_id", "visibility"}
    if order_by not in allowed_order:
        order_by = "updated_at"

    order_dir = "DESC" if desc else "ASC"
    sql = (
        "SELECT doc_id, original_filename, stored_path, visibility, uploader_user_id, uploader_username, "
        "chunk_count, created_at, updated_at "
        "FROM kb_documents WHERE is_deleted=0 "
    )
    args: list[Any] = []
    if visibility:
        sql += "AND visibility=%s "
        args.append(visibility)
    if q:
        q = q.strip()
        if q:
            sql += "AND (original_filename LIKE %s OR doc_id LIKE %s) "
            like = f"%{q}%"
            args.extend([like, like])

    sql += f"ORDER BY {order_by} {order_dir} LIMIT %s OFFSET %s"
    args.extend([limit, offset])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            return cur.fetchall()


def get_kb_document(doc_id: str,is_deleted:bool) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, original_filename, stored_path, visibility, uploader_user_id, uploader_username, chunk_count, created_at, updated_at "
                "FROM kb_documents WHERE doc_id=%s AND is_deleted=%s LIMIT 1",
                (doc_id,is_deleted),
            )
            return cur.fetchone()


def update_kb_document_visibility(doc_id: str, visibility: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_documents SET visibility=%s WHERE doc_id=%s AND is_deleted=0",
                (visibility, doc_id),
            )
            return cur.rowcount > 0


def update_kb_document_chunk_count(doc_id: str, chunk_count: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_documents SET chunk_count=%s WHERE doc_id=%s AND is_deleted=0",
                (int(chunk_count or 0), doc_id),
            )
            return cur.rowcount > 0


def soft_delete_kb_document(doc_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE kb_documents SET is_deleted=1 WHERE doc_id=%s AND is_deleted=0",
                (doc_id,),
            )
            return cur.rowcount > 0

def count_kb_documents(*, visibility: str | None = None, q: str | None = None) -> int:
    sql = "SELECT COUNT(*) AS cnt FROM kb_documents WHERE is_deleted=0 "
    args: list[Any] = []

    if visibility:
        sql += "AND visibility=%s "
        args.append(visibility)

    if q:
        q = q.strip()
        if q:
            sql += "AND (original_filename LIKE %s OR doc_id LIKE %s) "
            like = f"%{q}%"
            args.extend([like, like])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            row = cur.fetchone()
            return int(row["cnt"]) if row and "cnt" in row else 0


if __name__=="__main__":
    ...
    # upsert_kb_document(...)