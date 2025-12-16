# 处理请假工单数据库的操作
import json
from typing import Optional, Any, Dict

import pymysql
from contextlib import contextmanager
from app.config import settings


@contextmanager
def get_conn():
    conn = pymysql.connect(
        host=settings.MYSQL_HOST, port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DB, charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        yield conn
    finally:
        conn.close()


def get_leave_balance(requester: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT annual_days, sick_days, personal_days FROM leave_balances WHERE requester=%s",
                (requester,)
            )
            return cur.fetchone()


def insert_leave_request(req: dict) -> str:
    """
    req expects keys: leave_id, requester, leave_type, start_time, end_time, duration_days, reason
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO leave_requests
                (leave_id, requester, leave_type, start_time, end_time, duration_days, reason, status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'PENDING')
                """,
                (
                    req["leave_id"], req["requester"], req["leave_type"],
                    req["start_time"], req["end_time"], req["duration_days"],
                    req.get("reason")
                )
            )
    return req["leave_id"]


def get_leave_request(leave_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM leave_requests WHERE leave_id=%s", (leave_id,))
            return cur.fetchone()


def cancel_leave_request(leave_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE leave_requests SET status='CANCELLED' WHERE leave_id=%s AND status='PENDING'",
                (leave_id,)
            )
            return cur.rowcount > 0

def get_recent_leave_requests(requester: str , limit: int=5) -> list[dict]:
    limit = max(1,min(int(limit),20))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT leave_id, leave_type, start_time, end_time, duration_days, status, reason, created_at "
                "FROM leave_requests WHERE requester=%s "
                "ORDER BY id DESC LIMIT %s",
                (requester,limit),
            )
            return cur.fetchall()

def update_leave_request(leave_id: str, fields: dict) -> bool:
    """
    Only update PENDING requests.
    fields can include: leave_type, start_time, end_time, duration_days, reason
    """
    # 使用集合进行字段白名单过滤，防止SQL注入以及误操作
    allowed = {"leave_type", "start_time", "end_time", "duration_days", "reason"}
    sets = []
    params = []
    for k, v in fields.items():
        if k in allowed and v is not None:
            # 动态SQL构建
            sets.append(f"{k}=%s")  # sets=['leave_type=%s', 'end_time=%s']
            params.append(v)  # params=['年假', 'xx年月日']

    if not sets:
        return False

    params.extend([leave_id])
    sql = (
        "UPDATE leave_requests SET " + ", ".join(sets) +
        " WHERE leave_id=%s AND status='PENDING'"
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount > 0

def approve_leave_request(leave_id: str, approver: str) -> bool:
    """
    Approve a leave request.
    Only PENDING requests can be approved.
    Return True if updated, False otherwise.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE leave_requests SET status='APPROVED' "
                "WHERE leave_id=%s AND status='PENDING'",
                (leave_id,),
            )
            return cur.rowcount > 0


def reject_leave_request(leave_id: str, approver: str, reason: str | None = None) -> bool:
    """
    Reject a leave request.
    Only PENDING requests can be rejected.
    Optionally override reason.
    Return True if updated, False otherwise.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE leave_requests "
                "SET status='REJECTED', reason=COALESCE(%s, reason) "
                "WHERE leave_id=%s AND status='PENDING'",
                (reason, leave_id),
            )
            return cur.rowcount > 0


def create_ticket(
        question: str,
        user_id: Optional[str] = None,
        user_role: str = "public",
        status: str = "pending",
        priority: str = "normal",
        category: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None  # 如果需要额外的原因字段
) -> bool:
    """
    创建工单并插入数据库。
    返回新创建的工单ID，失败返回None。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # 序列化tags为JSON
            tags_json = json.dumps(tags) if tags else None

            cur.execute(
                """
                INSERT INTO tickets (question,
                                     user_id,
                                     user_role,
                                     status,
                                     priority,
                                     category,
                                     tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (question, user_id, user_role, status, priority, category, tags_json),
            )
            conn.commit()
            return cur.lastrowid  # 返回自增ID

if __name__=="__main__":
    get_conn()
    print(get_leave_balance("tom"))