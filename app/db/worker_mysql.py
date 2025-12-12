# 处理请假工单数据库的操作

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

if __name__=="__main__":
    get_conn()
    print(get_leave_balance("tom"))