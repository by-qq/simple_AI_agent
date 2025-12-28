from __future__ import annotations

from typing import List

from app.db.mysql_pool import get_conn


def get_visibility_name() -> List[str]:
    """获得所有可见性名称"""
    sql = """
        SELECT v.name
        FROM visibility v
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return rows