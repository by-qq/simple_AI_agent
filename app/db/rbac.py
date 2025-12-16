from __future__ import annotations
from typing import List, Set

from app.db.mysql_leave import get_conn


def get_user_roles(user_id: int) -> List[str]:
    sql = """
        SELECT r.code
        FROM roles r
        JOIN user_roles ur ON ur.role_id = r.id
        WHERE ur.user_id = %s
        GROUP BY r.code
        ORDER BY r.code
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
    return [r["code"] for r in rows]

def get_user_permissions(user_id: int) -> Set[str]:
    sql = """
        SELECT p.code
        FROM permissions p
        JOIN role_permissions rp ON rp.permission_id = p.id
        JOIN user_roles ur ON ur.role_id = rp.role_id
        WHERE ur.user_id = %s
        GROUP BY p.code
        ORDER BY p.code
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
    return {r["code"] for r in rows}

# if __name__=='__main__':
#     print(get_user_roles(1))
#     print(get_user_permissions(1))



