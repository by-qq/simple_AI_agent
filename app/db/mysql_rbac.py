from __future__ import annotations
from typing import List, Set, Optional

from app.db.mysql_leave import get_conn


# RBAC=Role-Based Access Control  基于角色的访问控制
def get_user_roles(user_id: int) -> List[str]:
    """根据用户的id拿到用户的角色，在灵活的状态下，用户可以有多个角色"""
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

def find_user_id(username: str) -> Optional[int]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username=%s LIMIT 1", (username,))
            row = cur.fetchone()
    return int(row["id"]) if row else None

def list_roles() -> list[dict]:
    """这个函数用于取出系统中所有的角色"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, code, name, description, is_system FROM roles ORDER BY code")
            return cur.fetchall()

def list_permissions(module: Optional[str] = None) -> list[dict]:
    sql = "SELECT id, code, name, module, description FROM permissions "
    args = []
    if module:
        sql += "WHERE module=%s "
        args.append(module)
    sql += "ORDER BY module, code"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            return cur.fetchall()

def _get_role_id(role_code: str) -> Optional[int]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM roles WHERE code=%s LIMIT 1", (role_code,))
            row = cur.fetchone()
    return int(row["id"]) if row else None

def get_role_permissions(role_code: str) -> list[str]:
    sql = """
        SELECT p.code
        FROM permissions p
        JOIN role_permissions rp ON rp.permission_id = p.id
        JOIN roles r ON r.id = rp.role_id
        WHERE r.code=%s
        ORDER BY p.code
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (role_code,))
            rows = cur.fetchall()
    return [r["code"] for r in rows]

def set_role_permissions(role_code: str, perm_codes: list[str]) -> None:
    """为了后续管理员给项目在运行时添加一些角色和权限，暂时用不上"""
    role_id = _get_role_id(role_code)
    if role_id is None:
        raise ValueError(f"role not found: {role_code}")

    perm_codes = sorted(set([p.strip() for p in (perm_codes or []) if p and p.strip()]))

    # 先找 permission ids
    perm_id_by_code: dict[str, int] = {}
    if perm_codes:
        placeholders = ",".join(["%s"] * len(perm_codes))
        sql = f"SELECT id, code FROM permissions WHERE code IN ({placeholders})"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(perm_codes))
                for row in cur.fetchall():
                    perm_id_by_code[row["code"]] = int(row["id"])

        missing = [c for c in perm_codes if c not in perm_id_by_code]
        if missing:
            raise ValueError(f"unknown permission codes: {missing}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # 清空再重建（最直观）
            cur.execute("DELETE FROM role_permissions WHERE role_id=%s", (role_id,))
            for code in perm_codes:
                cur.execute(
                    "INSERT IGNORE INTO role_permissions (role_id, permission_id) VALUES (%s,%s)",
                    (role_id, perm_id_by_code[code]),
                )

def set_user_roles(user_id: int, role_codes: list[str]) -> None:
    role_codes = [r.strip() for r in (role_codes or []) if r and r.strip()]
    if not role_codes:
        role_codes = ["public"]
    role_codes = sorted(set(role_codes))

    placeholders = ",".join(["%s"] * len(role_codes))
    sql = f"SELECT id, code FROM roles WHERE code IN ({placeholders})"

    role_id_by_code: dict[str, int] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(role_codes))
            for row in cur.fetchall():
                role_id_by_code[row["code"]] = int(row["id"])

    missing = [c for c in role_codes if c not in role_id_by_code]
    if missing:
        raise ValueError(f"unknown role codes: {missing}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_roles WHERE user_id=%s", (user_id,))
            for code in role_codes:
                cur.execute(
                    "INSERT IGNORE INTO user_roles (user_id, role_id) VALUES (%s,%s)",
                    (user_id, role_id_by_code[code]),
                )



