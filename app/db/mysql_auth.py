from datetime import datetime

from app.db.mysql_pool import get_conn
from app.models.auth_models import RegisterReq
from app.security.security import hash_password


def get_user_by_username(username: str) -> dict | None:
    sql = "SELECT * FROM users WHERE username=%s LIMIT 1"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (username,))
            return cur.fetchone()

def create_user(data: RegisterReq) -> dict:
    pwd_hash = hash_password(data.password)
    sql = """
        INSERT INTO users (username, email, phone, password_hash, full_name, is_active, is_super_admin)
        VALUES (%s,%s,%s,%s,%s,1,0)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (data.username, data.email, data.phone, pwd_hash, data.full_name))
            user_id = cur.lastrowid

    # 默认给 public 角色
    sql_bind = """
        INSERT IGNORE INTO user_roles (user_id, role_id)
        SELECT %s, r.id FROM roles r WHERE r.code='public'
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_bind, (user_id,))

    u = get_user_by_username(data.username)
    assert u is not None
    return u

def update_last_login(user_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET last_login_at=%s WHERE id=%s", (datetime.now(), user_id))

