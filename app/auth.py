
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Header, status

from app.db.mysql_auth import get_user_by_username, create_user, update_last_login
from app.models.auth_models import RegisterReq, UserInDB, TokenResp, LoginReq
from app.security.security import verify_password, create_access_token, decode_token



router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------- Dependencies ----------------
# 该函数要求必须有token才能够访问的接口
def get_current_user(authorization: str | None = Header(default=None)) -> UserInDB:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    # 截取token中bearer后面的
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    u = get_user_by_username(username)
    if not u or not u.get("is_active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled")

    return UserInDB(
        id=int(u["id"]),
        username=u["username"],
        email=u.get("email"),
        phone=u.get("phone"),
        full_name=u.get("full_name"),
        is_active=bool(u["is_active"]),
        is_super_admin=bool(u["is_super_admin"]),
    )
# 主要用于没有用户都能够进行访问的设计
def get_current_user_optional(authorization: str | None = Header(default=None)) -> UserInDB | None:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    try:
        return get_current_user(authorization=authorization)
    except HTTPException:
        return None

# ---------------- Routes ----------------

@router.post("/register", response_model=UserInDB)
def register(req: RegisterReq):
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="username already exists")
    u = create_user(req)
    return UserInDB(
        id=int(u["id"]),
        username=u["username"],
        email=u.get("email"),
        phone=u.get("phone"),
        full_name=u.get("full_name"),
        is_active=bool(u["is_active"]),
        is_super_admin=bool(u["is_super_admin"]),
    )

@router.post("/login", response_model=TokenResp)
def login(req: LoginReq):
    u = get_user_by_username(req.username)
    if not u:
        raise HTTPException(status_code=401, detail="bad credentials")

    if not verify_password(req.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="bad credentials")

    if not u.get("is_active"):
        raise HTTPException(status_code=403, detail="user disabled")

    update_last_login(int(u["id"]))
    token = create_access_token({"sub": u["username"], "uid": int(u["id"])})
    return TokenResp(access_token=token)

@router.get("/me", response_model=UserInDB)
def me(current_user: UserInDB = Depends(get_current_user)):
    return current_user
