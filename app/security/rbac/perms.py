from typing import Iterable, Any, Callable

from fastapi import HTTPException, status, Depends

from app.api.auth_api import get_current_user
from app.db.mysql_rbac import get_user_permissions
from app.models.auth_models import UserInDB

# 返回set主要是为了去重
def _resolve_perms(*,
                   user: UserInDB | None = None,
                   user_id: int | None = None,
                   perms: Iterable[str] | None = None) -> set[str]:
    if perms:
        return set(perms)

    if user is not None:
        p = getattr(user, "permissions", None)
        if p:
            return set(p)
        user_id = user_id or getattr(user, "id", None)

    if user_id is not None:
        return set(get_user_permissions(int(user_id)))

    return set()


def _raise_403(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
# ----------------helper-----------------
def require_permission(arg1: Any, arg2: str | None = None) -> Callable | None:
    # arg1是字符串，arg2必须是None，此时arg1里面放的是权限code
    if isinstance(arg1, str) and arg2 is None:
        perm_code = arg1

        def _checker(_user: UserInDB = Depends(get_current_user)) -> bool:
            if getattr(_user, "is_super_admin", False):
                return True
            if perm_code not in _resolve_perms(user=_user):
                _raise_403(f"Missing permission: {perm_code}")
            return True

        return _checker

    # 能运行到此时，一定是因为不满足arg1是字符串且arg2是None
    # 此时又是另外一种情况了，这种情况下arg1是userInDB，arg2是权限code
    user = arg1
    perm_code = arg2
    if not perm_code or not isinstance(perm_code, str):
        raise TypeError("require_permission(user, perm_code): perm_code must be str")

    if isinstance(user, dict):
        if user.get("is_super_admin"):
            return None
        perms = _resolve_perms(user_id=user.get("user_id"), perms=user.get("permissions"))
        if perm_code not in perms:
            _raise_403(f"Missing permission: {perm_code}")
        return None

    if getattr(user, "is_super_admin", False):
        return None
    perms = _resolve_perms(user=user)
    if perm_code not in perms:
        _raise_403(f"Missing permission: {perm_code}")
    return None

def check_permission(user: Any, perm_code: str) -> None:
    """检测user这个参数所表示的用户是否拥有perm_code权限，有就返回，没有就抛出异常"""
    if not perm_code or not isinstance(perm_code, str):
        raise TypeError("perm_code must be a non-empty str")

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Not authenticated",)

    if isinstance(user, dict):  # user是一个字段，实际上我们要的是类似UserInDB这种东西
        if user.get("is_super_admin", False):
            return  # 超级管理员，直接过，过不了的都抛出异常
        perms = set(user.get("permissions") or [])
        if perm_code not in perms:
            _raise_403(f"Missing permission: {perm_code}")
        return

    if getattr(user, "is_super_admin", False):
        return

    perms = set(getattr(user, "permissions", []) or [])
    if not perms:
        perms = set(get_user_permissions(user.id))

    if perm_code not in perms:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission: {perm_code}",
        )

def has_permission(*,
                   user_id: int | None = None,
                   perms: Iterable[str] | None = None,
                   perm_code: str) -> bool:
    # 判断perm_code是不是在user_id所拥有的perms里面，即检验是否有权限
    resolved = _resolve_perms(user_id=user_id, perms=perms)
    return perm_code in resolved

def allowed_kb_visibilities(perms: Iterable[str] | None) -> list[str]:
    """KB检索时，基于权限决定能看哪些visibility。"""
    p = set(perms or [])
    allowed = ["public"]
    if "kb.view_internal" in p or "kb.manage_docs" in p:
        allowed.append("internal")
    return allowed


