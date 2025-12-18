from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.db import mysql_rbac
from app.models.rbac_models import SetRolePermsReq, SetUserRolesReq
from app.security.rbac.perms import require_permission
from app.security.rbac.perms_code import Permission

router = APIRouter(prefix="/rbac", tags=["rbac"])

@router.get("/roles")
def api_list_roles(_: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_ROLES ))):
    return {"roles": mysql_rbac.list_roles()}

@router.get("/permissions")
def api_list_permissions(module: Optional[str] = None, _: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_ROLES ))):
    return {"permissions": mysql_rbac.list_permissions(module=module)}

@router.get("/roles/{role_code}/permissions")
def api_get_role_permissions(role_code: str, _: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_ROLES ))):
    return {"role": role_code, "permissions": mysql_rbac.get_role_permissions(role_code)}

@router.put("/roles/{role_code}/permissions")
def api_set_role_permissions(role_code: str, req: SetRolePermsReq, _: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_ROLES ))):
    try:
        mysql_rbac.set_role_permissions(role_code, req.perm_codes)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "role": role_code, "permissions": mysql_rbac.get_role_permissions(role_code)}

@router.get("/users/{username}/roles")
def api_get_user_roles(username: str, _: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_USERS ))):
    uid = mysql_rbac.find_user_id(username)
    if uid is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {"user": username, "user_id": uid, "roles": mysql_rbac.get_user_roles(uid)}

@router.put("/users/{username}/roles")
def api_set_user_roles(username: str, req: SetUserRolesReq, _: bool = Depends(require_permission(Permission.PERM_SYSTEM_MANAGE_USERS ))):
    uid = mysql_rbac.find_user_id(username)
    if uid is None:
        raise HTTPException(status_code=404, detail="user not found")
    mysql_rbac.set_user_roles(uid, req.role_codes)
    return {"ok": True, "user": username, "user_id": uid, "roles": mysql_rbac.get_user_roles(uid)}
