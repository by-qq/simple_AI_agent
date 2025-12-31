from fastapi import HTTPException

from app.db.mysql_visibility import get_visibility_name
from app.security.rbac.perms import _resolve_perms, allowed_kb_visibilities

ALLOWED_VISIBILITIES = name_set = {item['name'] for item in get_visibility_name()}

def normalize_visibility(v: str) -> str:
    v = (v or "").strip().lower()
    if v not in ALLOWED_VISIBILITIES:
        raise HTTPException(status_code=400, detail=f"invalid visibility: {v}")
    return v

def compute_allowed_kb_visibilities(user) -> list[str]:
    perms = _resolve_perms(user=user)
    return allowed_kb_visibilities(perms)