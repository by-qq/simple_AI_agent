from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth_api import UserInDB, get_current_user
from app.db import mysql_kb
from app.tools.kb_loader import load_single_file, split_with_visibility
from app.models.kb_models import KBDocListItem, KBDocDetail, KBDocVisibilityUpdateReq, KBDocReembedResp, KBDocPageResp
from app.security.kb_visibility import normalize_visibility
from app.security.rbac.perms import check_permission
from app.db.chroma_admin import count_by_doc_id, delete_by_doc_id, update_visibility_by_doc_id

router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/docs", response_model=list[KBDocListItem])
def list_docs(
        visibility: Optional[str] = Query(default=None),
        q: Optional[str] = Query(default=None),  # ⚠️
        order_by: str = Query(default="updated_at"),
        desc: bool = Query(default=True),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        include_chroma_count: bool = Query(default=False),
        current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")
    rows = mysql_kb.list_kb_documents(
        limit=limit,
        offset=offset,
        visibility=visibility,
        q=q,  # ⚠️
        order_by=order_by,
        desc=desc,
    )

    out: list[dict] = []
    for r in rows:
        item = dict(r)
        if include_chroma_count:
            item["chunk_count"] = count_by_doc_id(item["doc_id"])
        out.append(item)
    return out

@router.get("/docs/page", response_model=KBDocPageResp)
def list_docs_page(
    visibility: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    order_by: str = Query(default="updated_at"),
    desc: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_chroma_count: bool = Query(default=False),
    current_user: UserInDB = Depends(get_current_user),
):
    # pdb.set_trace()  # 程序会在这里暂停
    check_permission(current_user, "kb.manage_docs")

    items = list_docs(
        visibility = visibility,
        q = q,
        order_by = order_by,
        desc = desc,
        limit = limit,
        offset = offset,
        include_chroma_count = include_chroma_count,
        current_user = current_user,
    )
    total = mysql_kb.count_kb_documents(visibility=visibility, q=q)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }

@router.get("/docs/{doc_id}", response_model=KBDocDetail)
def get_doc(
    doc_id: str,
    include_chroma_count: bool = Query(default=True),
    current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")

    row = mysql_kb.get_kb_document(doc_id,is_deleted=False)
    if not row:
        raise HTTPException(status_code=404, detail="doc not found")

    data = dict(row)
    data["chroma_chunk_count"] = count_by_doc_id(doc_id) if include_chroma_count else 0
    return data


@router.patch("/docs/{doc_id}/visibility", response_model=KBDocDetail)
def update_doc_visibility(
    doc_id: str,
    req: KBDocVisibilityUpdateReq,
    current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")

    row = mysql_kb.get_kb_document(doc_id,is_deleted=False)
    if not row:
        raise HTTPException(status_code=404, detail="doc not found")

    visibility = normalize_visibility(req.visibility)

    updated = update_visibility_by_doc_id(doc_id, visibility)
    mysql_kb.update_kb_document_visibility(doc_id, visibility)
    mysql_kb.update_kb_document_chunk_count(doc_id, count_by_doc_id(doc_id))

    new_row = mysql_kb.get_kb_document(doc_id,is_deleted=False) or {}
    data = dict(new_row)
    data["chroma_chunk_count"] = updated
    return data


@router.delete("/docs/{doc_id}")
def delete_doc(
    doc_id: str,
    delete_file: bool = Query(default=False),
    current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")

    row = mysql_kb.get_kb_document(doc_id,is_deleted=False)
    if not row:
        raise HTTPException(status_code=404, detail="doc not found")

    deleted_chunks = delete_by_doc_id(doc_id)
    mysql_kb.soft_delete_kb_document(doc_id)

    deleted_file = False
    if delete_file:
        try:
            p = Path(row["stored_path"])
            if p.exists() and p.is_file():
                p.unlink()
                deleted_file = True
        except Exception:
            deleted_file = False

    return {"ok": True, "doc_id": doc_id, "deleted_chunks": deleted_chunks, "deleted_file": deleted_file}


@router.post("/docs/{doc_id}/reembed", response_model=KBDocReembedResp)
def reembed_doc(
    doc_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    check_permission(current_user, "kb.manage_docs")

    row = mysql_kb.get_kb_document(doc_id,is_deleted=True)
    if not row:
        raise HTTPException(status_code=404, detail="doc not found")

    stored_path = Path(row["stored_path"])
    if not stored_path.exists():
        raise HTTPException(status_code=404, detail=f"stored file not found: {stored_path}")

    deleted = delete_by_doc_id(doc_id)

    docs = load_single_file(stored_path)
    if not docs:
        raise HTTPException(status_code=400, detail="unsupported or empty file")

    visibility = (row.get("visibility") or "public").strip().lower()
    extra_meta = {
        "original_filename": row.get("original_filename"),
        "stored_path": str(stored_path),
        "uploader_user_id": row.get("uploader_user_id"),
        "uploader_username": row.get("uploader_username"),
    }

    chunks = split_with_visibility(docs, visibility=visibility, doc_id=doc_id, extra_meta=extra_meta)

    from app.deps import get_vs
    vs = get_vs()
    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        pass

    new_cnt = count_by_doc_id(doc_id)
    # mysql_kb.update_kb_document_chunk_count(doc_id, new_cnt)
    # mysql_kb.update_kb_document_visibility(doc_id, visibility)
    print(str(extra_meta.values()),)
    mysql_kb.upsert_kb_document(doc_id=doc_id,
                                original_filename=extra_meta.get("original_filename"),
                                stored_path=extra_meta.get("stored_path"),
                                visibility=visibility,
                                uploader_user_id=extra_meta.get("uploader_user_id"),
                                uploader_username=extra_meta.get("uploader_username"),
                                chunk_count=new_cnt)
    return KBDocReembedResp(doc_id=doc_id, deleted_chunks=deleted, new_chunks=new_cnt, visibility=visibility)

