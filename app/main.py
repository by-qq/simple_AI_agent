from __future__ import annotations

from starlette.middleware.cors import CORSMiddleware

from app.api import auth_api, rbac_api
from app.db.mysql_rbac import get_role_permissions
from app.db.redis_session import load_session, save_session
from app.deps import get_vs
from app.ingestion.loader import load_single_file, split_with_visibility, load_docs, split_docs
from app.config import settings
import time
import uuid
from pathlib import Path
from typing import Optional
import chromadb

from fastapi import FastAPI, UploadFile, File, Form, HTTPException


from app.models.chat_models import ChatResp, ChatReq
from app.router_graph import router_graph
from app.security.rbac.perms import require_permission, check_permission
app = FastAPI(title="Enterprise KB Assistant")

import fastapi_cdn_host # 解决docs访问超时导致的空白网页问题
fastapi_cdn_host.patch_docs(app)

app.include_router(auth_api.router)
app.include_router(rbac_api.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 本地开发可以先全开，线上再收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DOCS_DIR = Path("./data/docs")
DATA_DOCS_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/chat",response_model=ChatResp)
async def chat(req: ChatReq):
    # 这里可能需要更改
    # user_role = req.get("user_role")
    # current_user_code = get_role_permissions(user_role)
    # require_permission(current_user_code, "kb.view_public")
    # require_permission(current_user_code, "kb.view_internal")
    # check_permission(current_user, "kb.view_public")
    # check_permission(current_user, "kb.view_internal")

    # req.model_dump()，将请求对象转化为字典格式
    # 将字典数据输入到图中，之后就按照图定义的结构开始执行并返回最终结果
    # out = router_graph.invoke(req.model_dump())
    # return {"answer": out["answer"]}
    payload = req.model_dump()
    text = payload["text"] or payload["question"] or ""
    sid = payload.get("session_id") or f"sid-{uuid.uuid4().hex[:10]}"

    prev_state = load_session(sid)
    if prev_state:
        merged = {**prev_state, **payload}
        merged["text"] = text
        payload = merged

    out = router_graph.invoke(payload)

    new_state = {**payload,**out}
    save_session(sid,new_state)


    return {
        "answer": out.get("answer"),
        "session_id": sid,
        "active_route": new_state.get("active_route"),
    }

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    doc_id: Optional[str] = Form(None),
):
    """
    Upload a single document and upsert into Chroma.

    - Saves file to ./data/docs/
    - Loads & splits into chunks
    - Attaches visibility/doc_id metadata
    - Upserts into the configured Chroma collection
    """

    # require_permission(current_user_code, "kb.manage_docs")
    # check_permission(current_user, "kb.manage_docs")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    visibility = (visibility or "public").strip().lower()

    suffix = Path(file.filename).suffix
    # uuid全局唯一标识符，suffix是扩展名
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    save_path = DATA_DOCS_DIR / safe_name   # 真正部署的时候使用云盘

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    save_path.write_bytes(content)

    docs = load_single_file(save_path)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Unsupported or empty file type: {suffix}")

    chunks = split_with_visibility(docs, visibility=visibility, doc_id=doc_id)

    vs = get_vs()
    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        pass

    return {
        "saved_as": str(save_path),
        "visibility": visibility,
        "doc_id": doc_id,
        "chunks": len(chunks),
    }


@app.post("/reindex")
def reindex(
    visibility_default: str = Form("public"),
):
    """
    Full rebuild of the collection from ./data/docs.

    WARNING: This deletes the current collection first.
    """
    # require_permission(current_user_code, "kb.manage_docs")
    # check_permission(current_user, "kb.manage_docs")

    visibility_default = (visibility_default or "public").strip().lower()

    # 1) Delete & recreate collection via chromadb client
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    client.get_or_create_collection(settings.collection_name)

    # 2) Rebuild using LangChain wrapper
    vs = get_vs()
    raw_docs = load_docs(str(DATA_DOCS_DIR))
    if not raw_docs:
        return {"chunks": 0, "docs": 0, "message": "No documents found in data/docs"}

    chunks = split_docs(raw_docs)
    for c in chunks:
        c.metadata = dict(c.metadata or {})
        c.metadata.setdefault("visibility", visibility_default)

    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        pass

    return {"docs": len(raw_docs), "chunks": len(chunks), "visibility_default": visibility_default}


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)


