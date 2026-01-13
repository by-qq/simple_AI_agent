from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.middleware.cors import CORSMiddleware

from app.api import auth_api, rbac_api, kb_api, audio_api, audio_admin_api
from app.api.auth_api import get_current_user
from app.api.kb_api import normalize_visibility
from app.db import mysql_kb
from app.db.mysql_pool import init_pool, close_pool
from app.db.redis_session import load_session, save_session
from app.deps import get_vs
from app.tools.kb_loader import load_single_file, split_with_visibility, load_docs, split_docs
from app.config import settings
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends

from app.models.chat_models import ChatResp, ChatReq
from app.workflows.router_graph import router_graph
from app.security.rbac.perms import require_permission, check_permission
from app.db.chroma_admin import delete_by_doc_id, count_by_doc_id
from app.db.vectorstore import get_client

# 创建目录（移到 lifespan 外，因为这是配置项）
DATA_DOCS_DIR = Path("./data/docs")
DATA_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """sudo
    应用生命周期管理
    - 启动时：初始化连接池
    - 关闭时：关闭连接池
    """
    # 启动时执行
    print("Application starting up...")
    init_pool()  # 初始化数据库连接池
    yield
    # 关闭时执行
    print("Application shutting down...")
    close_pool()  # 关闭数据库连接池

# 使用 lifespan 参数创建 FastAPI 应用
app = FastAPI(
    title="Enterprise KB Assistant",
    version="1.0.0",
    lifespan=lifespan  # 添加 lifespan 管理器
)

import fastapi_cdn_host # 解决docs访问超时导致的空白网页问题
fastapi_cdn_host.patch_docs(app)

app.include_router(auth_api.router)
app.include_router(rbac_api.router)
app.include_router(kb_api.router)
app.include_router(audio_api.router)
app.include_router(audio_admin_api.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 本地开发可以先全开，线上再收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat",response_model=ChatResp)
async def chat(req: ChatReq,current_user = Depends(get_current_user)):
    require_permission(current_user, "kb.view_public")
    require_permission(current_user, "kb.view_internal")
    check_permission(current_user, "kb.view_public")
    check_permission(current_user, "kb.view_internal")


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
    current_user = Depends(get_current_user),
    delete_old_file: bool = Form(False),
    overwrite: bool = Form(False),
):
    """
    Upload a single document and upsert into Chroma.

    - Saves file to ./data/docs/
    - Loads & splits into chunks
    - Attaches visibility/doc_id metadata
    - Upserts into the configured Chroma collection
    """


    require_permission(current_user, "kb.manage_docs")
    check_permission(current_user, "kb.manage_docs")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    # visibility = (visibility or "public").strip().lower()
    visibility = normalize_visibility(visibility or "public")
    doc_id = (doc_id or f"doc-{uuid.uuid4().hex[:12]}").strip()

    existed = mysql_kb.get_kb_document(doc_id,is_deleted=False)
    if existed and not overwrite:
        raise HTTPException(status_code=409, detail=f"doc_id already exists: {doc_id}")

    # old_path里面放的是旧文档存放的路径
    old_path = existed["stored_path"] if existed else None

    # 1) 先把新文件保存下来
    suffix = Path(file.filename).suffix
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    save_path = DATA_DOCS_DIR / safe_name

    content = await file.read()  # 因为上传文件时间较长
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    save_path.write_bytes(content) # 新文件没问题，做写出操作

    # 2) 先解析新文件、切分出 chunks（确保新文件 OK）
    docs = load_single_file(save_path)
    if not docs:
        raise HTTPException(status_code=400, detail=f"Unsupported or empty file type: {suffix}")

    extra_meta = {
        "original_filename": file.filename,
        "stored_path": str(save_path),
        "uploader_user_id": current_user.id,
        "uploader_username": current_user.username,
        "uploaded_at": int(time.time()),
    }
    chunks = split_with_visibility(docs, visibility=visibility, doc_id=doc_id, extra_meta=extra_meta)
    # 程序到此处的时候，新文件已经彻底被分割并放好元数据

    # 3) 如果overwrite：现在再删旧的chroma chunks（此时新 chunks 已经准备好）
    if existed and overwrite:  # 旧文件要被覆盖，新文件也没问题，要彻底替换
        delete_by_doc_id(doc_id)

    # 4) 写入向量库
    vs = get_vs()
    vs.add_documents(chunks)

    # 5) 更新注册表——此处的注册表只是一个叫法，实际上就是mysql，和windwos的注册表无关
    chroma_cnt = count_by_doc_id(doc_id)

    mysql_kb.upsert_kb_document(
        doc_id=doc_id,
        original_filename=file.filename,
        stored_path=str(save_path),
        visibility=visibility,
        uploader_user_id=current_user.id,
        uploader_username=current_user.username,
        chunk_count=chroma_cnt,
    )

    # 6) overwrite 时可选删除旧文件（最后一步做）
    deleted_old_file = False
    if delete_old_file and old_path and old_path != str(save_path):
        try:
            p = Path(old_path)
            if p.exists() and p.is_file():  # p.is_file是担心对文件夹有影响
                p.unlink()  # unlink想像成为删除文件
                deleted_old_file = True
        except Exception:
            deleted_old_file = False

    return {
        "saved_as": str(save_path),
        "visibility": visibility,
        "doc_id": doc_id,
        "chunks": chroma_cnt,
        "overwrote": bool(existed and overwrite),
        "deleted_old_file": deleted_old_file,
    }


@app.post("/reindex")
def reindex(
    visibility_default: str = Form("public"),
    current_user = Depends(get_current_user)
):
    """
    Full rebuild of the collection from ./data/docs.

    WARNING: This deletes the current collection first.
    """

    require_permission(current_user, "kb.manage_docs")
    check_permission(current_user, "kb.manage_docs")


    visibility_default = (visibility_default or "public").strip().lower()

    # 1) Delete & recreate collection via chromadb client
    client = get_client()
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


