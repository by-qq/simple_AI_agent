from __future__ import annotations

from app.deps import get_vs
from app.ingestion.loader import load_single_file, split_with_visibility, load_docs, split_docs
from app.config import settings
import time
import uuid
from pathlib import Path
from typing import Optional
import chromadb

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.router_graph import router_graph
app = FastAPI(title="Enterprise KB Assistant")

DATA_DOCS_DIR = Path("./data/docs")
DATA_DOCS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS: dict[str, str] = {}       # 后期使用redis进行存储


# 这个类继承自BaseModel，转化成json
class ChatReq(BaseModel):
    text: str
    user_role: str = "public"
    requester: str = "anonymous"
    session_id: Optional[str] = None  # 通过这一行给大模型添加记忆

class ChatResp(BaseModel):
    answer: str


@app.post("/chat",response_model=ChatResp)
async def chat(req: ChatReq):
    # req.model_dump()，将请求对象转化为字典格式
    # 将字典数据输入到图中，之后就按照图定义的结构开始执行并返回最终结果
    # out = router_graph.invoke(req.model_dump())
    # return {"answer": out["answer"]}
    payload = req.model_dump()
    sid = payload.get("session_id")

    if sid and sid in SESSIONS:
        prev = SESSIONS[sid]
        merged = {**prev, **payload}
        merged["text"] = payload.get("text")
        payload = merged

    out = router_graph.invoke(payload)
    if sid:
        SESSIONS[sid] = {**payload, **out}

    return {"answer":out["answer"]}

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


