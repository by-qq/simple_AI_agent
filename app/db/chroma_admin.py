# 这个文件里面存放的更多是管理员要用的chroma的功能。比如删除，重新索引、更新可见性。
# 这些应该都是普通用户做不了的，或者说没有权限去做的。

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable
from app.config import settings
from app.db.vectorstore import get_client

client = lru_cache(maxsize=1)(get_client)
@lru_cache(maxsize=8)
def get_collection():
    return client.get_or_create_collection(settings.collection_name)

def delete_collection_by_name(name: str) -> None:
    client = get_client()
    client.delete_collection(name=name)

def delete_where_and_count(col, where: dict[str, Any]) -> int:
    try:
        got = col.get(where=where)
        ids = got.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return int(len(ids))
    except Exception:
        return 0

def delete_by_doc_id(doc_id: str) -> int:
    # chromadb特有的api用来按照doc_id删除其中的一个文档
    col = get_collection()  # 这个函数用于获取我向量数据中集合的名字
    return delete_where_and_count(col, {"doc_id": doc_id})


def get_ids_and_metadatas_by_doc_id(doc_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    col = get_collection()
    got = col.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    return list(ids), list(metas)


def count_by_doc_id(doc_id: str) -> int:
    ids, _ = get_ids_and_metadatas_by_doc_id(doc_id)
    return len(ids)


def update_visibility_by_doc_id(doc_id: str, visibility: str) -> int:
    # 更新文档的可见性
    col = get_collection()
    ids, metas = get_ids_and_metadatas_by_doc_id(doc_id)
    if not ids:
        return 0
    new_metas: list[dict[str, Any]] = []
    for m in metas:
        mm = dict(m or {})
        mm["visibility"] = visibility
        new_metas.append(mm)
    col.update(ids=ids, metadatas=new_metas)
    return len(ids)


def reset_kb_collection() -> None:
    delete_collection_by_name(settings.collection_name)
    get_collection.cache_clear()
    get_collection()

@lru_cache(maxsize=8)
def get_audio_collection():
    client = get_client()
    return client.get_or_create_collection(settings.audio_collection_name)

def get_ids_and_metadatas_by_audio_id(audio_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    col = get_audio_collection()
    got = col.get(where={"audio_id": audio_id}, include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    return list(ids), list(metas)

def delete_by_audio_id(audio_id: str) -> int:
    col = get_audio_collection()
    return delete_where_and_count(col, {"audio_id": audio_id})

def update_visibility_by_audio_id(audio_id: str, visibility: str) -> int:
    col = get_audio_collection()
    ids, metas = get_ids_and_metadatas_by_audio_id(audio_id)
    if not ids:
        return 0

    new_metas: list[dict[str, Any]] = []
    for m in metas:
        mm = dict(m or {})
        mm["visibility"] = visibility
        new_metas.append(mm)

    col.update(ids=ids, metadatas=new_metas)
    return len(ids)


def delete_many_audio_ids(audio_ids: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for aid in audio_ids:
        aid = (aid or "").strip()
        if not aid:
            continue
        out[aid] = delete_by_audio_id(aid)
    return out

def reset_audio_collection() -> None:
    delete_collection_by_name(settings.audio_collection_name)
    get_audio_collection.cache_clear()
    get_audio_collection()
